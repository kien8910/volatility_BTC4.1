from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .baselines import fit_har_qlike
from .config import Fold, MainPilotConfig, SPIKE_DIAGNOSTIC_FOLDS
from .data import MarketData, load_market_data, write_market_audit
from .event_aware_longtext_audit import (
    EventAwarePolicy,
    _fold_feature,
    evaluate_filter_on_silver_holdout,
    fit_event_aware_policy,
    load_event_aware_articles,
)
from .model import (
    CrossAttentionBlock,
    sinusoidal_encoding,
    trainable_parameter_count,
)
from .news import FilteredArticle
from .news_filter_labeling import _event_family_hint
from .news_representation_audit import (
    LAMBDA_SUM_GRID,
    _fit_probe,
    _prediction_frame,
    _target_rv,
)
from .pipeline import _block_dates
from .point_in_time_gate_diagnostic import (
    _market_gate_features,
    refined_event_decision,
    refined_policy,
)
from .preprocess import FoldNewsFeatures
from .spike_diagnostic import (
    _annotate_predictions,
    _diagnostic_metrics,
    _pooled_metrics,
)
from .training import predict_dataset, train_model
from .utils import (
    ensure_finite,
    file_fingerprint,
    hash_arrays,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)
from .vector_integration_diagnostic import (
    EVENT_FAMILIES,
    MARKET_QUERY_NAMES,
    HarVectorCrossAttention,
    _build_transformer_datasets,
    _eligible_dates,
    _embed_articles,
    _feature_matrix as _v1_feature_matrix,
    _linear_feature_names as _v1_linear_feature_names,
    _set_dates,
    _validate_vector_scheduler,
    build_core_event_prototype_features,
)


PROFILE = "development-slow-transformer-v2-diagnostic"
SPIKE_QUANTILE = 0.90
BLEND_ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
DEEP_CANDIDATES = (
    "slow_calendar_control",
    "fast_calendar_control",
    "slow_update_tokens",
    "slow_update_multiquery",
    "slow_update_multiquery_gated",
)
CANDIDATES = (
    "har_qlike",
    "finbert_normal",
    "core_centered_event_prototypes",
    *DEEP_CANDIDATES,
    "finbert_normal_slow_blend",
)
UPDATE_AUXILIARY_SUFFIXES = (
    "__count_surprise_365d",
    "__log1p_source_count",
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    output = np.zeros_like(values, dtype=np.float64)
    valid = norms[:, 0] > 0
    output[valid] = values[valid] / norms[valid]
    return output


def build_core_centered_directional_features(
    articles: list[FilteredArticle],
    semantics: np.ndarray,
    dates: pd.DatetimeIndex,
    fold: Fold,
    causal_event_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Core-center BGE and score each family against its strongest alternative."""
    if len(articles) != len(semantics):
        raise ValueError("Article/vector length mismatch")
    core_start = pd.Timestamp(fold.core_start, tz="UTC")
    core_end_exclusive = pd.Timestamp(fold.core_end, tz="UTC") + pd.Timedelta(
        days=1
    )
    core_positions = [
        index
        for index, article in enumerate(articles)
        if core_start <= article.timestamp < core_end_exclusive
    ]
    if not core_positions:
        raise ValueError(f"{fold.name} has no core articles for centering")
    core_mean = np.mean(semantics[core_positions], axis=0)
    centered = semantics - core_mean
    centered_unit = _unit_rows(centered)
    families = [
        _event_family_hint(article.title, article.cleaned_text[:500])
        for article in articles
    ]
    prototypes: dict[str, np.ndarray] = {}
    prototype_counts: dict[str, int] = {}
    for family in EVENT_FAMILIES:
        positions = [
            index
            for index in core_positions
            if families[index] == family
            and float(np.linalg.norm(centered_unit[index])) > 0
        ]
        prototype_counts[family] = len(positions)
        if positions:
            mean = np.mean(centered_unit[positions], axis=0)
            norm = float(np.linalg.norm(mean))
            prototypes[family] = mean / norm if norm > 0 else np.zeros_like(mean)
        else:
            prototypes[family] = np.zeros(semantics.shape[1], dtype=np.float64)

    daily_vectors: dict[pd.Timestamp, list[np.ndarray]] = {}
    available = set(dates)
    for article, vector in zip(articles, centered_unit):
        day = article.timestamp.floor("D")
        if day in available and float(np.linalg.norm(vector)) > 0:
            daily_vectors.setdefault(day, []).append(vector)
    output = pd.DataFrame(
        0.0,
        index=dates,
        columns=[
            f"{family}__centered_directional_margin"
            for family in EVENT_FAMILIES
        ],
    )
    for day, vectors in daily_vectors.items():
        centroid = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0:
            continue
        centroid /= norm
        cosine = {
            family: float(np.dot(centroid, prototypes[family]))
            if prototype_counts[family] > 0
            else 0.0
            for family in EVENT_FAMILIES
        }
        for family in EVENT_FAMILIES:
            alternatives = [
                value
                for other, value in cosine.items()
                if other != family and prototype_counts[other] > 0
            ]
            strongest_other = max(alternatives) if alternatives else 0.0
            output.loc[
                day, f"{family}__centered_directional_margin"
            ] = (cosine[family] - strongest_other)

    auxiliary_columns = [
        column
        for column in causal_event_features.columns
        if column.endswith(UPDATE_AUXILIARY_SUFFIXES)
    ]
    output = pd.concat(
        [output, causal_event_features[auxiliary_columns].reindex(dates)],
        axis=1,
    )
    if output.isna().any().any():
        raise ValueError("Directional event features have missing dates")
    ensure_finite("core-centered directional features", output.to_numpy())
    margin = output.filter(like="__centered_directional_margin")
    nonzero = margin.to_numpy()[np.any(margin.to_numpy() != 0, axis=1)]
    metadata = {
        "families": list(EVENT_FAMILIES),
        "prototype_counts": prototype_counts,
        "center_fit_start": fold.core_start,
        "center_fit_end_inclusive": fold.core_end,
        "prototype_fit_scope": "core articles only",
        "score": "cosine(centered_daily, family) - max cosine(other families)",
        "target_outcomes_used": False,
        "margin_nonzero_days": int(len(nonzero)),
        "margin_mean_abs": (
            float(np.mean(np.abs(nonzero))) if len(nonzero) else 0.0
        ),
        "margin_std": float(np.std(nonzero)) if len(nonzero) else 0.0,
        "core_mean_hash": hash_arrays({"core_embedding_mean": core_mean}),
    }
    return output, metadata


def _directional_probe_matrix(
    dates: list[pd.Timestamp],
    news: FoldNewsFeatures,
    directional: pd.DataFrame,
) -> np.ndarray:
    rows = []
    for target in dates:
        day = target - pd.Timedelta(days=1)
        index = int(news.dates.get_loc(day))
        rows.append(
            np.concatenate(
                [
                    news.sentiment_slow[index],
                    news.sentiment_fast[index],
                    directional.loc[day].to_numpy(dtype=np.float64),
                ]
            )
        )
    matrix = np.stack(rows)
    ensure_finite("core-centered prototype probe", matrix)
    return matrix


class MaskedPreNormSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        if key_padding_mask.shape != x.shape[:2]:
            raise ValueError("Self-attention padding mask shape mismatch")
        if bool(key_padding_mask.all(dim=1).any()):
            raise AssertionError("Every context token is masked")
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attended)
        return x + self.dropout2(self.ffn(self.norm2(x)))


class SlowUpdateDataset(Dataset):
    def __init__(
        self,
        dates: list[pd.Timestamp],
        market_queries: np.ndarray,
        gate_features: np.ndarray,
        update_daily: np.ndarray,
        update_available: np.ndarray,
        level_daily: np.ndarray,
        news_dates: pd.DatetimeIndex,
        har_anchor_log: np.ndarray,
        true_rv: np.ndarray,
        lookback: int,
    ):
        if not (
            len(dates)
            == len(market_queries)
            == len(gate_features)
            == len(har_anchor_log)
            == len(true_rv)
        ):
            raise ValueError("Slow-update dataset length mismatch")
        lookup = {date: index for index, date in enumerate(news_dates)}
        windows = []
        masks = []
        levels = []
        for target in dates:
            positions = [
                lookup[target - pd.Timedelta(days=lag)]
                for lag in range(lookback, 0, -1)
            ]
            windows.append(update_daily[positions])
            masks.append(~update_available[positions])
            levels.append(level_daily[positions[-1]])
        self.target_dates = [date.strftime("%Y-%m-%d") for date in dates]
        self.market_queries = torch.tensor(
            market_queries, dtype=torch.float32
        )
        self.gate_features = torch.tensor(gate_features, dtype=torch.float32)
        self.update_tokens = torch.tensor(
            np.stack(windows), dtype=torch.float32
        )
        self.update_padding_mask = torch.tensor(
            np.stack(masks), dtype=torch.bool
        )
        self.slow_level = torch.tensor(
            np.stack(levels), dtype=torch.float32
        )
        self.har_anchor_log = torch.tensor(
            har_anchor_log, dtype=torch.float64
        )
        self.true_rv = torch.tensor(true_rv, dtype=torch.float64)
        self.true_log_rv = torch.log(self.true_rv)

    def __len__(self) -> int:
        return len(self.target_dates)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "target_date": self.target_dates[index],
            "market_query": self.market_queries[index],
            "gate_features": self.gate_features[index],
            "update_tokens": self.update_tokens[index],
            "update_padding_mask": self.update_padding_mask[index],
            "slow_level": self.slow_level[index],
            "har_anchor_log_rv": self.har_anchor_log[index],
            "true_rv": self.true_rv[index],
            "true_log_rv": self.true_log_rv[index],
        }


class SlowUpdateTransformer(nn.Module):
    DAILY_QUERY_INDICES = (0, 5, 6)
    WEEKLY_QUERY_INDICES = (1, 3, 6)
    MONTHLY_QUERY_INDICES = (2, 4, 6)

    def __init__(
        self,
        config: MainPilotConfig,
        update_dim: int,
        level_dim: int,
        gate_dim: int,
        variant: Literal[
            "slow_update_tokens",
            "slow_update_multiquery",
            "slow_update_multiquery_gated",
        ],
    ):
        super().__init__()
        if variant not in {
            "slow_update_tokens",
            "slow_update_multiquery",
            "slow_update_multiquery_gated",
        }:
            raise ValueError(f"Unsupported slow Transformer variant: {variant}")
        self.variant = variant
        self.multiquery = variant != "slow_update_tokens"
        self.gated = variant == "slow_update_multiquery_gated"
        self.d_model = config.d_model
        d_model = config.d_model
        self.update_projection = nn.Linear(update_dim, d_model)
        self.level_projection = nn.Linear(level_dim, d_model)
        self.context_type_embedding = nn.Embedding(3, d_model)
        self.null_news_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_news_token, std=0.02)
        self.news_blocks = nn.ModuleList(
            [
                MaskedPreNormSelfAttentionBlock(
                    d_model,
                    config.attention_heads,
                    config.ffn_dim,
                    config.dropout,
                )
                for _ in range(config.news_layers)
            ]
        )
        if self.multiquery:
            self.query_projections = nn.ModuleList(
                [nn.Linear(3, d_model) for _ in range(3)]
            )
            self.query_type_embedding = nn.Embedding(3, d_model)
            self.query_pool_gate = nn.Linear(d_model, 1)
            self.market_projection = None
        else:
            self.market_projection = nn.Linear(
                len(MARKET_QUERY_NAMES), d_model
            )
            self.query_projections = nn.ModuleList()
            self.query_type_embedding = None
            self.query_pool_gate = None
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model,
                    config.attention_heads,
                    config.ffn_dim,
                    config.dropout,
                )
                for _ in range(config.cross_layers)
            ]
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )
        output = self.correction_head[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.correction_gate: nn.Module | None = (
            nn.Sequential(
                nn.Linear(d_model + gate_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            if self.gated
            else None
        )

    def _queries(self, market: torch.Tensor) -> torch.Tensor:
        if not self.multiquery:
            assert self.market_projection is not None
            return self.market_projection(market).unsqueeze(1)
        assert self.query_type_embedding is not None
        index_sets = (
            self.DAILY_QUERY_INDICES,
            self.WEEKLY_QUERY_INDICES,
            self.MONTHLY_QUERY_INDICES,
        )
        queries = []
        for query_index, (projection, indices) in enumerate(
            zip(self.query_projections, index_sets)
        ):
            selected = market[:, list(indices)]
            queries.append(
                projection(selected)
                + self.query_type_embedding.weight[query_index]
            )
        return torch.stack(queries, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw_updates = batch["update_tokens"]
        batch_size, token_count, _ = raw_updates.shape
        update = self.update_projection(raw_updates)
        update = (
            update
            + sinusoidal_encoding(
                token_count, self.d_model, raw_updates.device
            )[None, :, :]
            + self.context_type_embedding.weight[0]
        )
        level = (
            self.level_projection(batch["slow_level"]).unsqueeze(1)
            + self.context_type_embedding.weight[1]
        )
        null = (
            self.null_news_token.expand(batch_size, -1, -1)
            + self.context_type_embedding.weight[2]
        )
        context = torch.cat([update, level, null], dim=1)
        update_mask = batch["update_padding_mask"].bool()
        unmasked = torch.zeros(
            (batch_size, 2), dtype=torch.bool, device=raw_updates.device
        )
        context_mask = torch.cat([update_mask, unmasked], dim=1)
        if bool(context_mask[:, -2:].any()):
            raise AssertionError("Slow-level and null tokens must be unmasked")
        for block in self.news_blocks:
            context = block(context, context_mask)
        queries = self._queries(batch["market_query"])
        for block in self.cross_blocks:
            queries = block(queries, context, context_mask)
        if self.multiquery:
            assert self.query_pool_gate is not None
            weights = torch.softmax(
                self.query_pool_gate(queries).squeeze(-1), dim=1
            )
            pooled = torch.sum(queries * weights.unsqueeze(-1), dim=1)
        else:
            pooled = queries[:, 0]
        raw_delta = self.correction_head(pooled).squeeze(-1)
        if self.correction_gate is not None:
            gate = self.correction_gate(
                torch.cat([pooled, batch["gate_features"]], dim=-1)
            ).squeeze(-1)
        else:
            gate = torch.ones_like(raw_delta)
        delta = (gate * raw_delta).double()
        anchor = batch["har_anchor_log_rv"].double()
        predicted_log = anchor + delta
        return {
            "har_anchor_log_rv": anchor,
            "delta_log_rv": delta,
            "correction_gate": gate,
            "predicted_log_rv": predicted_log,
            "predicted_rv": torch.exp(predicted_log),
        }


def _prepare_update_daily(
    news: FoldNewsFeatures,
    causal_event: pd.DataFrame,
    fold: Fold,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
    list[str],
    list[str],
]:
    semantic_delta = np.vstack(
        [np.zeros((1, news.semantic_slow.shape[1])), np.diff(news.semantic_slow, axis=0)]
    )
    sentiment_delta = np.vstack(
        [
            np.zeros((1, news.sentiment_slow.shape[1])),
            np.diff(news.sentiment_slow, axis=0),
        ]
    )
    auxiliary_columns = [
        column
        for column in causal_event.columns
        if column.endswith(UPDATE_AUXILIARY_SUFFIXES)
    ]
    auxiliary = causal_event[auxiliary_columns].reindex(news.dates)
    if auxiliary.isna().any().any():
        raise ValueError("Causal event auxiliary dates do not align")
    update_available = news.daily_scalars[:, -1] < 0.5
    update_raw = np.column_stack(
        [
            semantic_delta,
            sentiment_delta,
            news.daily_scalars,
            auxiliary.to_numpy(dtype=np.float64),
        ]
    )
    level_raw = np.column_stack(
        [
            news.semantic_slow,
            news.sentiment_slow,
            news.daily_scalars,
        ]
    )
    core_mask = (
        (news.dates >= pd.Timestamp(fold.core_start, tz="UTC"))
        & (news.dates <= pd.Timestamp(fold.core_end, tz="UTC"))
    )
    core_updates = core_mask & update_available
    if int(core_updates.sum()) < 30:
        raise ValueError(f"{fold.name} has fewer than 30 core news updates")
    update_scaler = StandardScaler().fit(update_raw[core_updates])
    level_scaler = StandardScaler().fit(level_raw[core_mask])
    update_scaled = update_scaler.transform(update_raw)
    update_scaled[~update_available] = 0.0
    level_scaled = level_scaler.transform(level_raw)
    ensure_finite("scaled slow update tokens", update_scaled)
    ensure_finite("scaled slow level tokens", level_scaled)
    update_names = [
        *[
            f"delta_semantic_slow_pc_{i:02d}"
            for i in range(1, news.semantic_slow.shape[1] + 1)
        ],
        *[
            f"delta_sentiment_slow_{label}"
            for label in ("positive", "negative", "neutral")
        ],
        *[f"daily_scalar_{i:02d}" for i in range(news.daily_scalars.shape[1])],
        *auxiliary_columns,
    ]
    level_names = [
        *[
            f"semantic_slow_pc_{i:02d}"
            for i in range(1, news.semantic_slow.shape[1] + 1)
        ],
        *[
            f"sentiment_slow_{label}"
            for label in ("positive", "negative", "neutral")
        ],
        *[f"daily_scalar_{i:02d}" for i in range(news.daily_scalars.shape[1])],
    ]
    return (
        update_scaled,
        update_available,
        level_scaled,
        update_scaler,
        level_scaler,
        update_names,
        level_names,
    )


def _gate_feature_row(
    market: MarketData,
    news: FoldNewsFeatures,
    target: pd.Timestamp,
) -> np.ndarray:
    day = target - pd.Timedelta(days=1)
    index = int(news.dates.get_loc(day))
    start = index - 29
    if start < 0:
        raise ValueError(f"Insufficient gate history for {target}")
    update_count = float(np.sum(news.daily_scalars[start : index + 1, -1] < 0.5))
    semantic_change = (
        news.semantic_slow[index] - news.semantic_slow[index - 1]
    )
    sentiment_change = (
        news.sentiment_slow[index] - news.sentiment_slow[index - 1]
    )
    return np.concatenate(
        [
            _market_gate_features(market, target),
            news.daily_scalars[index].astype(np.float64),
            np.asarray(
                [
                    np.linalg.norm(semantic_change),
                    np.linalg.norm(sentiment_change),
                    update_count / 30.0,
                ],
                dtype=np.float64,
            ),
        ]
    )


def _build_update_datasets(
    market: MarketData,
    news: FoldNewsFeatures,
    causal_event: pd.DataFrame,
    fold: Fold,
    core_dates: list[pd.Timestamp],
    validation_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    anchors: tuple[np.ndarray, np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: MainPilotConfig,
) -> tuple[
    SlowUpdateDataset,
    SlowUpdateDataset,
    SlowUpdateDataset,
    dict[str, Any],
]:
    (
        update_daily,
        update_available,
        level_daily,
        update_scaler,
        level_scaler,
        update_names,
        level_names,
    ) = _prepare_update_daily(news, causal_event, fold)
    date_blocks = (core_dates, validation_dates, test_dates)
    query_core_raw = np.stack(
        [_market_gate_features(market, date) for date in core_dates]
    )
    query_scaler = StandardScaler().fit(query_core_raw)
    query_blocks = tuple(
        query_scaler.transform(
            np.stack([_market_gate_features(market, date) for date in dates])
        )
        for dates in date_blocks
    )
    gate_core_raw = np.stack(
        [_gate_feature_row(market, news, date) for date in core_dates]
    )
    gate_scaler = StandardScaler().fit(gate_core_raw)
    gate_blocks = tuple(
        gate_scaler.transform(
            np.stack([_gate_feature_row(market, news, date) for date in dates])
        )
        for dates in date_blocks
    )
    datasets = tuple(
        SlowUpdateDataset(
            dates,
            query,
            gate,
            update_daily,
            update_available,
            level_daily,
            news.dates,
            anchor,
            target,
            config.news_lookback_days,
        )
        for dates, query, gate, anchor, target in zip(
            date_blocks, query_blocks, gate_blocks, anchors, targets
        )
    )
    preprocessor_hash = hash_arrays(
        {
            "news_preprocessor_hash": np.frombuffer(
                news.preprocessor_hash.encode("utf-8"), dtype=np.uint8
            ),
            "update_mean": update_scaler.mean_,
            "update_scale": update_scaler.scale_,
            "level_mean": level_scaler.mean_,
            "level_scale": level_scaler.scale_,
            "query_mean": query_scaler.mean_,
            "query_scale": query_scaler.scale_,
            "gate_mean": gate_scaler.mean_,
            "gate_scale": gate_scaler.scale_,
        }
    )
    metadata = {
        "preprocessor_hash": preprocessor_hash,
        "update_dim": int(update_daily.shape[1]),
        "level_dim": int(level_daily.shape[1]),
        "gate_dim": int(gate_core_raw.shape[1]),
        "update_feature_names": update_names,
        "level_feature_names": level_names,
        "gate_features": [
            *MARKET_QUERY_NAMES,
            *[
                f"t_minus_1_daily_scalar_{i:02d}"
                for i in range(news.daily_scalars.shape[1])
            ],
            "semantic_slow_update_l2",
            "sentiment_slow_update_l2",
            "news_update_fraction_30d",
        ],
        "news_lookback_days": config.news_lookback_days,
        "update_mask_rule": "Mask calendar days with no newly retained news",
        "always_unmasked_tokens": ["t_minus_1_slow_level", "learned_null_news"],
        "fit_scope": "Every scaler is fit on fold core only",
        "information_cutoff": "t-1",
    }
    return (*datasets, metadata)


def select_log_blend(
    validation_finbert: pd.DataFrame,
    validation_slow: pd.DataFrame,
    test_finbert: pd.DataFrame,
    test_slow: pd.DataFrame,
    alpha_grid: tuple[float, ...] = BLEND_ALPHA_GRID,
    min_delta: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if not (
        len(validation_finbert)
        == len(validation_slow)
        and len(test_finbert) == len(test_slow)
    ):
        raise ValueError("Blend prediction support mismatch")
    rows = []
    best_alpha: float | None = None
    best_qlike = float("inf")
    for alpha in alpha_grid:
        predicted_log = (
            alpha * validation_slow["predicted_log_rv"].to_numpy()
            + (1.0 - alpha)
            * validation_finbert["predicted_log_rv"].to_numpy()
        )
        predicted = np.exp(predicted_log)
        ratio = validation_slow["true_rv"].to_numpy() / predicted
        qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
        rows.append({"alpha_slow": alpha, "validation_qlike": qlike})
        better = qlike < best_qlike - min_delta
        tied_prefer_slow = (
            abs(qlike - best_qlike) <= min_delta
            and (best_alpha is None or alpha > best_alpha)
        )
        if better or tied_prefer_slow:
            best_alpha = alpha
            best_qlike = qlike
    if best_alpha is None:
        raise RuntimeError("No finite FinBERT/slow blend was selected")

    def blend(
        finbert: pd.DataFrame, slow: pd.DataFrame
    ) -> pd.DataFrame:
        output = slow[
            ["target_date", "true_rv", "true_log_rv"]
        ].copy()
        output["predicted_log_rv"] = (
            best_alpha * slow["predicted_log_rv"].to_numpy()
            + (1.0 - best_alpha)
            * finbert["predicted_log_rv"].to_numpy()
        )
        output["predicted_rv"] = np.exp(output["predicted_log_rv"])
        output["har_anchor_log_rv"] = slow["har_anchor_log_rv"].to_numpy()
        output["delta_log_rv"] = (
            output["predicted_log_rv"] - output["har_anchor_log_rv"]
        )
        ensure_finite("FinBERT/slow blend predictions", output["predicted_rv"])
        return output

    return (
        blend(validation_finbert, validation_slow),
        blend(test_finbert, test_slow),
        {
            "alpha_slow": best_alpha,
            "alpha_finbert": 1.0 - best_alpha,
            "validation_qlike": best_qlike,
            "alpha_grid": list(alpha_grid),
            "selection_scope": "validation only",
            "tie_break": "prefer larger slow weight within min_delta",
        },
        pd.DataFrame(rows),
    )


@torch.no_grad()
def _predict_gate_values(
    model: nn.Module,
    dataset: Dataset,
    config: MainPilotConfig,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=config.physical_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    model.eval()
    values: list[np.ndarray] = []
    for raw_batch in loader:
        batch = {
            key: (
                value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in raw_batch.items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        gate = output.get("correction_gate")
        if gate is None:
            raise RuntimeError("Gated model did not return correction_gate")
        values.append(gate.float().cpu().numpy())
    result = np.concatenate(values)
    ensure_finite("point-in-time correction gate", result)
    if np.any((result < 0) | (result > 1)):
        raise FloatingPointError("Correction gate is outside [0, 1]")
    return result


def _safe_pooled_metrics(
    frame: pd.DataFrame, model_name: str
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        metrics = _pooled_metrics(frame, model_name)
    if not frame["is_spike"].astype(bool).any():
        metrics.update(
            {
                "spike_qlike": None,
                "spike_capture_rate": None,
                "mean_spike_predicted_to_true_rv_ratio": None,
            }
        )
    return metrics


def _assert_epoch_zero_har(
    model: nn.Module,
    batch: dict[str, Any],
    variant: str,
    seed: int,
) -> None:
    """Check the zero correction without perturbing the training RNG stream."""
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            initial = model(batch)
    finally:
        model.train(was_training)
    if not torch.equal(
        initial["predicted_log_rv"].cpu(),
        initial["har_anchor_log_rv"].cpu(),
    ):
        raise AssertionError(f"{variant} is not exactly HAR at epoch zero")
    seed_everything(seed)


def _screen(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    min_delta: float,
) -> dict[str, Any]:
    def finite(value: Any) -> bool:
        return value is not None and bool(np.isfinite(value))

    def better(candidate: Any, anchor: Any) -> bool:
        return (
            finite(candidate)
            and finite(anchor)
            and float(candidate) < float(anchor) - min_delta
        )

    anchor_fold = fold_metrics[
        fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    anchor_pool = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    candidates: dict[str, Any] = {}
    for name in CANDIDATES[1:]:
        current_fold = fold_metrics[
            fold_metrics["model"] == name
        ].set_index("fold")
        current_pool = pooled_metrics[
            pooled_metrics["model"] == name
        ]
        if current_fold.empty or current_pool.empty:
            candidates[name] = {
                "folds_compared": 0,
                "required_fold_wins": int(len(anchor_fold)),
                "passes_predeclared_screen": False,
                "ineligible_reason": (
                    "missing finite OOS predictions after numerical failure"
                ),
            }
            continue
        current_pool = current_pool.iloc[0]
        common = sorted(set(anchor_fold.index) & set(current_fold.index))
        required_wins = min(3, len(common))
        all_folds_complete = len(common) == len(anchor_fold)

        def wins(metric: str) -> int:
            return sum(
                better(
                    current_fold.loc[fold, metric],
                    anchor_fold.loc[fold, metric],
                )
                for fold in common
            )

        overall_wins = wins("mean_qlike")
        spike_wins = wins("spike_qlike")
        pooled_overall_better = bool(
            current_pool["mean_qlike"]
            < anchor_pool["mean_qlike"] - min_delta
        )
        pooled_spike_better = better(
            current_pool["spike_qlike"], anchor_pool["spike_qlike"]
        )
        normal_guard = bool(
            current_pool["normal_qlike"]
            <= 1.01 * anchor_pool["normal_qlike"]
        )
        candidates[name] = {
            "folds_compared": len(common),
            "required_fold_wins": required_wins,
            "all_folds_complete": all_folds_complete,
            "overall_fold_wins": overall_wins,
            "normal_fold_wins": wins("normal_qlike"),
            "spike_fold_wins": spike_wins,
            "pooled_overall_delta_vs_har": float(
                current_pool["mean_qlike"] - anchor_pool["mean_qlike"]
            ),
            "pooled_normal_delta_vs_har": float(
                current_pool["normal_qlike"] - anchor_pool["normal_qlike"]
            ),
            "pooled_spike_delta_vs_har": (
                float(
                    current_pool["spike_qlike"]
                    - anchor_pool["spike_qlike"]
                )
                if finite(current_pool["spike_qlike"])
                and finite(anchor_pool["spike_qlike"])
                else None
            ),
            "normal_guard_at_most_one_percent_worse": normal_guard,
            "passes_predeclared_screen": bool(
                overall_wins >= required_wins
                and spike_wins >= required_wins
                and all_folds_complete
                and pooled_overall_better
                and pooled_spike_better
                and normal_guard
            ),
        }
    return {
        "rule": (
            "Pass only with overall and spike QLIKE wins in at least three "
            "folds (or every fold when fewer than three are run), improved "
            "pooled overall and spike QLIKE, and pooled normal QLIKE no more "
            "than 1% worse than HAR."
        ),
        "min_delta": min_delta,
        "candidates": candidates,
        "statistical_claim": "None; development-only screen.",
    }


def _influence_sensitivity(
    pooled: dict[str, list[pd.DataFrame]],
) -> list[dict[str, Any]]:
    anchor = pd.concat(pooled["har_qlike"], ignore_index=True)
    anchor["_key"] = (
        anchor["fold"].astype(str) + "|" + anchor["target_date"].astype(str)
    )
    rows = []
    for drop_n in (0, 1, 2, 3):
        excluded_keys = (
            set(
                anchor.loc[anchor["is_spike"]]
                .nlargest(drop_n, "qlike")["_key"]
            )
            if drop_n
            else set()
        )
        for name, frames in pooled.items():
            if not frames:
                continue
            current = pd.concat(frames, ignore_index=True)
            current_keys = (
                current["fold"].astype(str)
                + "|"
                + current["target_date"].astype(str)
            )
            keep = ~current_keys.isin(excluded_keys)
            rows.append(
                {
                    "model": name,
                    "excluded_top_har_spike_days": drop_n,
                    "n": int(keep.sum()),
                    "mean_qlike": float(current.loc[keep, "qlike"].mean()),
                    "diagnostic_only_not_for_selection": True,
                }
            )
    return rows


def _run_folds(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: MarketData,
    daily: pd.DataFrame,
    articles: list[FilteredArticle],
    semantics: np.ndarray,
    output_dir: Path,
    folds: tuple[Fold, ...],
    device: torch.device,
    horizon_epochs: int,
    scheduler_path: Path,
    scheduler_hash: str,
    resume: bool,
    lambda_grid: tuple[float, ...] = LAMBDA_SUM_GRID,
    smoke: bool = False,
    selection_mode: Literal[
        "development_screen",
        "single_fold_evaluation",
        "confirmatory_evaluation",
    ] = "development_screen",
) -> dict[str, Any]:
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    features_dir = output_dir / "features"
    checkpoints_dir = output_dir / "checkpoints"
    for directory in (
        predictions_dir,
        metrics_dir,
        features_dir,
        checkpoints_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[pd.DataFrame]] = {
        name: [] for name in CANDIDATES
    }
    fold_metadata: dict[str, Any] = {}
    numerical_failures: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "SLOW V2 FOLD %d/%d | %s build t-1 information set",
            fold_index,
            len(folds),
            fold.name,
        )
        core_dates, validation_dates, test_dates = _block_dates(
            market,
            fold,
            output_dir,
            logger,
            include_test=True,
            config=config,
            sampling_rule="har_text",
        )
        core_dates = _eligible_dates(
            core_dates, daily.index, config.news_lookback_days
        )
        validation_dates = _eligible_dates(
            validation_dates, daily.index, config.news_lookback_days
        )
        test_dates = _eligible_dates(
            test_dates, daily.index, config.news_lookback_days
        )
        if not smoke and len(validation_dates) < 60:
            raise RuntimeError(
                f"{fold.name} has only {len(validation_dates)} validation targets"
            )
        news = _fold_feature(daily, fold, config, logger)
        causal_event, causal_event_metadata = (
            build_core_event_prototype_features(
                articles, semantics, news.dates, fold
            )
        )
        directional, directional_metadata = (
            build_core_centered_directional_features(
                articles,
                semantics,
                news.dates,
                fold,
                causal_event,
            )
        )
        causal_event.to_csv(
            features_dir / f"{fold.name}_causal_event_features.csv"
        )
        directional.to_csv(
            features_dir
            / f"{fold.name}_core_centered_directional_features.csv"
        )
        har = fit_har_qlike(market, core_dates)
        anchors = (
            har.predict_log_rv(market, core_dates),
            har.predict_log_rv(market, validation_dates),
            har.predict_log_rv(market, test_dates),
        )
        targets = (
            _target_rv(market, core_dates),
            _target_rv(market, validation_dates),
            _target_rv(market, test_dates),
        )
        spike_threshold = float(np.quantile(targets[0], SPIKE_QUANTILE))
        har_validation = _prediction_frame(
            validation_dates,
            targets[1],
            anchors[1],
            np.zeros(len(validation_dates)),
        )
        har_test = _prediction_frame(
            test_dates,
            targets[2],
            anchors[2],
            np.zeros(len(test_dates)),
        )
        har_validation.to_csv(
            predictions_dir / f"{fold.name}_har_qlike_validation.csv",
            index=False,
        )
        har_annotated = _annotate_predictions(
            har_test, fold.name, "har_qlike", spike_threshold
        )
        har_annotated.to_csv(
            predictions_dir / f"{fold.name}_har_qlike.csv", index=False
        )
        pooled["har_qlike"].append(har_annotated)
        fold_rows.append(
            {
                "fold": fold.name,
                **_diagnostic_metrics(
                    har_annotated, spike_threshold, "har_qlike"
                ),
                "best_epoch": None,
                "early_stopped": None,
                "best_validation_qlike": float(
                    np.mean(
                        targets[1] / np.exp(anchors[1])
                        - np.log(targets[1] / np.exp(anchors[1]))
                        - 1.0
                    )
                ),
                "correction_selected": False,
            }
        )
        metadata: dict[str, Any] = {
            "sample_counts": {
                "core": len(core_dates),
                "validation": len(validation_dates),
                "test": len(test_dates),
            },
            "har": har.metadata(),
            "spike_threshold_core_p90": spike_threshold,
            "news_preprocessor": news.metadata,
            "causal_event": causal_event_metadata,
            "core_centered_directional": directional_metadata,
            "linear": {},
            "transformer": {},
            "blend": {},
        }

        linear_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for name in ("finbert_normal", "core_centered_event_prototypes"):
            logger.info("SLOW V2 LINEAR | fold=%s candidate=%s", fold.name, name)
            if name == "finbert_normal":
                matrices = tuple(
                    _v1_feature_matrix(
                        dates, news, causal_event, "finbert_normal"
                    )
                    for dates in (
                        core_dates,
                        validation_dates,
                        test_dates,
                    )
                )
                feature_names = _v1_linear_feature_names(
                    "finbert_normal", causal_event
                )
            else:
                matrices = tuple(
                    _directional_probe_matrix(dates, news, directional)
                    for dates in (
                        core_dates,
                        validation_dates,
                        test_dates,
                    )
                )
                feature_names = [
                    *[
                        f"finbert_{state}_{label}"
                        for state in ("slow", "fast")
                        for label in ("positive", "negative", "neutral")
                    ],
                    *directional.columns.tolist(),
                ]
            validation_frame, test_frame, candidate_meta, grid = _fit_probe(
                name,
                *matrices,
                *targets,
                *anchors,
                feature_names,
                lambda_grid,
                config.min_delta,
            )
            validation_frame = _set_dates(
                validation_frame, validation_dates
            )
            test_frame = _set_dates(test_frame, test_dates)
            validation_frame.to_csv(
                predictions_dir / f"{fold.name}_{name}_validation.csv",
                index=False,
            )
            annotated = _annotate_predictions(
                test_frame, fold.name, name, spike_threshold
            )
            annotated.to_csv(
                predictions_dir / f"{fold.name}_{name}.csv", index=False
            )
            grid.to_csv(
                metrics_dir / f"{fold.name}_{name}_lambda_grid.csv",
                index=False,
            )
            pooled[name].append(annotated)
            linear_frames[name] = {
                "validation": validation_frame,
                "test": test_frame,
            }
            fold_rows.append(
                {
                    "fold": fold.name,
                    **_diagnostic_metrics(
                        annotated, spike_threshold, name
                    ),
                    "best_epoch": None,
                    "early_stopped": None,
                    "best_validation_qlike": candidate_meta[
                        "candidate_validation_qlike"
                    ],
                    "correction_selected": candidate_meta[
                        "correction_selected"
                    ],
                }
            )
            metadata["linear"][name] = candidate_meta

        control_sets = _build_transformer_datasets(
            market,
            news,
            causal_event,
            fold,
            "slow",
            core_dates,
            validation_dates,
            test_dates,
            anchors,
            targets,
            config,
        )
        fast_control_sets = _build_transformer_datasets(
            market,
            news,
            causal_event,
            fold,
            "fast",
            core_dates,
            validation_dates,
            test_dates,
            anchors,
            targets,
            config,
        )
        update_sets = _build_update_datasets(
            market,
            news,
            causal_event,
            fold,
            core_dates,
            validation_dates,
            test_dates,
            anchors,
            targets,
            config,
        )
        gated_frames: dict[str, pd.DataFrame] = {}
        for name in DEEP_CANDIDATES:
            logger.info(
                "SLOW V2 TRANSFORMER | fold=%s candidate=%s",
                fold.name,
                name,
            )
            if name in {"slow_calendar_control", "fast_calendar_control"}:
                state = "fast" if name.startswith("fast_") else "slow"
                selected_sets = (
                    fast_control_sets if state == "fast" else control_sets
                )
                core_set, validation_set, test_set, prep_meta = selected_sets
                seed_everything(config.seed)
                model: nn.Module = HarVectorCrossAttention(
                    config, prep_meta["token_dim"], state
                )
                model.variant = name
            else:
                core_set, validation_set, test_set, prep_meta = update_sets
                seed_everything(config.seed)
                model = SlowUpdateTransformer(
                    config,
                    prep_meta["update_dim"],
                    prep_meta["level_dim"],
                    prep_meta["gate_dim"],
                    name,
                )
            model = model.to(device)
            count = trainable_parameter_count(model)
            if count > config.parameter_budget:
                raise AssertionError(
                    f"{name} parameter count {count} exceeds "
                    f"{config.parameter_budget}"
                )
            first = core_set[0]
            batch = {
                key: (
                    value.unsqueeze(0).to(device)
                    if isinstance(value, torch.Tensor)
                    else [value]
                )
                for key, value in first.items()
            }
            _assert_epoch_zero_har(
                model,
                batch,
                name,
                config.seed,
            )
            run_name = f"{fold.name}_{name}_seed_{config.seed}"
            training = train_model(
                model,
                core_set,
                validation_set,
                config,
                horizon_epochs,
                checkpoints_dir / run_name,
                prep_meta["preprocessor_hash"],
                scheduler_hash,
                run_name,
                logger,
                device,
                resume=resume,
                max_epochs_override=(
                    config.smoke_epochs if smoke else None
                ),
                max_train_batches=(
                    config.smoke_max_train_batches if smoke else None
                ),
                max_eval_batches=(
                    config.smoke_max_eval_batches if smoke else None
                ),
                include_epoch_zero_checkpoint=True,
            )
            if training.numerical_failure:
                failure = {
                    "fold": fold.name,
                    "model": name,
                    "seed": config.seed,
                    "run_name": run_name,
                    "failure_reason": training.failure_reason,
                    "best_epoch_before_failure": training.best_epoch,
                    "epochs_run": training.epochs_run,
                    "best_validation_qlike_before_failure": (
                        training.best_validation_qlike
                    ),
                    "amp_overflow_recoveries": (
                        training.amp_overflow_recoveries
                    ),
                    "prediction_generated": False,
                    "main_table_eligible": False,
                }
                numerical_failures.append(failure)
                metadata["transformer"][name] = {
                    **prep_meta,
                    "parameter_count": count,
                    **failure,
                }
                logger.error(
                    "NUMERICAL RUN EXCLUDED | fold=%s model=%s seed=%d "
                    "reason=%s | continuing benchmark",
                    fold.name,
                    name,
                    config.seed,
                    training.failure_reason,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            validation_frame = predict_dataset(
                model,
                validation_set,
                config,
                device,
                scheduler_path,
                scheduler_hash,
                max_batches=None,
            )
            test_frame = predict_dataset(
                model,
                test_set,
                config,
                device,
                scheduler_path,
                scheduler_hash,
                max_batches=None,
            )
            if name == "slow_update_multiquery_gated":
                validation_frame["correction_gate"] = _predict_gate_values(
                    model, validation_set, config, device
                )
                test_frame["correction_gate"] = _predict_gate_values(
                    model, test_set, config, device
                )
            validation_frame.to_csv(
                predictions_dir / f"{fold.name}_{name}_validation.csv",
                index=False,
            )
            annotated = _annotate_predictions(
                test_frame, fold.name, name, spike_threshold
            )
            annotated.to_csv(
                predictions_dir / f"{fold.name}_{name}.csv", index=False
            )
            pooled[name].append(annotated)
            fold_rows.append(
                {
                    "fold": fold.name,
                    **_diagnostic_metrics(
                        annotated, spike_threshold, name
                    ),
                    "best_epoch": training.best_epoch,
                    "epochs_run": training.epochs_run,
                    "early_stopped": training.early_stopped,
                    "best_validation_qlike": (
                        training.best_validation_qlike
                    ),
                    "correction_selected": training.best_epoch > 0,
                    "training_seconds": training.training_seconds,
                    "peak_gpu_memory_bytes": (
                        training.peak_gpu_memory_bytes
                    ),
                    "amp_overflow_recoveries": (
                        training.amp_overflow_recoveries
                    ),
                }
            )
            metadata["transformer"][name] = {
                **prep_meta,
                "parameter_count": count,
                "best_epoch": training.best_epoch,
                "epochs_run": training.epochs_run,
                "early_stopped": training.early_stopped,
                "best_validation_qlike": training.best_validation_qlike,
                "amp_overflow_recoveries": (
                    training.amp_overflow_recoveries
                ),
                "epoch_zero_exact_har": True,
                "uses_update_mask": name not in {
                    "slow_calendar_control",
                    "fast_calendar_control",
                },
                "news_state": (
                    "fast" if name == "fast_calendar_control" else "slow"
                ),
                "market_query_count": (
                    1
                    if name
                    in {
                        "slow_calendar_control",
                        "fast_calendar_control",
                        "slow_update_tokens",
                    }
                    else 3
                ),
                "point_in_time_gate": name.endswith("_gated"),
            }
            if name == "slow_update_multiquery_gated":
                metadata["transformer"][name]["gate_distribution"] = {
                    block: {
                        "mean": float(frame["correction_gate"].mean()),
                        "std": float(frame["correction_gate"].std(ddof=0)),
                        "min": float(frame["correction_gate"].min()),
                        "max": float(frame["correction_gate"].max()),
                    }
                    for block, frame in (
                        ("validation", validation_frame),
                        ("test", test_frame),
                    )
                }
                gated_frames = {
                    "validation": validation_frame,
                    "test": test_frame,
                }

        blend_name = "finbert_normal_slow_blend"
        if gated_frames:
            blend_validation, blend_test, blend_meta, blend_grid = (
                select_log_blend(
                    linear_frames["finbert_normal"]["validation"],
                    gated_frames["validation"],
                    linear_frames["finbert_normal"]["test"],
                    gated_frames["test"],
                    min_delta=config.min_delta,
                )
            )
            blend_validation.to_csv(
                predictions_dir
                / f"{fold.name}_{blend_name}_validation.csv",
                index=False,
            )
            blend_annotated = _annotate_predictions(
                blend_test, fold.name, blend_name, spike_threshold
            )
            blend_annotated.to_csv(
                predictions_dir / f"{fold.name}_{blend_name}.csv",
                index=False,
            )
            blend_grid.to_csv(
                metrics_dir / f"{fold.name}_{blend_name}_alpha_grid.csv",
                index=False,
            )
            pooled[blend_name].append(blend_annotated)
            fold_rows.append(
                {
                    "fold": fold.name,
                    **_diagnostic_metrics(
                        blend_annotated, spike_threshold, blend_name
                    ),
                    "best_epoch": None,
                    "early_stopped": None,
                    "best_validation_qlike": blend_meta[
                        "validation_qlike"
                    ],
                    "correction_selected": bool(
                        blend_meta["alpha_slow"] > 0
                        or linear_frames["finbert_normal"]["validation"][
                            "delta_log_rv"
                        ]
                        .abs()
                        .max()
                        > 0
                    ),
                }
            )
            metadata["blend"][blend_name] = blend_meta
        else:
            failure = {
                "fold": fold.name,
                "model": blend_name,
                "seed": config.seed,
                "run_name": f"{fold.name}_{blend_name}_seed_{config.seed}",
                "failure_reason": (
                    "dependency slow_update_multiquery_gated has no eligible "
                    "prediction"
                ),
                "prediction_generated": False,
                "main_table_eligible": False,
            }
            numerical_failures.append(failure)
            metadata["blend"][blend_name] = failure
            logger.error(
                "DEPENDENT RUN EXCLUDED | fold=%s model=%s seed=%d",
                fold.name,
                blend_name,
                config.seed,
            )
        fold_metadata[fold.name] = metadata
        write_json(
            features_dir / f"{fold.name}_metadata.json", metadata
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pooled_rows = []
    expected_folds = {fold.name for fold in folds}
    model_completion: dict[str, Any] = {}
    for name, frames in pooled.items():
        completed_folds = (
            sorted(
                set(
                    pd.concat(frames, ignore_index=True)["fold"].astype(str)
                )
            )
            if frames
            else []
        )
        eligible = set(completed_folds) == expected_folds
        model_completion[name] = {
            "expected_folds": sorted(expected_folds),
            "completed_folds": completed_folds,
            "eligible_for_seed_ensemble": eligible,
            "missing_folds": sorted(expected_folds - set(completed_folds)),
        }
        if not frames:
            continue
        pooled_frame = pd.concat(frames, ignore_index=True)
        pooled_frame.to_csv(
            predictions_dir
            / (
                f"pooled_{name}.csv"
                if eligible
                else f"pooled_partial_{name}.csv"
            ),
            index=False,
        )
        row = _safe_pooled_metrics(pooled_frame, name)
        row["eligible_for_seed_ensemble"] = eligible
        row["completed_fold_count"] = len(completed_folds)
        pooled_rows.append(row)
    pooled_metrics = pd.DataFrame(pooled_rows)
    pooled_metrics.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    if selection_mode == "development_screen":
        selection = _screen(
            fold_metrics, pooled_metrics, config.min_delta
        )
    elif selection_mode == "single_fold_evaluation":
        selection = {
            "rule": (
                "No model or PCA selection is permitted on Fold 5. "
                "Configuration was frozen before OOS prediction."
            ),
            "folds_compared": int(fold_metrics["fold"].nunique()),
            "primary_model": "slow_calendar_control",
            "statistical_claim": (
                "None; single-fold temporal generalization diagnostic."
            ),
        }
    elif selection_mode == "confirmatory_evaluation":
        selection = {
            "rule": (
                "No architecture, PCA dimension, hyperparameter, blend, or "
                "checkpoint rule may be selected from final-holdout outcomes."
            ),
            "folds_compared": int(fold_metrics["fold"].nunique()),
            "primary_model": "slow_calendar_control",
            "statistical_claim": (
                "Confirmatory evaluation of the frozen development protocol."
            ),
        }
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    influence = _influence_sensitivity(pooled)
    pd.DataFrame(influence).to_csv(
        metrics_dir / "top_spike_influence_sensitivity.csv", index=False
    )
    write_json(metrics_dir / "selection_screen.json", selection)
    write_json(metrics_dir / "model_completion.json", model_completion)
    write_json(metrics_dir / "numerical_failures.json", numerical_failures)
    write_json(features_dir / "fold_metadata.json", fold_metadata)
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "selection": selection,
        "top_spike_influence_sensitivity": influence,
        "fold_metadata": fold_metadata,
        "model_completion": model_completion,
        "numerical_failures": numerical_failures,
    }


def run_development_slow_transformer_v2_diagnostic(
    config: MainPilotConfig,
    logger: logging.Logger,
    review_audit_dir: Path,
    silver_path: Path,
    longtext_cache_path: Path,
    scheduler_path: Path,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != PROFILE:
        raise ValueError("Wrong profile for slow-Transformer-v2 diagnostic")
    output_dir = config.output_path
    expanded = review_audit_dir / "stratified_news_filter_review.csv"
    original = (
        review_audit_dir / "stratified_news_filter_review_original_366.csv"
    )
    for required in (expanded, original, silver_path, scheduler_path):
        if not required.exists():
            raise RuntimeError(f"Required artifact is missing: {required}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full slow-Transformer-v2 diagnostic is "
            "blocked on CPU; use --smoke locally."
        )
    policy = refined_policy(fit_event_aware_policy(expanded, original))
    schedule, scheduler_hash, scheduler_validation = (
        _validate_vector_scheduler(config, scheduler_path)
    )
    signature = stable_hash(
        {
            "config": config.to_dict(),
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "expanded_review": file_fingerprint(expanded),
            "silver": file_fingerprint(silver_path),
            "policy": asdict(policy),
            "scheduler": schedule,
            "candidates": CANDIDATES,
            "blend_alpha_grid": BLEND_ALPHA_GRID,
            "event_families": EVENT_FAMILIES,
        }
    )
    report_path = output_dir / "metrics" / "diagnostic_report.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info("SLOW V2 RESUME | completed report reused")
            return report
    seed_everything(config.seed)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "audit" / "scheduler_compatibility.json",
        scheduler_validation,
    )
    write_json(output_dir / "audit" / "event_aware_policy.json", asdict(policy))
    silver_evaluation = evaluate_filter_on_silver_holdout(
        silver_path,
        policy,
        output_dir / "audit",
        decision_fn=refined_event_decision,
        warning=(
            "GPT-silver development audit; not expert ground truth or an "
            "independent holdout."
        ),
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "profile": PROFILE,
            "scope": "Development Fold 1-4 only",
            "candidates": list(CANDIDATES),
            "seed": config.seed,
            "scheduler_validation": scheduler_validation,
            "information_cutoff": "t-1",
            "prototype_fit_scope": "core-only independently by fold",
            "blend_selection_scope": "validation only",
            "locked_architecture": {
                "d_model": config.d_model,
                "heads": config.attention_heads,
                "news_layers": config.news_layers,
                "cross_layers": config.cross_layers,
                "ffn_dim": config.ffn_dim,
                "dropout": config.dropout,
            },
            "excluded": [
                "Fold_5",
                "final_test",
                "COVID_fold",
                "MCS",
                "five_seeds",
                "PCA16",
                "combined_PCA",
                "post_validation_refit",
                "realized_spike_routing",
            ],
            "created_utc": utc_now(),
        },
    )
    logger.info("SLOW V2 STEP 1/5 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("SLOW V2 STEP 2/5 | Apply refined event-aware filter")
    articles, filter_audit = load_event_aware_articles(
        Path(config.news_path),
        config.research_start,
        config.development_end,
        policy,
        logger,
        decision_fn=refined_event_decision,
    )
    write_json(
        output_dir / "audit" / "event_filter_data_audit.json", filter_audit
    )
    logger.info("SLOW V2 STEP 3/5 | Reuse cached BGE and FinBERT vectors")
    daily, semantics, embedding_audit = _embed_articles(
        articles,
        config,
        longtext_cache_path,
        torch.device("cuda"),
        logger,
        smoke=False,
        end=config.development_end,
    )
    write_json(
        output_dir / "audit" / "embedding_audit.json", embedding_audit
    )
    logger.info(
        "SLOW V2 STEP 4/5 | Fit control, updates, multi-query, gate, blend, prototype"
    )
    results = _run_folds(
        config,
        logger,
        market,
        daily,
        articles,
        semantics,
        output_dir,
        SPIKE_DIAGNOSTIC_FOLDS,
        torch.device("cuda"),
        int(schedule["H_cos"]),
        scheduler_path,
        scheduler_hash,
        resume,
    )
    logger.info("SLOW V2 STEP 5/5 | Pool, influence audit, and screen")
    report = {
        "status": "completed",
        "run_signature": signature,
        "scheduler_validation": scheduler_validation,
        "silver_filter_evaluation": silver_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "candidates": list(CANDIDATES),
        **results,
        "statistical_claim": (
            "None; convergence and competitiveness diagnostic on development "
            "Fold 1-4 only."
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("SLOW V2 COMPLETED | %s", report_path)
    return report


def run_slow_transformer_v2_smoke(
    config: MainPilotConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    smoke_dir = config.output_path / "smoke"
    smoke_config = replace(
        config,
        smoke=True,
        output_dir=str(smoke_dir),
        embedding_cache_path=None,
        num_workers=0,
        physical_batch_size=8,
        effective_batch_size=8,
    )
    policy = EventAwarePolicy(
        enabled_context_families=(
            "regulation_etf",
            "exchange_custody",
            "macro_liquidity",
            "mining_energy",
        ),
        relevant_rate_threshold=0.0,
        minimum_family_examples=0,
        training_rows=0,
        holdout_rows=0,
        family_statistics=tuple(),
        label_source="synthetic smoke policy",
    )
    seed_everything(smoke_config.seed)
    market = load_market_data(
        Path(smoke_config.market_path),
        smoke_config,
        logger,
        smoke_config.smoke_start,
        smoke_config.smoke_end,
    )
    articles, filter_audit = load_event_aware_articles(
        Path(smoke_config.news_path),
        smoke_config.smoke_start,
        smoke_config.smoke_end,
        policy,
        logger,
        smoke_early_stop=True,
        decision_fn=refined_event_decision,
    )
    scheduler_path = smoke_dir / "scheduler_horizon.json"
    schedule = {"H_cos": 200, "smoke": True, "created_utc": utc_now()}
    write_json(scheduler_path, schedule)
    scheduler_hash = stable_hash(schedule)
    daily, semantics, embedding_audit = _embed_articles(
        articles,
        smoke_config,
        smoke_dir / "cache" / "longtext_embeddings.sqlite",
        torch.device("cpu"),
        logger,
        smoke=True,
        end=smoke_config.smoke_end,
    )
    fold = Fold(
        "smoke_fold",
        "2018-03-02",
        "2018-04-15",
        "2018-04-16",
        "2018-04-30",
        "2018-05-01",
        "2018-05-31",
    )
    results = _run_folds(
        smoke_config,
        logger,
        market,
        daily,
        articles,
        semantics,
        smoke_dir,
        (fold,),
        torch.device("cpu"),
        200,
        scheduler_path,
        scheduler_hash,
        resume=False,
        lambda_grid=(1.0,),
        smoke=True,
    )
    report = {
        "status": "passed",
        "backend": "CPU deterministic smoke encoder",
        "metrics": str(smoke_dir / "metrics" / "pooled_metrics.csv"),
        "candidates": list(CANDIDATES),
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        **results,
    }
    write_json(
        smoke_dir / "slow_transformer_v2_smoke_report.json", report
    )
    return report
