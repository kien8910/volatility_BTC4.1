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
from torch.utils.data import Dataset

from .baselines import fit_har_qlike
from .config import Fold, MainPilotConfig, SPIKE_DIAGNOSTIC_FOLDS
from .data import MarketData, load_market_data, write_market_audit
from .event_aware_longtext_audit import (
    EventAwarePolicy,
    VariantVectorCache,
    _embed_fixed_variant,
    _fold_feature,
    _token_budgeted_text,
    evaluate_filter_on_silver_holdout,
    fit_event_aware_policy,
    load_event_aware_articles,
)
from .model import (
    CrossAttentionBlock,
    PreNormSelfAttentionBlock,
    sinusoidal_encoding,
    trainable_parameter_count,
)
from .news import (
    DeterministicSmokeEncoder,
    FilteredArticle,
    OfflineBgeFinbertEncoder,
    aggregate_daily_news,
)
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
    _validate_main_scheduler,
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


PROFILE = "development-vector-integration-diagnostic"
SPIKE_QUANTILE = 0.90
EVENT_BASELINE_DAYS = 365
EVENT_BASELINE_MIN_DAYS = 30
EVENT_FAMILIES = (
    "direct_bitcoin",
    "regulation_etf",
    "exchange_custody",
    "macro_liquidity",
    "mining_energy",
    "security_hack",
)
LINEAR_CANDIDATES = (
    "finbert_normal",
    "finbert_bge_directional",
    "finbert_event_prototypes",
)
TRANSFORMER_CANDIDATES = (
    "transformer_cross_attention_slow",
    "transformer_cross_attention_fast",
)
CANDIDATES = (
    "har_qlike",
    *LINEAR_CANDIDATES,
    *TRANSFORMER_CANDIDATES,
)
MARKET_QUERY_NAMES = (
    "market_logrv_d",
    "market_logrv_w",
    "market_logrv_m",
    "market_logrv_d_minus_w",
    "market_logrv_w_minus_m",
    "market_logrv_change_1d",
    "market_logrv_std_5d",
)
SCHEDULER_RUNTIME_CONFIG_KEYS = (
    "output_dir",
    "embedding_cache_path",
    "physical_batch_size",
    "num_workers",
    "smoke",
    "smoke_start",
    "smoke_end",
    "smoke_max_train_batches",
    "smoke_max_eval_batches",
    "smoke_epochs",
)
SCHEDULER_LOCKED_TRAINING_KEYS = (
    "seed",
    "optimizer",
    "learning_rate",
    "min_learning_rate",
    "adam_betas",
    "adam_epsilon",
    "weight_decay",
    "warmup_steps",
    "provisional_horizon_epochs",
    "max_epochs",
    "patience",
    "min_delta",
    "gradient_clip_norm",
    "amp_grad_scaler_initial_scale",
    "amp_grad_scaler_growth_interval",
    "effective_batch_size",
    "training_loss",
)


def _config_payload_hash(payload: dict[str, Any]) -> str:
    locked = dict(payload)
    for key in SCHEDULER_RUNTIME_CONFIG_KEYS:
        locked.pop(key, None)
    return stable_hash(locked)


def _validate_vector_scheduler(
    config: MainPilotConfig,
    scheduler_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Accept an authenticated legacy main scheduler when training locks match."""
    try:
        schedule, scheduler_hash = _validate_main_scheduler(
            config, scheduler_path
        )
        return schedule, scheduler_hash, {
            "mode": "exact_current_config_hash",
            "scheduler_path": str(scheduler_path),
            "scheduler_config_hash": schedule["config_hash"],
            "archived_config_path": None,
            "training_keys_verified": list(SCHEDULER_LOCKED_TRAINING_KEYS),
        }
    except RuntimeError as error:
        if "locked config hash differs" not in str(error):
            raise

    schedule = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if schedule.get("pilot_completed_without_numerical_failure") is not True:
        raise RuntimeError(
            "The scheduler does not certify a numerically successful pilot"
        )
    h_cos = schedule.get("H_cos")
    if (
        not isinstance(h_cos, int)
        or isinstance(h_cos, bool)
        or h_cos <= 0
        or h_cos > config.max_epochs
    ):
        raise RuntimeError(f"Invalid legacy scheduler H_cos: {h_cos!r}")
    archived_path = scheduler_path.parent / "config.json"
    if not archived_path.exists():
        raise RuntimeError(
            "The scheduler config hash differs and its archived main-pilot "
            f"config is missing: {archived_path}. Do not edit the scheduler; "
            "copy the matching outputs/main_pilot/config.json or rerun the pilot."
        )
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    archived_hash = _config_payload_hash(archived)
    if archived_hash != schedule.get("config_hash"):
        raise RuntimeError(
            "Archived main-pilot config does not authenticate the scheduler: "
            f"computed={archived_hash} scheduler={schedule.get('config_hash')}"
        )
    if archived.get("profile") != "main-pilot":
        raise RuntimeError("Archived scheduler config is not a main-pilot config")
    current = replace(config, profile="main-pilot").to_dict()
    mismatches = {
        key: {"archived": archived.get(key), "current": current.get(key)}
        for key in SCHEDULER_LOCKED_TRAINING_KEYS
        if stable_hash(archived.get(key)) != stable_hash(current.get(key))
    }
    if mismatches:
        raise RuntimeError(
            "Authenticated legacy scheduler differs in locked training "
            f"hyperparameters: {json.dumps(mismatches, sort_keys=True)}"
        )
    changed_nontraining_keys = sorted(
        key
        for key in set(archived) | set(current)
        if key not in SCHEDULER_RUNTIME_CONFIG_KEYS
        and key not in SCHEDULER_LOCKED_TRAINING_KEYS
        and stable_hash(archived.get(key)) != stable_hash(current.get(key))
    )
    scheduler_hash = stable_hash(schedule)
    return schedule, scheduler_hash, {
        "mode": "authenticated_legacy_config_training_locks_match",
        "scheduler_path": str(scheduler_path),
        "scheduler_config_hash": schedule["config_hash"],
        "archived_config_path": str(archived_path),
        "archived_config_hash_verified": True,
        "training_keys_verified": list(SCHEDULER_LOCKED_TRAINING_KEYS),
        "allowed_nontraining_differences": changed_nontraining_keys,
        "reason": (
            "The vector diagnostic does not use the legacy PatchTST market "
            "window, but reuses its authenticated H_cos and identical locked "
            "optimizer/objective schedule."
        ),
    }


def _embed_articles(
    articles: list[FilteredArticle],
    config: MainPilotConfig,
    cache_path: Path,
    device: torch.device,
    logger: logging.Logger,
    smoke: bool,
    end: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder = (
        DeterministicSmokeEncoder(config.embedding_dim)
        if smoke
        else OfflineBgeFinbertEncoder(config, device)
    )
    tokenizer = None if smoke else getattr(encoder, "semantic_tokenizer", None)
    texts = [
        _token_budgeted_text(
            article,
            tokenizer,
            config.max_tokens,
            include_title=True,
        )
        for article in articles
    ]
    semantic_model = (
        f"{config.semantic_model}::smoke-v1"
        if smoke
        else config.semantic_model
    )
    sentiment_model = (
        f"{config.sentiment_model}::smoke-v1"
        if smoke
        else config.sentiment_model
    )
    cache = VariantVectorCache(cache_path)
    try:
        semantics, semantic_stats = _embed_fixed_variant(
            articles,
            texts,
            cache,
            encoder,
            semantic_model,
            "title_lead_token_budget_512",
            config.embedding_dim,
            config.embedding_batch_size,
            logger,
        )
        sentiments, sentiment_stats = _embed_fixed_variant(
            articles,
            texts,
            cache,
            encoder,
            sentiment_model,
            "finbert_title_lead_token_budget_512",
            3,
            config.embedding_batch_size,
            logger,
            sentiment=True,
        )
    finally:
        cache.close()
    semantic_weights_loaded = bool(
        not smoke and getattr(encoder, "semantic_model", None) is not None
    )
    sentiment_weights_loaded = bool(
        not smoke and getattr(encoder, "sentiment_model", None) is not None
    )
    if not smoke:
        logger.info(
            "CACHE FIRST | BGE weights_loaded=%s FinBERT "
            "weights_loaded=%s",
            semantic_weights_loaded,
            sentiment_weights_loaded,
        )
    semantics = np.asarray(semantics, dtype=np.float64)
    ensure_finite("article semantic embeddings", semantics)
    daily = aggregate_daily_news(
        articles,
        semantics,
        np.asarray(sentiments, dtype=np.float64),
        config.research_start,
        end,
    )
    return daily, semantics, {
        "semantic": semantic_stats,
        "sentiment": sentiment_stats,
        "representations": [
            "title_lead_token_budget_512",
            "finbert_title_lead_token_budget_512",
        ],
        "cache_first_model_loading": True,
        "semantic_model_weights_loaded": semantic_weights_loaded,
        "sentiment_model_weights_loaded": sentiment_weights_loaded,
        "article_vectors_returned_for_core_only_event_prototypes": True,
    }


def _article_family(article: FilteredArticle) -> str:
    return _event_family_hint(article.title, article.cleaned_text[:500])


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


def build_core_event_prototype_features(
    articles: list[FilteredArticle],
    semantics: np.ndarray,
    dates: pd.DatetimeIndex,
    fold: Fold,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build outcome-blind family features; prototype fitting stops at core_end."""
    if len(articles) != len(semantics):
        raise ValueError("Article/vector length mismatch")
    core_start = pd.Timestamp(fold.core_start, tz="UTC")
    core_end = pd.Timestamp(fold.core_end, tz="UTC") + pd.Timedelta(days=1)
    families = [_article_family(article) for article in articles]
    prototypes: dict[str, np.ndarray] = {}
    prototype_counts: dict[str, int] = {}
    for family in EVENT_FAMILIES:
        positions = [
            index
            for index, article in enumerate(articles)
            if families[index] == family
            and core_start <= article.timestamp < core_end
        ]
        prototype_counts[family] = len(positions)
        prototypes[family] = (
            _unit_vector(np.mean(semantics[positions], axis=0))
            if positions
            else np.zeros(semantics.shape[1], dtype=np.float64)
        )

    count = {
        family: pd.Series(0.0, index=dates, dtype=np.float64)
        for family in EVENT_FAMILIES
    }
    sources: dict[str, dict[pd.Timestamp, set[str]]] = {
        family: {} for family in EVENT_FAMILIES
    }
    daily_vectors: dict[pd.Timestamp, list[np.ndarray]] = {}
    for article, vector, family in zip(articles, semantics, families):
        day = article.timestamp.floor("D")
        if day not in dates:
            continue
        daily_vectors.setdefault(day, []).append(vector)
        if family in count:
            count[family].loc[day] += 1.0
            sources[family].setdefault(day, set()).add(article.source)

    daily_centroid = {
        day: _unit_vector(np.mean(vectors, axis=0))
        for day, vectors in daily_vectors.items()
    }
    output: dict[str, np.ndarray] = {}
    for family in EVENT_FAMILIES:
        raw_log_count = np.log1p(count[family])
        rolling = raw_log_count.rolling(
            EVENT_BASELINE_DAYS,
            min_periods=EVENT_BASELINE_MIN_DAYS,
        ).median()
        expanding = raw_log_count.expanding(min_periods=1).median()
        baseline = rolling.fillna(expanding)
        output[f"{family}__prototype_cosine"] = np.asarray(
            [
                float(np.dot(daily_centroid[day], prototypes[family]))
                if day in daily_centroid and prototype_counts[family] > 0
                else 0.0
                for day in dates
            ],
            dtype=np.float64,
        )
        output[f"{family}__count_surprise_365d"] = (
            raw_log_count - baseline
        ).to_numpy(dtype=np.float64)
        output[f"{family}__log1p_source_count"] = np.asarray(
            [
                np.log1p(len(sources[family].get(day, set())))
                for day in dates
            ],
            dtype=np.float64,
        )
    frame = pd.DataFrame(output, index=dates)
    ensure_finite("event prototype daily features", frame.to_numpy())
    metadata = {
        "families": list(EVENT_FAMILIES),
        "prototype_counts": prototype_counts,
        "prototype_fit_start": fold.core_start,
        "prototype_fit_end_inclusive": fold.core_end,
        "prototype_input": "L2-normalized BGE article vectors",
        "daily_similarity": "cosine(daily BGE centroid, core-only family prototype)",
        "count_surprise": (
            "log1p(family article count) minus causal rolling-365-day median; "
            "expanding median used before 30 observations"
        ),
        "source_breadth": "log1p distinct canonical sources by family and day",
        "target_outcomes_used": False,
    }
    return frame, metadata


def _feature_matrix(
    dates: list[pd.Timestamp],
    news: FoldNewsFeatures,
    event: pd.DataFrame,
    name: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for target in dates:
        news_day = target - pd.Timedelta(days=1)
        index = int(news.dates.get_loc(news_day))
        finbert = np.concatenate(
            [news.sentiment_slow[index], news.sentiment_fast[index]]
        )
        if name == "finbert_normal":
            row = finbert
        elif name == "finbert_bge_directional":
            row = np.concatenate(
                [
                    finbert,
                    news.semantic_slow[index],
                    news.semantic_fast[index],
                ]
            )
        elif name == "finbert_event_prototypes":
            row = np.concatenate(
                [finbert, event.loc[news_day].to_numpy(dtype=np.float64)]
            )
        else:
            raise ValueError(f"Unknown linear candidate: {name}")
        rows.append(np.asarray(row, dtype=np.float64))
    matrix = np.stack(rows)
    ensure_finite(f"{name} matrix", matrix)
    return matrix


def _linear_feature_names(name: str, event: pd.DataFrame) -> list[str]:
    finbert = [
        f"finbert_{state}_{label}"
        for state in ("slow", "fast")
        for label in ("positive", "negative", "neutral")
    ]
    if name == "finbert_normal":
        return finbert
    if name == "finbert_bge_directional":
        return [
            *finbert,
            *[f"bge_slow_pc_{i:02d}" for i in range(1, 9)],
            *[f"bge_fast_pc_{i:02d}" for i in range(1, 9)],
        ]
    return [*finbert, *event.columns.tolist()]


def _eligible_dates(
    dates: list[pd.Timestamp],
    news_dates: pd.DatetimeIndex,
    lookback: int,
) -> list[pd.Timestamp]:
    available = set(news_dates)
    return [
        target
        for target in dates
        if all(
            target - pd.Timedelta(days=lag) in available
            for lag in range(1, lookback + 1)
        )
    ]


def _token_daily_matrix(
    news: FoldNewsFeatures,
    event: pd.DataFrame,
    state: Literal["slow", "fast", "no_slow"],
) -> tuple[np.ndarray, list[str]]:
    event_aligned = event.reindex(news.dates)
    if event_aligned.isna().any().any():
        raise ValueError("Event feature dates do not align with news features")
    state_blocks: list[np.ndarray] = []
    state_names: list[str] = []
    if state != "no_slow":
        semantic = (
            news.semantic_slow if state == "slow" else news.semantic_fast
        )
        sentiment = (
            news.sentiment_slow if state == "slow" else news.sentiment_fast
        )
        state_blocks.extend([semantic, sentiment])
        state_names.extend(
            [
                *[
                    f"bge_{state}_pc_{i:02d}"
                    for i in range(1, semantic.shape[1] + 1)
                ],
                *[
                    f"finbert_{state}_{label}"
                    for label in ("positive", "negative", "neutral")
                ],
            ]
        )
    matrix = np.column_stack(
        [
            *state_blocks,
            news.daily_scalars,
            event_aligned.to_numpy(dtype=np.float32),
        ]
    ).astype(np.float64)
    names = [
        *state_names,
        *[f"daily_scalar_{i:02d}" for i in range(news.daily_scalars.shape[1])],
        *event.columns.tolist(),
    ]
    ensure_finite(f"{state} Transformer token features", matrix)
    return matrix, names


class VectorAttentionDataset(Dataset):
    def __init__(
        self,
        dates: list[pd.Timestamp],
        market_queries: np.ndarray,
        token_daily: np.ndarray,
        news_dates: pd.DatetimeIndex,
        har_anchor_log: np.ndarray,
        true_rv: np.ndarray,
        lookback: int,
    ):
        if not (
            len(dates)
            == len(market_queries)
            == len(har_anchor_log)
            == len(true_rv)
        ):
            raise ValueError("Vector-attention dataset length mismatch")
        lookup = {date: index for index, date in enumerate(news_dates)}
        token_windows = []
        for target in dates:
            sequence = [
                lookup[target - pd.Timedelta(days=lag)]
                for lag in range(lookback, 0, -1)
            ]
            token_windows.append(token_daily[sequence])
        self.target_dates = [date.strftime("%Y-%m-%d") for date in dates]
        self.market_queries = torch.tensor(
            market_queries, dtype=torch.float32
        )
        self.news_tokens = torch.tensor(
            np.stack(token_windows), dtype=torch.float32
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
            "news_tokens": self.news_tokens[index],
            "har_anchor_log_rv": self.har_anchor_log[index],
            "true_rv": self.true_rv[index],
            "true_log_rv": self.true_log_rv[index],
        }


class HarVectorCrossAttention(nn.Module):
    """Small pre-LN Transformer correction queried by the t-1 market state."""

    def __init__(
        self,
        config: MainPilotConfig,
        token_dim: int,
        state: Literal["slow", "fast", "no_slow"],
    ):
        super().__init__()
        if state not in {"slow", "fast", "no_slow"}:
            raise ValueError(
                "Cross-attention state must be slow, fast, or no_slow"
            )
        self.state = state
        self.variant = f"transformer_cross_attention_{state}"
        d_model = config.d_model
        self.d_model = d_model
        self.market_projection = nn.Linear(len(MARKET_QUERY_NAMES), d_model)
        self.token_projection = nn.Linear(token_dim, d_model)
        self.null_news_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_news_token, std=0.02)
        self.news_blocks = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(
                    d_model,
                    config.attention_heads,
                    config.ffn_dim,
                    config.dropout,
                )
                for _ in range(config.news_layers)
            ]
        )
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

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw_tokens = batch["news_tokens"]
        batch_size, token_count, _ = raw_tokens.shape
        context = self.token_projection(raw_tokens)
        position = sinusoidal_encoding(
            token_count, self.d_model, raw_tokens.device
        )
        context = context + position[None, :, :]
        null = self.null_news_token.expand(batch_size, -1, -1)
        context = torch.cat([context, null], dim=1)
        for block in self.news_blocks:
            context = block(context)
        query = self.market_projection(batch["market_query"]).unsqueeze(1)
        key_padding_mask = torch.zeros(
            (batch_size, token_count + 1),
            dtype=torch.bool,
            device=raw_tokens.device,
        )
        for block in self.cross_blocks:
            query = block(query, context, key_padding_mask)
        delta = self.correction_head(query[:, 0]).squeeze(-1).double()
        anchor = batch["har_anchor_log_rv"].double()
        predicted_log = anchor + delta
        return {
            "har_anchor_log_rv": anchor,
            "delta_log_rv": delta,
            "predicted_log_rv": predicted_log,
            "predicted_rv": torch.exp(predicted_log),
        }


def _build_transformer_datasets(
    market: MarketData,
    news: FoldNewsFeatures,
    event: pd.DataFrame,
    fold: Fold,
    state: Literal["slow", "fast", "no_slow"],
    core_dates: list[pd.Timestamp],
    validation_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    anchors: tuple[np.ndarray, np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: MainPilotConfig,
) -> tuple[
    VectorAttentionDataset,
    VectorAttentionDataset,
    VectorAttentionDataset,
    dict[str, Any],
]:
    daily, token_names = _token_daily_matrix(news, event, state)
    core_day_mask = (
        (news.dates >= pd.Timestamp(fold.core_start, tz="UTC"))
        & (news.dates <= pd.Timestamp(fold.core_end, tz="UTC"))
    )
    token_scaler = StandardScaler().fit(daily[core_day_mask])
    daily_scaled = token_scaler.transform(daily)
    query_core = np.stack(
        [_market_gate_features(market, date) for date in core_dates]
    )
    query_scaler = StandardScaler().fit(query_core)
    date_blocks = (core_dates, validation_dates, test_dates)
    query_blocks = tuple(
        query_scaler.transform(
            np.stack([_market_gate_features(market, date) for date in dates])
        )
        for dates in date_blocks
    )
    datasets = tuple(
        VectorAttentionDataset(
            dates,
            query,
            daily_scaled,
            news.dates,
            anchor,
            target,
            config.news_lookback_days,
        )
        for dates, query, anchor, target in zip(
            date_blocks, query_blocks, anchors, targets
        )
    )
    preprocessor_hash = hash_arrays(
        {
            "news_preprocessor_hash": np.frombuffer(
                news.preprocessor_hash.encode("utf-8"), dtype=np.uint8
            ),
            "token_scaler_mean": token_scaler.mean_,
            "token_scaler_scale": token_scaler.scale_,
            "query_scaler_mean": query_scaler.mean_,
            "query_scaler_scale": query_scaler.scale_,
            "state": np.frombuffer(state.encode("utf-8"), dtype=np.uint8),
        }
    )
    metadata = {
        "state": state,
        "slow_state_included": state == "slow",
        "removed_state_features": (
            [
                "bge_slow_pca",
                "finbert_slow_probabilities",
            ]
            if state == "no_slow"
            else []
        ),
        "token_dim": int(daily.shape[1]),
        "token_feature_names": token_names,
        "market_query_names": list(MARKET_QUERY_NAMES),
        "news_lookback_calendar_days": config.news_lookback_days,
        "token_scaler_fit_start": fold.core_start,
        "token_scaler_fit_end": fold.core_end,
        "query_scaler_fit_scope": "core target dates only",
        "preprocessor_hash": preprocessor_hash,
        "information_cutoff": "Every token and market query ends at t-1",
    }
    return (*datasets, metadata)


def _set_dates(frame: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    output = frame.copy()
    output["target_date"] = [date.strftime("%Y-%m-%d") for date in dates]
    return output


def _screen(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    min_delta: float,
) -> dict[str, Any]:
    def finite_number(value: Any) -> bool:
        return value is not None and bool(np.isfinite(value))

    def better(candidate: Any, anchor: Any) -> bool:
        return (
            finite_number(candidate)
            and finite_number(anchor)
            and float(candidate) < float(anchor) - min_delta
        )

    def delta(candidate: Any, anchor: Any) -> float | None:
        if not (finite_number(candidate) and finite_number(anchor)):
            return None
        return float(candidate) - float(anchor)

    anchor_fold = fold_metrics[
        fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    anchor_pool = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    result: dict[str, Any] = {}
    for name in CANDIDATES[1:]:
        candidate_fold = fold_metrics[
            fold_metrics["model"] == name
        ].set_index("fold")
        candidate_pool = pooled_metrics[
            pooled_metrics["model"] == name
        ].iloc[0]
        common = sorted(set(anchor_fold.index) & set(candidate_fold.index))
        wins = {}
        for metric in ("mean_qlike", "normal_qlike", "spike_qlike"):
            wins[metric] = sum(
                better(
                    candidate_fold.loc[fold, metric],
                    anchor_fold.loc[fold, metric],
                )
                for fold in common
            )
        result[name] = {
            "folds_compared": len(common),
            "overall_fold_wins": wins["mean_qlike"],
            "normal_fold_wins": wins["normal_qlike"],
            "spike_fold_wins": wins["spike_qlike"],
            "pooled_overall_delta_vs_har": delta(
                candidate_pool["mean_qlike"], anchor_pool["mean_qlike"]
            ),
            "pooled_normal_delta_vs_har": delta(
                candidate_pool["normal_qlike"], anchor_pool["normal_qlike"]
            ),
            "pooled_spike_delta_vs_har": delta(
                candidate_pool["spike_qlike"], anchor_pool["spike_qlike"]
            ),
            "passes_development_screen": bool(
                wins["mean_qlike"] >= 3
                and better(
                    candidate_pool["mean_qlike"],
                    anchor_pool["mean_qlike"],
                )
            ),
        }
    return {
        "rule": (
            "A candidate passes only with overall QLIKE wins in at least 3/4 "
            "folds and improved pooled overall QLIKE versus HAR by min_delta."
        ),
        "min_delta": min_delta,
        "candidates": result,
        "statistical_claim": "None; development screening only.",
    }


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
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in CANDIDATES}
    fold_metadata: dict[str, Any] = {}
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "VECTOR FOLD %d/%d | %s sample t-1 information set",
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
        event, prototype_metadata = build_core_event_prototype_features(
            articles, semantics, news.dates, fold
        )
        event.to_csv(features_dir / f"{fold.name}_event_features.csv")
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
                    np.mean(har_validation["true_rv"] / har_validation["predicted_rv"]
                    - np.log(har_validation["true_rv"] / har_validation["predicted_rv"])
                    - 1.0)
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
            "event_prototypes": prototype_metadata,
            "linear": {},
            "transformer": {},
        }
        for name in LINEAR_CANDIDATES:
            logger.info(
                "VECTOR LINEAR | fold=%s candidate=%s", fold.name, name
            )
            matrices = tuple(
                _feature_matrix(dates, news, event, name)
                for dates in (core_dates, validation_dates, test_dates)
            )
            validation_frame, test_frame, candidate_meta, grid = _fit_probe(
                name,
                *matrices,
                *targets,
                *anchors,
                _linear_feature_names(name, event),
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

        for state in ("slow", "fast"):
            name = f"transformer_cross_attention_{state}"
            logger.info(
                "VECTOR TRANSFORMER | fold=%s candidate=%s", fold.name, name
            )
            core_set, validation_set, test_set, prep_meta = (
                _build_transformer_datasets(
                    market,
                    news,
                    event,
                    fold,
                    state,
                    core_dates,
                    validation_dates,
                    test_dates,
                    anchors,
                    targets,
                    config,
                )
            )
            seed_everything(config.seed)
            model = HarVectorCrossAttention(
                config, prep_meta["token_dim"], state
            ).to(device)
            parameter_count = trainable_parameter_count(model)
            if parameter_count > config.parameter_budget:
                raise AssertionError(
                    f"{name} parameter count {parameter_count} exceeds "
                    f"{config.parameter_budget}"
                )
            with torch.no_grad():
                initial = model(
                    {
                        key: (
                            value[:2].to(device)
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in {
                            key: torch.stack(
                                [core_set[i][key] for i in range(min(2, len(core_set)))]
                            )
                            if isinstance(core_set[0][key], torch.Tensor)
                            else [core_set[i][key] for i in range(min(2, len(core_set)))]
                            for key in core_set[0]
                        }.items()
                    }
                )
            if not torch.equal(
                initial["predicted_log_rv"].cpu(),
                initial["har_anchor_log_rv"].cpu(),
            ):
                raise AssertionError(f"{name} is not exactly HAR at epoch zero")
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
            validation_frame = predict_dataset(
                model,
                validation_set,
                config,
                device,
                scheduler_path,
                scheduler_hash,
                max_batches=(
                    config.smoke_max_eval_batches if smoke else None
                ),
            )
            test_frame = predict_dataset(
                model,
                test_set,
                config,
                device,
                scheduler_path,
                scheduler_hash,
                max_batches=(
                    config.smoke_max_eval_batches if smoke else None
                ),
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
                }
            )
            metadata["transformer"][name] = {
                **prep_meta,
                "parameter_count": parameter_count,
                "best_epoch": training.best_epoch,
                "epochs_run": training.epochs_run,
                "early_stopped": training.early_stopped,
                "best_validation_qlike": training.best_validation_qlike,
                "epoch_zero_exact_har": True,
                "architecture": (
                    "pre-LN news self-attention + residual FFN; pre-LN "
                    "market-query/news multi-head cross-attention + residual FFN"
                ),
            }
        fold_metadata[fold.name] = metadata
        write_json(
            features_dir / f"{fold.name}_metadata.json", metadata
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pooled_rows = []
    for name, frames in pooled.items():
        pooled_frame = pd.concat(frames, ignore_index=True)
        pooled_frame.to_csv(
            predictions_dir / f"pooled_{name}.csv", index=False
        )
        pooled_rows.append(_safe_pooled_metrics(pooled_frame, name))
    pooled_metrics = pd.DataFrame(pooled_rows)
    pooled_metrics.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    selection = _screen(fold_metrics, pooled_metrics, config.min_delta)
    write_json(metrics_dir / "selection_screen.json", selection)
    write_json(features_dir / "fold_metadata.json", fold_metadata)
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "selection": selection,
        "fold_metadata": fold_metadata,
    }


def run_development_vector_integration_diagnostic(
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
        raise ValueError("Wrong profile for vector-integration diagnostic")
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
            "CUDA is unavailable. Full vector integration is blocked on CPU; "
            "use --smoke locally."
        )
    policy = refined_policy(fit_event_aware_policy(expanded, original))
    schedule, scheduler_hash, scheduler_validation = _validate_vector_scheduler(
        config, scheduler_path
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
            logger.info("VECTOR INTEGRATION RESUME | completed report reused")
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
            "information_cutoff": "t-1",
            "prototype_fit_scope": "core-only independently by fold",
            "transformer_comparison": "same architecture; slow vs fast state only",
            "scheduler_validation": scheduler_validation,
            "excluded": [
                "Fold_5",
                "final_test",
                "COVID_fold",
                "MCS",
                "five_seeds",
                "PCA16",
                "combined_PCA",
                "post_validation_refit",
            ],
            "created_utc": utc_now(),
        },
    )
    logger.info("VECTOR INTEGRATION STEP 1/5 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("VECTOR INTEGRATION STEP 2/5 | Refined event-aware filter")
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
    logger.info(
        "VECTOR INTEGRATION STEP 3/5 | Cached BGE and FinBERT embeddings"
    )
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
        "VECTOR INTEGRATION STEP 4/5 | Fit linear probes and Transformer branches"
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
    logger.info("VECTOR INTEGRATION STEP 5/5 | Pool and screen")
    report = {
        "status": "completed",
        "run_signature": signature,
        "silver_filter_evaluation": silver_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "scheduler_validation": scheduler_validation,
        "candidates": list(CANDIDATES),
        **results,
        "statistical_claim": (
            "None; convergence and competitiveness diagnostic on development "
            "Fold 1-4 only."
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("VECTOR INTEGRATION COMPLETED | %s", report_path)
    return report


def run_vector_integration_smoke(
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
    schedule = {
        "H_cos": 200,
        "smoke": True,
        "created_utc": utc_now(),
    }
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
        smoke_dir / "vector_integration_smoke_report.json", report
    )
    return report
