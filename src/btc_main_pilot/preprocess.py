from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import MainPilotConfig
from .data import MarketData
from .news import DAILY_SCALAR_COLUMNS, UNDEFINED_NO_NEWS_COLUMNS
from .utils import ensure_finite, hash_arrays


@dataclass
class FoldNewsFeatures:
    dates: pd.DatetimeIndex
    semantic_slow: np.ndarray
    semantic_fast: np.ndarray
    sentiment_slow: np.ndarray
    sentiment_fast: np.ndarray
    daily_scalars: np.ndarray
    preprocessor_hash: str
    metadata: dict[str, Any]


@dataclass
class FoldMarketScaler:
    mean: np.ndarray
    scale: np.ndarray
    fine_patch_logrv_mean: float
    fine_patch_logrv_scale: float
    coarse_patch_logrv_mean: float
    coarse_patch_logrv_scale: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (values - self.mean) / self.scale
        ensure_finite("scaled market features", transformed)
        return transformed.astype(np.float32)


def _captured_variance_ratio(
    vectors: np.ndarray, scaler: StandardScaler, pca: PCA
) -> float:
    if len(vectors) == 0:
        return float("nan")
    standardized = scaler.transform(vectors).astype(np.float64)
    centered = standardized - pca.mean_
    projected = centered @ pca.components_.T
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(projected**2) / denominator)


def fit_transform_news_for_fold(
    daily: pd.DataFrame,
    core_start: str,
    core_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str | None,
    test_end: str | None,
    config: MainPilotConfig,
    logger: logging.Logger,
) -> FoldNewsFeatures:
    core_mask = (daily.index >= pd.Timestamp(core_start, tz="UTC")) & (
        daily.index <= pd.Timestamp(core_end, tz="UTC")
    )
    news_mask = daily["no_news_dummy"].to_numpy() == 0
    core_news_mask = core_mask & news_mask
    core_semantics = np.stack(daily.loc[core_news_mask, "semantic"].to_list())
    if len(core_semantics) < config.pca_dim:
        raise ValueError("Core train has fewer news days than PCA components")
    semantic_scaler = StandardScaler().fit(core_semantics)
    pca = PCA(n_components=config.pca_dim, svd_solver="full").fit(
        semantic_scaler.transform(core_semantics)
    )
    z = np.full((len(daily), config.pca_dim), np.nan, dtype=np.float64)
    if news_mask.any():
        all_semantics = np.stack(daily.loc[news_mask, "semantic"].to_list())
        z[news_mask] = pca.transform(semantic_scaler.transform(all_semantics))

    continuous_columns = DAILY_SCALAR_COLUMNS[:-1]
    scalar_raw = daily[continuous_columns].copy()
    medians = scalar_raw.loc[core_mask].median(axis=0, skipna=True)
    scalar_imputed = scalar_raw.fillna(medians)
    scalar_scaler = StandardScaler().fit(scalar_imputed.loc[core_mask])
    scalar_scaled = scalar_scaler.transform(scalar_imputed)
    daily_scalars = np.column_stack(
        [scalar_scaled, daily["no_news_dummy"].to_numpy(dtype=np.float64)]
    )
    undefined_positions = [
        continuous_columns.index(column) for column in UNDEFINED_NO_NEWS_COLUMNS
    ]
    daily_scalars[np.ix_(~news_mask, undefined_positions)] = 0.0
    ensure_finite("daily scalar features", daily_scalars)

    semantic_slow = np.zeros_like(z)
    semantic_fast = np.zeros_like(z)
    sentiment_slow = np.zeros((len(daily), 3), dtype=np.float64)
    sentiment_fast = np.zeros((len(daily), 3), dtype=np.float64)
    initialized = False
    last_semantic = np.zeros(config.pca_dim, dtype=np.float64)
    last_sentiment = np.zeros(3, dtype=np.float64)
    for i in range(len(daily)):
        if not news_mask[i]:
            semantic_slow[i] = last_semantic
            sentiment_slow[i] = last_sentiment
            continue
        observation_semantic = z[i]
        observation_sentiment = np.asarray(daily.iloc[i]["sentiment"], dtype=np.float64)
        if not initialized:
            last_semantic = observation_semantic.copy()
            last_sentiment = observation_sentiment.copy()
            initialized = True
        else:
            last_semantic = (
                (1.0 - config.slow_alpha) * last_semantic
                + config.slow_alpha * observation_semantic
            )
            last_sentiment = (
                (1.0 - config.slow_alpha) * last_sentiment
                + config.slow_alpha * observation_sentiment
            )
            semantic_fast[i] = observation_semantic - last_semantic
            sentiment_fast[i] = observation_sentiment - last_sentiment
        semantic_slow[i] = last_semantic
        sentiment_slow[i] = last_sentiment
    for name, value in [
        ("semantic_slow", semantic_slow),
        ("semantic_fast", semantic_fast),
        ("sentiment_slow", sentiment_slow),
        ("sentiment_fast", sentiment_fast),
    ]:
        ensure_finite(name, value)

    preprocessor_hash = hash_arrays(
        {
            "semantic_scaler_mean": semantic_scaler.mean_,
            "semantic_scaler_scale": semantic_scaler.scale_,
            "pca_mean": pca.mean_,
            "pca_components": pca.components_,
            "pca_explained_variance": pca.explained_variance_,
            "pca_n_components": pca.n_components_,
        }
    )
    block_masks = {
        "core": core_mask,
        "validation": (
            (daily.index >= pd.Timestamp(validation_start, tz="UTC"))
            & (daily.index <= pd.Timestamp(validation_end, tz="UTC"))
        ),
        "test": (
            (
                (daily.index >= pd.Timestamp(test_start, tz="UTC"))
                & (daily.index <= pd.Timestamp(test_end, tz="UTC"))
            )
            if test_start is not None and test_end is not None
            else np.zeros(len(daily), dtype=bool)
        ),
    }
    cvr = {}
    for block, mask in block_masks.items():
        block_news = mask & news_mask
        vectors = (
            np.stack(daily.loc[block_news, "semantic"].to_list())
            if block_news.any()
            else np.empty((0, config.embedding_dim))
        )
        cvr[block] = _captured_variance_ratio(vectors, semantic_scaler, pca)
    relative_drop = (
        1.0 - cvr["test"] / cvr["core"]
        if np.isfinite(cvr["test"]) and np.isfinite(cvr["core"]) and cvr["core"] != 0
        else float("nan")
    )
    metadata = {
        "preprocessor_hash": preprocessor_hash,
        "pca_n_components": config.pca_dim,
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "CVR_core": cvr["core"],
        "CVR_validation": cvr["validation"],
        "CVR_test": cvr["test"],
        "relative_drop": relative_drop,
        "semantic_scaler_fit_start": core_start,
        "semantic_scaler_fit_end": core_end,
        "recursion_start": str(daily.index[0]),
        "scalar_columns": DAILY_SCALAR_COLUMNS,
        "scalar_medians": medians.to_dict(),
        "scalar_scaler_mean": scalar_scaler.mean_.tolist(),
        "scalar_scaler_scale": scalar_scaler.scale_.tolist(),
    }
    logger.info(
        "NEWS PREPROCESS | hash=%s CVR core=%.4f validation=%.4f test=%.4f drop=%.4f",
        preprocessor_hash[:12],
        cvr["core"],
        cvr["validation"],
        cvr["test"],
        relative_drop,
    )
    return FoldNewsFeatures(
        dates=daily.index,
        semantic_slow=semantic_slow.astype(np.float32),
        semantic_fast=semantic_fast.astype(np.float32),
        sentiment_slow=sentiment_slow.astype(np.float32),
        sentiment_fast=sentiment_fast.astype(np.float32),
        daily_scalars=daily_scalars.astype(np.float32),
        preprocessor_hash=preprocessor_hash,
        metadata=metadata,
    )


def _patch_logrv_values(
    market: MarketData,
    target_indices: list[int],
    lookback_days: int,
    patch_length: int,
    epsilon: float,
) -> np.ndarray:
    values = []
    for target_index in target_indices:
        returns = market.raw_returns[target_index - lookback_days : target_index].reshape(-1)
        patches = returns.reshape(-1, patch_length)
        values.append(np.log(np.sum(patches**2, axis=1) + epsilon))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def fit_market_scaler(
    market: MarketData,
    core_dates: list[pd.Timestamp],
    config: MainPilotConfig,
) -> FoldMarketScaler:
    core_calendar_mask = (
        (market.dates >= core_dates[0].normalize())
        & (market.dates <= core_dates[-1].normalize())
        & market.valid
    )
    values = market.features[core_calendar_mask].reshape(-1, len(market.audit["model_channels"]))
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale = np.maximum(scale, config.scaler_epsilon)
    target_indices = [market.date_to_index[date] for date in core_dates]
    fine = _patch_logrv_values(
        market,
        target_indices,
        config.fine_lookback_days,
        config.fine_patch_length,
        config.epsilon_rv,
    )
    coarse = _patch_logrv_values(
        market,
        target_indices,
        config.coarse_lookback_days,
        config.coarse_patch_length,
        config.epsilon_rv,
    )
    scaler = FoldMarketScaler(
        mean=mean,
        scale=scale,
        fine_patch_logrv_mean=float(np.mean(fine)),
        fine_patch_logrv_scale=max(float(np.std(fine)), config.scaler_epsilon),
        coarse_patch_logrv_mean=float(np.mean(coarse)),
        coarse_patch_logrv_scale=max(float(np.std(coarse)), config.scaler_epsilon),
    )
    ensure_finite("market scaler mean", scaler.mean)
    ensure_finite("market scaler scale", scaler.scale)
    return scaler
