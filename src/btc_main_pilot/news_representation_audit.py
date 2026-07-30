from __future__ import annotations

import json
import logging
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import StandardScaler

from .baselines import fit_har_qlike
from .config import Fold, MainPilotConfig, SPIKE_DIAGNOSTIC_FOLDS
from .data import load_market_data, write_market_audit
from .losses import exact_qlike_numpy
from .metrics import prediction_metrics
from .news import (
    DeterministicSmokeEncoder,
    EmbeddingCache,
    FilteredArticle,
    OfflineBgeFinbertEncoder,
    aggregate_daily_news,
    conservative_deduplicate_articles,
    embed_articles,
    load_filtered_articles,
    write_news_audits,
)
from .pipeline import _block_dates
from .preprocess import FoldNewsFeatures, fit_transform_news_for_fold
from .spike_diagnostic import _annotate_predictions, _pooled_metrics
from .utils import (
    ensure_finite,
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


NEWS_PROBE_NAMES = (
    "news_scalars_11",
    "finbert_slow_fast_6",
    "bge_pca8_slow_fast_16",
    "bge_pca16_slow_fast_32",
    "surprise_norms_2",
    "combined_pca8_33",
    "source_balanced_dedup_pca8_33",
    "combined_pca16_surprise_51",
)
LAMBDA_SUM_GRID = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
)


def _write_quality_review(
    audit: dict[str, Any],
    output_dir: Path,
    confirmed: bool,
) -> Path:
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "stratified_news_filter_review.csv"
    frame = pd.DataFrame(audit["stratified_review_examples"])
    frame["manual_relevant"] = ""
    frame["review_notes"] = ""
    frame.to_csv(path, index=False)
    write_json(
        audit_dir / "stratified_news_filter_review_status.json",
        {
            "development_only": True,
            "operator_confirmed": confirmed,
            "sample_path": str(path),
            "sample_n": len(frame),
            "strata": [
                "decision",
                "year",
                "evidence_type",
                "score_band",
            ],
            "selection": (
                "Deterministic hash-ranked sample, four records per non-empty "
                "stratum; no forecast outcome is used."
            ),
            "created_utc": utc_now(),
        },
    )
    return path


def _source_audit(articles: list[FilteredArticle]) -> dict[str, Any]:
    rows = pd.DataFrame(
        {
            "year": [article.timestamp.year for article in articles],
            "source": [
                article.source or "__unknown_source__" for article in articles
            ],
        }
    )
    yearly: list[dict[str, Any]] = []
    for year, group in rows.groupby("year"):
        shares = group["source"].value_counts(normalize=True)
        yearly.append(
            {
                "year": int(year),
                "article_n": int(len(group)),
                "source_n": int(group["source"].nunique()),
                "largest_source": str(shares.index[0]),
                "largest_source_share": float(shares.iloc[0]),
                "source_hhi": float(np.sum(shares.to_numpy() ** 2)),
            }
        )
    return {"yearly": yearly}


def _token_length_audit(
    articles: list[FilteredArticle],
    tokenizer: Any | None,
    logger: logging.Logger | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    lengths: list[int] = []
    backend = "exact_BGE_tokenizer"
    for offset in range(0, len(articles), batch_size):
        texts = [
            article.encoder_text
            for article in articles[offset : offset + batch_size]
        ]
        if tokenizer is None:
            backend = "smoke_whitespace_approximation"
            lengths.extend(len(text.split()) + 2 for text in texts)
        else:
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_length=True,
            )
            lengths.extend(int(value) for value in encoded["length"])
        completed = min(offset + batch_size, len(articles))
        if logger is not None and (
            completed == len(articles)
            or (offset // batch_size + 1) % 20 == 0
        ):
            logger.info(
                "TOKEN AUDIT | %d/%d (%.1f%%)",
                completed,
                len(articles),
                100.0 * completed / max(len(articles), 1),
            )
    array = np.asarray(lengths, dtype=np.int64)
    return {
        "backend": backend,
        "article_n": len(articles),
        "token_length_p50": float(np.quantile(array, 0.50)),
        "token_length_p90": float(np.quantile(array, 0.90)),
        "token_length_p99": float(np.quantile(array, 0.99)),
        "over_512_n": int(np.sum(array > 512)),
        "over_512_rate": float(np.mean(array > 512)),
        "max_tokens_used_by_encoder": 512,
    }


def _quality_audits(
    articles: list[FilteredArticle],
    filter_audit: dict[str, Any],
    output_dir: Path,
    tokenizer: Any | None,
    confirmed: bool,
    logger: logging.Logger | None = None,
) -> tuple[list[int], dict[str, Any], Path]:
    review_path = _write_quality_review(filter_audit, output_dir, confirmed)
    kept_indices, duplicate_audit = conservative_deduplicate_articles(
        articles
    )
    source_audit = _source_audit(articles)
    token_audit = _token_length_audit(articles, tokenizer, logger)
    report = {
        "filter": {
            key: filter_audit[key]
            for key in (
                "records_scanned",
                "outside_date_range",
                "invalid_timestamp",
                "filtered_irrelevant",
                "retained",
                "monotonic_timestamp_order",
                "timestamp_limitation",
                "canonical_cluster_limitation",
            )
        },
        "duplicate": duplicate_audit,
        "source": source_audit,
        "token_length": token_audit,
        "review_path": str(review_path),
        "review_confirmed": confirmed,
    }
    write_json(output_dir / "audit" / "news_quality_report.json", report)
    pd.DataFrame(duplicate_audit["examples"]).to_csv(
        output_dir / "audit" / "duplicate_examples.csv", index=False
    )
    pd.DataFrame(source_audit["yearly"]).to_csv(
        output_dir / "audit" / "source_concentration_yearly.csv", index=False
    )
    return kept_indices, report, review_path


def run_news_representation_review(
    config: MainPilotConfig,
    logger: logging.Logger,
) -> Path:
    articles, filter_audit = load_filtered_articles(
        Path(config.news_path),
        config.research_start,
        config.development_end,
        logger,
        stratified_review_per_cell=4,
    )
    _, _, review_path = _quality_audits(
        articles,
        filter_audit,
        config.output_path,
        tokenizer=None,
        confirmed=False,
        logger=logger,
    )
    logger.info(
        "NEWS QUALITY | retained=%d stratified_review=%s",
        len(articles),
        review_path,
    )
    logger.info(
        "NEWS REPRESENTATION REVIEW READY | sample=%s", review_path
    )
    return review_path


def _embedding_inputs(
    config: MainPilotConfig,
    output_dir: Path,
    logger: logging.Logger,
    device: torch.device,
    smoke: bool,
    confirm_review: bool,
) -> tuple[
    list[FilteredArticle],
    np.ndarray,
    np.ndarray,
    list[int],
    dict[str, Any],
]:
    end = config.smoke_end if smoke else config.development_end
    articles, filter_audit = load_filtered_articles(
        Path(config.news_path),
        config.research_start,
        end,
        logger,
        smoke_early_stop=smoke,
        stratified_review_per_cell=4,
    )
    if not smoke and not confirm_review:
        _, _, review_path = _quality_audits(
            articles,
            filter_audit,
            output_dir,
            tokenizer=None,
            confirmed=False,
            logger=logger,
        )
        raise RuntimeError(
            "Stratified development-only news review is required before "
            f"loading encoders or running probes: {review_path}"
        )
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder
    encoder = (
        DeterministicSmokeEncoder(config.embedding_dim)
        if smoke
        else OfflineBgeFinbertEncoder(config, device)
    )
    tokenizer = (
        None if smoke else getattr(encoder, "semantic_tokenizer", None)
    )
    kept_indices, quality_report, review_path = _quality_audits(
        articles,
        filter_audit,
        output_dir,
        tokenizer,
        confirmed=confirm_review,
        logger=logger,
    )
    duplicate = quality_report["duplicate"]
    token = quality_report["token_length"]
    logger.info(
        "NEWS QUALITY | retained=%d deduplicated=%d removed=%d "
        "over_512=%d/%d review=%s",
        len(articles),
        duplicate["retained_articles"],
        duplicate["removed_articles"],
        token["over_512_n"],
        token["article_n"],
        review_path,
    )
    cache_semantic = (
        f"{config.semantic_model}::deterministic-smoke-v1"
        if smoke
        else config.semantic_model
    )
    cache_sentiment = (
        f"{config.sentiment_model}::deterministic-smoke-v1"
        if smoke
        else config.sentiment_model
    )
    cache_path = (
        output_dir / "cache" / "article_embeddings.sqlite"
        if smoke or config.embedding_cache_path is None
        else Path(config.embedding_cache_path)
    )
    cache = EmbeddingCache(cache_path, cache_semantic, cache_sentiment)
    try:
        semantics, sentiments, cache_stats = embed_articles(
            articles,
            cache,
            encoder,
            config.embedding_batch_size,
            logger,
        )
    finally:
        cache.close()
    quality_report["embedding_cache"] = cache_stats
    write_json(output_dir / "audit" / "news_quality_report.json", quality_report)
    return articles, semantics, sentiments, kept_indices, quality_report


def _fold_features(
    daily: pd.DataFrame,
    fold: Fold,
    config: MainPilotConfig,
    logger: logging.Logger,
    pca_dim: int,
) -> FoldNewsFeatures:
    candidate_config = replace(config, pca_dim=pca_dim)
    return fit_transform_news_for_fold(
        daily,
        fold.core_start,
        fold.core_end,
        fold.validation_start,
        fold.validation_end,
        fold.test_start,
        fold.test_end,
        candidate_config,
        logger,
    )


def _feature_row(
    target: pd.Timestamp,
    current8: FoldNewsFeatures,
    current16: FoldNewsFeatures,
    balanced8: FoldNewsFeatures,
    name: str,
) -> np.ndarray:
    index = int(current8.dates.get_loc(target - pd.Timedelta(days=1)))
    if name == "news_scalars_11":
        return current8.daily_scalars[index].astype(np.float64)
    if name == "finbert_slow_fast_6":
        return np.concatenate(
            [current8.sentiment_slow[index], current8.sentiment_fast[index]]
        ).astype(np.float64)
    if name == "bge_pca8_slow_fast_16":
        return np.concatenate(
            [current8.semantic_slow[index], current8.semantic_fast[index]]
        ).astype(np.float64)
    if name == "bge_pca16_slow_fast_32":
        return np.concatenate(
            [current16.semantic_slow[index], current16.semantic_fast[index]]
        ).astype(np.float64)
    if name == "surprise_norms_2":
        return np.asarray(
            [
                np.linalg.norm(current8.semantic_fast[index]),
                np.linalg.norm(current8.sentiment_fast[index]),
            ],
            dtype=np.float64,
        )
    if name == "combined_pca8_33":
        return np.concatenate(
            [
                current8.semantic_slow[index],
                current8.semantic_fast[index],
                current8.sentiment_slow[index],
                current8.sentiment_fast[index],
                current8.daily_scalars[index],
            ]
        ).astype(np.float64)
    if name == "source_balanced_dedup_pca8_33":
        return np.concatenate(
            [
                balanced8.semantic_slow[index],
                balanced8.semantic_fast[index],
                balanced8.sentiment_slow[index],
                balanced8.sentiment_fast[index],
                balanced8.daily_scalars[index],
            ]
        ).astype(np.float64)
    if name == "combined_pca16_surprise_51":
        return np.concatenate(
            [
                current16.semantic_slow[index],
                current16.semantic_fast[index],
                current16.sentiment_slow[index],
                current16.sentiment_fast[index],
                current16.daily_scalars[index],
                [
                    np.linalg.norm(current16.semantic_fast[index]),
                    np.linalg.norm(current16.sentiment_fast[index]),
                ],
            ]
        ).astype(np.float64)
    raise ValueError(f"Unknown probe: {name}")


def _feature_matrix(
    dates: list[pd.Timestamp],
    current8: FoldNewsFeatures,
    current16: FoldNewsFeatures,
    balanced8: FoldNewsFeatures,
    name: str,
) -> np.ndarray:
    matrix = np.stack(
        [
            _feature_row(
                date, current8, current16, balanced8, name
            )
            for date in dates
        ]
    )
    ensure_finite(f"{name} feature matrix", matrix)
    return matrix


def _feature_names(name: str) -> list[str]:
    semantic8 = [
        f"semantic_{state}_pc_{index:02d}"
        for state in ("slow", "fast")
        for index in range(1, 9)
    ]
    semantic16 = [
        f"semantic_{state}_pc_{index:02d}"
        for state in ("slow", "fast")
        for index in range(1, 17)
    ]
    sentiment = [
        f"sentiment_{state}_{label}"
        for state in ("slow", "fast")
        for label in ("positive", "negative", "neutral")
    ]
    scalars = [
        "news_intensity",
        "log1p_canonical_source_count",
        "negative_ratio",
        "log1p_negative_count_070",
        "negative_probability_max",
        "negative_probability_std",
        "positive_probability_max",
        "sentiment_entropy_mean",
        "semantic_dispersion",
        "mean_relevance",
        "no_news_dummy",
    ]
    surprise = ["semantic_fast_l2", "sentiment_fast_l2"]
    mapping = {
        "news_scalars_11": scalars,
        "finbert_slow_fast_6": sentiment,
        "bge_pca8_slow_fast_16": semantic8,
        "bge_pca16_slow_fast_32": semantic16,
        "surprise_norms_2": surprise,
        "combined_pca8_33": semantic8 + sentiment + scalars,
        "source_balanced_dedup_pca8_33": (
            semantic8 + sentiment + scalars
        ),
        "combined_pca16_surprise_51": (
            semantic16 + sentiment + scalars + surprise
        ),
    }
    return mapping[name]


def _target_rv(market: Any, dates: list[pd.Timestamp]) -> np.ndarray:
    return np.asarray(
        [market.rv[market.date_to_index[date]] for date in dates],
        dtype=np.float64,
    )


def _prediction_frame(
    dates: list[pd.Timestamp] | None,
    true_rv: np.ndarray,
    anchor_log: np.ndarray,
    correction_log: np.ndarray,
) -> pd.DataFrame:
    predicted_log = anchor_log + correction_log
    predicted_rv = np.exp(predicted_log)
    ensure_finite("probe predicted RV", predicted_rv)
    return pd.DataFrame(
        {
            "target_date": (
                [date.strftime("%Y-%m-%d") for date in dates]
                if dates is not None
                else [""] * len(true_rv)
            ),
            "true_rv": true_rv,
            "true_log_rv": np.log(true_rv),
            "har_anchor_log_rv": anchor_log,
            "delta_log_rv": correction_log,
            "predicted_rv": predicted_rv,
            "predicted_log_rv": predicted_log,
        }
    )


def _fit_probe(
    name: str,
    x_core: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    core_rv: np.ndarray,
    validation_rv: np.ndarray,
    test_rv: np.ndarray,
    core_anchor_log: np.ndarray,
    validation_anchor_log: np.ndarray,
    test_anchor_log: np.ndarray,
    feature_names: list[str],
    lambda_grid: tuple[float, ...],
    min_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    scaler = StandardScaler().fit(x_core)
    core_scaled = scaler.transform(x_core)
    validation_scaled = scaler.transform(x_validation)
    test_scaled = scaler.transform(x_test)
    relative_core = core_rv / np.exp(core_anchor_log)
    anchor_validation_qlike = float(
        np.mean(
            exact_qlike_numpy(
                validation_rv, np.exp(validation_anchor_log)
            )
        )
    )
    best_model: GammaRegressor | None = None
    best_lambda: float | None = None
    best_validation = float("inf")
    grid_rows: list[dict[str, Any]] = []
    for lambda_sum in lambda_grid:
        alpha = 2.0 * lambda_sum / len(core_rv)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = GammaRegressor(
                alpha=alpha,
                fit_intercept=True,
                solver="lbfgs",
                max_iter=5000,
                tol=1e-8,
                warm_start=False,
            )
            try:
                model.fit(core_scaled, relative_core)
                relative_validation = model.predict(validation_scaled)
                predicted_validation = (
                    np.exp(validation_anchor_log) * relative_validation
                )
                validation_qlike = float(
                    np.mean(
                        exact_qlike_numpy(
                            validation_rv, predicted_validation
                        )
                    )
                )
                converged = not any(
                    issubclass(item.category, ConvergenceWarning)
                    for item in caught
                )
            except (FloatingPointError, ValueError):
                validation_qlike = float("inf")
                converged = False
        grid_rows.append(
            {
                "model": name,
                "lambda_sum": lambda_sum,
                "alpha_mean_loss": alpha,
                "validation_qlike": validation_qlike,
                "converged": converged,
                "selection_eligible": converged,
            }
        )
        better = (
            converged
            and validation_qlike < best_validation - min_delta
        )
        tied_prefer_larger = (
            converged
            and abs(validation_qlike - best_validation) <= min_delta
            and (best_lambda is None or lambda_sum > best_lambda)
        )
        if np.isfinite(validation_qlike) and (
            better or tied_prefer_larger
        ):
            best_model = model
            best_lambda = lambda_sum
            best_validation = validation_qlike
    if best_model is None or best_lambda is None:
        raise RuntimeError(f"Every lambda failed for probe {name}")
    correction_selected = bool(
        best_validation < anchor_validation_qlike - min_delta
    )
    if correction_selected:
        validation_correction = np.log(
            best_model.predict(validation_scaled)
        )
        test_correction = np.log(best_model.predict(test_scaled))
    else:
        validation_correction = np.zeros(len(validation_rv))
        test_correction = np.zeros(len(test_rv))
    validation_frame = _prediction_frame(
        None,
        validation_rv,
        validation_anchor_log,
        validation_correction,
    )
    test_frame = _prediction_frame(
        None,
        test_rv,
        test_anchor_log,
        test_correction,
    )
    metadata = {
        "model": name,
        "feature_dim": int(x_core.shape[1]),
        "ordered_feature_names": feature_names,
        "core_n": len(core_rv),
        "selected_lambda_sum": best_lambda,
        "selected_alpha_mean_loss": 2.0 * best_lambda / len(core_rv),
        "anchor_validation_qlike": anchor_validation_qlike,
        "candidate_validation_qlike": best_validation,
        "validation_improvement": anchor_validation_qlike - best_validation,
        "correction_selected": correction_selected,
        "coefficient_l2": float(np.linalg.norm(best_model.coef_)),
        "coefficients": best_model.coef_.tolist(),
        "intercept": float(best_model.intercept_),
        "feature_scaler_mean": scaler.mean_.tolist(),
        "feature_scaler_scale": scaler.scale_.tolist(),
        "convergence_rule": (
            "Select minimum validation exact QLIKE; within 1e-5 prefer "
            "larger lambda. Apply correction only if it beats HAR anchor "
            "validation QLIKE by at least 1e-5."
        ),
    }
    return (
        validation_frame,
        test_frame,
        metadata,
        pd.DataFrame(grid_rows),
    )


def _dated_probe_frames(
    validation_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    result: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    validation, test, metadata, grid = result
    validation["target_date"] = [
        date.strftime("%Y-%m-%d") for date in validation_dates
    ]
    test["target_date"] = [
        date.strftime("%Y-%m-%d") for date in test_dates
    ]
    return validation, test, metadata, grid


def _screen(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    min_delta: float,
) -> dict[str, Any]:
    anchor_fold = fold_metrics[
        fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    anchor_pooled = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    candidates: dict[str, Any] = {}
    for name in NEWS_PROBE_NAMES:
        candidate_fold = fold_metrics[
            fold_metrics["model"] == name
        ].set_index("fold")
        overall_wins = sum(
            candidate_fold.loc[fold, "mean_qlike"]
            < anchor_fold.loc[fold, "mean_qlike"] - min_delta
            for fold in anchor_fold.index
        )
        spike_wins = sum(
            candidate_fold.loc[fold, "spike_qlike"]
            < anchor_fold.loc[fold, "spike_qlike"] - min_delta
            for fold in anchor_fold.index
        )
        pooled = pooled_metrics[pooled_metrics["model"] == name].iloc[0]
        passes_overall = bool(
            overall_wins >= 3
            and pooled["mean_qlike"]
            < anchor_pooled["mean_qlike"] - min_delta
        )
        passes_spike = bool(
            spike_wins >= 3
            and pooled["spike_qlike"]
            < anchor_pooled["spike_qlike"] - min_delta
            and pooled["mean_qlike"]
            <= anchor_pooled["mean_qlike"] + min_delta
        )
        candidates[name] = {
            "correction_selected_folds": int(
                candidate_fold["correction_selected"].astype(bool).sum()
            ),
            "overall_fold_wins_vs_har_qlike": int(overall_wins),
            "spike_fold_wins_vs_har_qlike": int(spike_wins),
            "pooled_overall_delta_vs_har_qlike": float(
                pooled["mean_qlike"] - anchor_pooled["mean_qlike"]
            ),
            "pooled_spike_delta_vs_har_qlike": float(
                pooled["spike_qlike"] - anchor_pooled["spike_qlike"]
            ),
            "passes_overall_screen": passes_overall,
            "passes_spike_screen": passes_spike,
            "recommended_for_deep_followup": bool(
                passes_overall or passes_spike
            ),
        }
    return {
        "scope": "development-only rolling Fold 1-4, seed 11",
        "min_delta": min_delta,
        "overall_rule": (
            "Overall QLIKE beats HAR-QLIKE in at least 3/4 folds and pooled."
        ),
        "spike_rule": (
            "Spike QLIKE beats HAR-QLIKE in at least 3/4 folds and pooled, "
            "while pooled overall QLIKE is not worse."
        ),
        "candidates": candidates,
        "statistical_claim": "None; representation screening only.",
    }


def _run_probes(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: Any,
    daily: pd.DataFrame,
    balanced_daily: pd.DataFrame,
    output_dir: Path,
    folds: tuple[Fold, ...],
    lambda_grid: tuple[float, ...] = LAMBDA_SUM_GRID,
) -> dict[str, Any]:
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    features_dir = output_dir / "features"
    for directory in (predictions_dir, metrics_dir, features_dir):
        directory.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("har_qlike", *NEWS_PROBE_NAMES)
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "NEWS PROBE FOLD %d/%d | %s fit PCA8/PCA16 on core only",
            fold_index,
            len(folds),
            fold.name,
        )
        core_dates, validation_dates, test_dates = _block_dates(
            market, fold, output_dir, logger, include_test=True
        )
        current8 = _fold_features(daily, fold, config, logger, 8)
        current16 = _fold_features(daily, fold, config, logger, 16)
        balanced8 = _fold_features(
            balanced_daily, fold, config, logger, 8
        )
        write_json(
            features_dir / f"{fold.name}_representation_metadata.json",
            {
                "current_pca8": current8.metadata,
                "current_pca16": current16.metadata,
                "source_balanced_dedup_pca8": balanced8.metadata,
            },
        )
        anchor = fit_har_qlike(market, core_dates)
        core_rv = _target_rv(market, core_dates)
        validation_rv = _target_rv(market, validation_dates)
        test_rv = _target_rv(market, test_dates)
        core_anchor_log = anchor.predict_log_rv(market, core_dates)
        validation_anchor_log = anchor.predict_log_rv(
            market, validation_dates
        )
        test_anchor_log = anchor.predict_log_rv(market, test_dates)
        threshold = float(np.quantile(core_rv, 0.90))
        anchor_test = _prediction_frame(
            test_dates,
            test_rv,
            test_anchor_log,
            np.zeros(len(test_dates)),
        )
        anchor_annotated = _annotate_predictions(
            anchor_test, fold.name, "har_qlike", threshold
        )
        all_predictions["har_qlike"].append(anchor_annotated)
        anchor_metrics = prediction_metrics(
            anchor_test, threshold, "har_qlike"
        )
        anchor_metrics.update(
            {
                "fold": fold.name,
                "correction_selected": False,
                "validation_improvement": 0.0,
            }
        )
        fold_rows.append(anchor_metrics)

        for probe_index, name in enumerate(NEWS_PROBE_NAMES, start=1):
            logger.info(
                "NEWS PROBE %02d/%02d | fold=%s representation=%s",
                (fold_index - 1) * len(NEWS_PROBE_NAMES) + probe_index,
                len(folds) * len(NEWS_PROBE_NAMES),
                fold.name,
                name,
            )
            matrices = [
                _feature_matrix(
                    dates, current8, current16, balanced8, name
                )
                for dates in (core_dates, validation_dates, test_dates)
            ]
            fitted = _fit_probe(
                name,
                matrices[0],
                matrices[1],
                matrices[2],
                core_rv,
                validation_rv,
                test_rv,
                core_anchor_log,
                validation_anchor_log,
                test_anchor_log,
                _feature_names(name),
                lambda_grid,
                config.min_delta,
            )
            validation_frame, test_frame, metadata, grid = (
                _dated_probe_frames(
                    validation_dates, test_dates, fitted
                )
            )
            logger.info(
                "NEWS PROBE VALIDATE | fold=%s representation=%s "
                "anchor=%.8f candidate=%.8f gain=%.8f lambda=%g selected=%s",
                fold.name,
                name,
                metadata["anchor_validation_qlike"],
                metadata["candidate_validation_qlike"],
                metadata["validation_improvement"],
                metadata["selected_lambda_sum"],
                metadata["correction_selected"],
            )
            label = f"{fold.name}_{name}"
            validation_frame.to_csv(
                predictions_dir / f"{label}_validation.csv", index=False
            )
            test_frame.to_csv(
                predictions_dir / f"{label}.csv", index=False
            )
            grid.to_csv(
                metrics_dir / f"{label}_lambda_grid.csv", index=False
            )
            metrics = prediction_metrics(test_frame, threshold, name)
            metrics.update({"fold": fold.name, **metadata})
            logger.info(
                "NEWS PROBE OOS | fold=%s representation=%s QLIKE=%.8f "
                "spike=%s normal=%s NaN/Inf=%d",
                fold.name,
                name,
                metrics["mean_qlike"],
                metrics["spike_qlike"],
                metrics["normal_qlike"],
                metrics["n_nan_inf_or_nonpositive"],
            )
            write_json(metrics_dir / f"{label}.json", metrics)
            fold_rows.append(metrics)
            annotated = _annotate_predictions(
                test_frame, fold.name, name, threshold
            )
            all_predictions[name].append(annotated)

    pooled_rows: list[dict[str, Any]] = []
    for name, parts in all_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        pooled.to_csv(
            predictions_dir / f"pooled_{name}.csv", index=False
        )
        pooled_rows.append(_pooled_metrics(pooled, name))
    fold_frame = pd.DataFrame(fold_rows)
    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    screen = _screen(fold_frame, pooled_frame, config.min_delta)
    write_json(metrics_dir / "representation_screen.json", screen)
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "screen": screen,
    }


def run_development_news_representation_audit(
    config: MainPilotConfig,
    logger: logging.Logger,
    resume: bool,
    confirm_news_filter_reviewed: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != "development-news-representation-audit":
        raise ValueError("Wrong profile for news representation audit")
    output_dir = config.output_path
    signature = stable_hash(
        {
            "config": config.to_dict(),
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "probes": NEWS_PROBE_NAMES,
            "lambda_grid": LAMBDA_SUM_GRID,
        }
    )
    report_path = output_dir / "metrics" / "diagnostic_report.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info(
                "NEWS REPRESENTATION RESUME | completed report reused: %s",
                report_path,
            )
            return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Full BGE/FinBERT representation audit is "
            "blocked on CPU; use --smoke locally."
        )
    device = torch.device("cuda")
    seed_everything(config.seed)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "run_manifest.json",
        {
            "profile": config.profile,
            "run_signature": signature,
            "scope": "development-only rolling Fold 1-4",
            "probes": list(NEWS_PROBE_NAMES),
            "lambda_sum_grid": LAMBDA_SUM_GRID,
            "folds": [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS],
            "excluded": [
                "Fold_5",
                "final_test",
                "deep_model_training",
                "spike_gating",
                "MCS",
                "five_seeds",
            ],
            "created_utc": utc_now(),
        },
    )
    logger.info("NEWS AUDIT STEP 1/4 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info(
        "NEWS AUDIT STEP 2/4 | Stratified filter, duplicate, source and token audit"
    )
    articles, semantics, sentiments, kept_indices, quality = (
        _embedding_inputs(
            config,
            output_dir,
            logger,
            device,
            smoke=False,
            confirm_review=confirm_news_filter_reviewed,
        )
    )
    logger.info("NEWS AUDIT STEP 3/4 | Build current and source-balanced daily news")
    daily = aggregate_daily_news(
        articles,
        semantics,
        sentiments,
        config.research_start,
        config.development_end,
    )
    write_news_audits(daily, articles, output_dir, quality)
    deduplicated = [articles[index] for index in kept_indices]
    balanced_daily = aggregate_daily_news(
        deduplicated,
        semantics[kept_indices],
        sentiments[kept_indices],
        config.research_start,
        config.development_end,
        source_balanced=True,
    )
    balanced_daily[
        ["news_count", "canonical_source_count", "no_news_dummy"]
    ].to_csv(
        output_dir / "audit" / "source_balanced_daily_counts.csv",
        index_label="date",
    )
    logger.info(
        "NEWS AUDIT STEP 4/4 | Run HAR-anchored exact-QLIKE linear probes"
    )
    results = _run_probes(
        config,
        logger,
        market,
        daily,
        balanced_daily,
        output_dir,
        SPIKE_DIAGNOSTIC_FOLDS,
    )
    report = {
        "status": "completed",
        "run_signature": signature,
        "quality_audit": quality,
        "probes": list(NEWS_PROBE_NAMES),
        **results,
        "completed_utc": utc_now(),
        "statistical_claim": "None; development-only representation audit.",
    }
    write_json(report_path, report)
    logger.info(
        "NEWS REPRESENTATION AUDIT COMPLETED | report=%s", report_path
    )
    return report


def run_news_representation_smoke(
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
    )
    seed_everything(smoke_config.seed)
    market = load_market_data(
        Path(smoke_config.market_path),
        smoke_config,
        logger,
        smoke_config.smoke_start,
        smoke_config.smoke_end,
    )
    articles, semantics, sentiments, kept_indices, quality = (
        _embedding_inputs(
            smoke_config,
            smoke_dir,
            logger,
            torch.device("cpu"),
            smoke=True,
            confirm_review=True,
        )
    )
    daily = aggregate_daily_news(
        articles,
        semantics,
        sentiments,
        smoke_config.research_start,
        smoke_config.smoke_end,
    )
    deduplicated = [articles[index] for index in kept_indices]
    balanced_daily = aggregate_daily_news(
        deduplicated,
        semantics[kept_indices],
        sentiments[kept_indices],
        smoke_config.research_start,
        smoke_config.smoke_end,
        source_balanced=True,
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
    results = _run_probes(
        smoke_config,
        logger,
        market,
        daily,
        balanced_daily,
        smoke_dir,
        (fold,),
        lambda_grid=(1.0,),
    )
    report = {
        "status": "passed",
        "backend": "CPU",
        "embedding_backend": "deterministic smoke test double",
        "quality_audit": quality,
        "probes": list(NEWS_PROBE_NAMES),
        "runs": results["pooled_metrics"],
        "pooled_metrics": results["pooled_metrics"],
    }
    write_json(
        smoke_dir / "news_representation_smoke_report.json", report
    )
    return report
