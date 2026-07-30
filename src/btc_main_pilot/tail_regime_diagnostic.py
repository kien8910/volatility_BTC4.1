from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

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
from .losses import exact_qlike_numpy
from .news import (
    DAILY_SCALAR_COLUMNS,
    DeterministicSmokeEncoder,
    FilteredArticle,
    OfflineBgeFinbertEncoder,
    aggregate_daily_news,
)
from .news_filter_labeling import _event_family_hint
from .news_representation_audit import (
    LAMBDA_SUM_GRID,
    _prediction_frame,
    _target_rv,
)
from .pipeline import _block_dates
from .point_in_time_gate_diagnostic import (
    GATE_LOGISTIC_C,
    refined_event_decision,
    refined_policy,
)
from .preprocess import FoldNewsFeatures
from .spike_diagnostic import (
    _annotate_predictions,
    _diagnostic_metrics,
    _pooled_metrics,
)
from .utils import (
    ensure_finite,
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


PROFILE = "development-tail-regime-diagnostic"
TAIL_QUANTILE = 0.80
SPIKE_QUANTILE = 0.90
ROLLING_TRAIN_DAYS = 730
EXPONENTIAL_HALF_LIFE_DAYS = 365.0
NORMAL_FEATURE_NAMES = tuple(
    f"sentiment_{state}_{label}"
    for state in ("slow", "fast")
    for label in ("positive", "negative", "neutral")
)
TAIL_FEATURE_NAMES = (
    *NORMAL_FEATURE_NAMES,
    "semantic_fast_l2",
    "sentiment_fast_l2",
    "news_intensity",
    "log1p_canonical_source_count",
    "no_news_dummy",
)
CORRECTION_KINDS = ("finbert_normal", "finbert_tail_soft_mixture")


@dataclass(frozen=True)
class TemporalTrainingSpec:
    name: str
    description: str


TEMPORAL_SPECS = (
    TemporalTrainingSpec(
        "expanding",
        "All eligible core observations receive equal weight.",
    ),
    TemporalTrainingSpec(
        "rolling730",
        "Only observations in the final 730 calendar days of core are used.",
    ),
    TemporalTrainingSpec(
        "exp_decay365",
        "All core observations are weighted with a 365-day half-life.",
    ),
)
CONFIGURATION_NAMES = tuple(
    f"{temporal.name}__{correction}"
    for temporal in TEMPORAL_SPECS
    for correction in CORRECTION_KINDS
)


def _temporal_training_weights(
    dates: list[pd.Timestamp],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not dates:
        raise ValueError("Temporal weighting requires at least one core date")
    index = pd.DatetimeIndex(dates)
    end = index.max()
    age_days = (end - index).days.to_numpy(dtype=np.float64)
    if mode == "expanding":
        mask = np.ones(len(index), dtype=bool)
        weights = np.ones(len(index), dtype=np.float64)
    elif mode == "rolling730":
        cutoff = end - pd.Timedelta(days=ROLLING_TRAIN_DAYS - 1)
        mask = np.asarray(index >= cutoff, dtype=bool)
        weights = np.ones(int(mask.sum()), dtype=np.float64)
    elif mode == "exp_decay365":
        mask = np.ones(len(index), dtype=bool)
        weights = np.power(
            0.5,
            age_days / EXPONENTIAL_HALF_LIFE_DAYS,
            dtype=np.float64,
        )
        weights /= float(np.mean(weights))
    else:
        raise ValueError(f"Unknown temporal weighting mode: {mode}")
    if not mask.any() or len(weights) != int(mask.sum()):
        raise AssertionError("Invalid temporal training support")
    ensure_finite(f"{mode} temporal weights", weights)
    if np.any(weights <= 0):
        raise AssertionError("Temporal weights must be strictly positive")
    selected = index[mask]
    metadata = {
        "mode": mode,
        "core_n": int(len(index)),
        "fit_n": int(mask.sum()),
        "fit_start": selected.min().strftime("%Y-%m-%d"),
        "fit_end": selected.max().strftime("%Y-%m-%d"),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "rolling_calendar_days": (
            ROLLING_TRAIN_DAYS if mode == "rolling730" else None
        ),
        "half_life_calendar_days": (
            EXPONENTIAL_HALF_LIFE_DAYS
            if mode == "exp_decay365"
            else None
        ),
    }
    return mask, weights, metadata


def _build_tail_daily_frame(
    articles: list[FilteredArticle],
    config: MainPilotConfig,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    smoke: bool,
    end: str,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder = (
        DeterministicSmokeEncoder(config.embedding_dim)
        if smoke
        else OfflineBgeFinbertEncoder(config, device)
    )
    tokenizer = (
        None if smoke else getattr(encoder, "semantic_tokenizer", None)
    )
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
    cache = VariantVectorCache(
        cache_path
        if cache_path is not None
        else output_dir / "cache" / "longtext_embeddings.sqlite"
    )
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
    daily = aggregate_daily_news(
        articles,
        semantics,
        sentiments,
        config.research_start,
        end,
    )
    return daily, {
        "semantic": semantic_stats,
        "sentiment": sentiment_stats,
        "representations": [
            "title_lead_token_budget_512",
            "finbert_title_lead_token_budget_512",
        ],
        "excluded_representations": [
            "title_only",
            "content_only",
            "chunk_mean",
            "PCA16",
        ],
    }


def _normal_feature_row(
    target: pd.Timestamp,
    features: FoldNewsFeatures,
) -> np.ndarray:
    index = int(features.dates.get_loc(target - pd.Timedelta(days=1)))
    return np.concatenate(
        [features.sentiment_slow[index], features.sentiment_fast[index]]
    ).astype(np.float64)


def _tail_feature_row(
    target: pd.Timestamp,
    features: FoldNewsFeatures,
) -> np.ndarray:
    index = int(features.dates.get_loc(target - pd.Timedelta(days=1)))
    finbert = np.concatenate(
        [features.sentiment_slow[index], features.sentiment_fast[index]]
    )
    surprise = np.asarray(
        [
            np.linalg.norm(features.semantic_fast[index]),
            np.linalg.norm(features.sentiment_fast[index]),
        ],
        dtype=np.float64,
    )
    count_indices = [
        DAILY_SCALAR_COLUMNS.index("news_intensity"),
        DAILY_SCALAR_COLUMNS.index("log1p_canonical_source_count"),
        DAILY_SCALAR_COLUMNS.index("no_news_dummy"),
    ]
    counts = features.daily_scalars[index, count_indices]
    return np.concatenate([finbert, surprise, counts]).astype(np.float64)


def _feature_matrix(
    dates: list[pd.Timestamp],
    features: FoldNewsFeatures,
    kind: str,
) -> np.ndarray:
    row_fn = _normal_feature_row if kind == "normal" else _tail_feature_row
    matrix = np.stack([row_fn(date, features) for date in dates])
    ensure_finite(f"{kind} feature matrix", matrix)
    return matrix


def _fit_gamma(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    lambda_sum: float,
) -> GammaRegressor:
    alpha = 2.0 * lambda_sum / len(target)
    model = GammaRegressor(
        alpha=alpha,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-8,
        warm_start=False,
    )
    model.fit(x, target, sample_weight=weights)
    return model


def _fit_tail_gate(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> LogisticRegression:
    if np.unique(labels).size != 2:
        raise ValueError("Tail gate training labels must contain both classes")
    model = LogisticRegression(
        C=GATE_LOGISTIC_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=11,
    )
    model.fit(x, labels, sample_weight=weights)
    return model


def _fit_configuration(
    name: str,
    correction_kind: str,
    temporal_mode: str,
    core_dates: list[pd.Timestamp],
    validation_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    core_rv: np.ndarray,
    validation_rv: np.ndarray,
    test_rv: np.ndarray,
    core_anchor_log: np.ndarray,
    validation_anchor_log: np.ndarray,
    test_anchor_log: np.ndarray,
    normal_matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    tail_matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    lambda_grid: tuple[float, ...],
    min_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    mask, weights, temporal_metadata = _temporal_training_weights(
        core_dates, temporal_mode
    )
    relative_core = core_rv / np.exp(core_anchor_log)
    tail_threshold = float(np.quantile(core_rv, TAIL_QUANTILE))
    selected_tail = core_rv[mask] > tail_threshold
    if selected_tail.sum() < 5 or (~selected_tail).sum() < 5:
        raise RuntimeError(
            f"Insufficient P80 classes for {name}: "
            f"tail={selected_tail.sum()} normal={(~selected_tail).sum()}"
        )

    normal_scaler = StandardScaler().fit(
        normal_matrices[0][mask],
        sample_weight=weights,
    )
    normal_core = normal_scaler.transform(normal_matrices[0][mask])
    normal_validation = normal_scaler.transform(normal_matrices[1])
    normal_test = normal_scaler.transform(normal_matrices[2])

    tail_scaler: StandardScaler | None = None
    gate: LogisticRegression | None = None
    tail_core: np.ndarray | None = None
    tail_validation: np.ndarray | None = None
    tail_test: np.ndarray | None = None
    validation_probability = np.zeros(len(validation_dates), dtype=np.float64)
    test_probability = np.zeros(len(test_dates), dtype=np.float64)
    gate_metadata: dict[str, Any] = {}
    if correction_kind == "finbert_tail_soft_mixture":
        tail_scaler = StandardScaler().fit(
            tail_matrices[0][mask],
            sample_weight=weights,
        )
        tail_core = tail_scaler.transform(tail_matrices[0][mask])
        tail_validation = tail_scaler.transform(tail_matrices[1])
        tail_test = tail_scaler.transform(tail_matrices[2])
        gate = _fit_tail_gate(
            tail_core,
            selected_tail.astype(np.int64),
            weights,
        )
        validation_probability = gate.predict_proba(tail_validation)[:, 1]
        test_probability = gate.predict_proba(tail_test)[:, 1]
        validation_tail_label = validation_rv > tail_threshold
        validation_has_both_classes = (
            np.unique(validation_tail_label).size == 2
        )
        gate_metadata = {
            "gate_logistic_c": GATE_LOGISTIC_C,
            "gate_train_tail_n": int(selected_tail.sum()),
            "gate_train_normal_n": int((~selected_tail).sum()),
            "gate_validation_average_precision": (
                float(
                    average_precision_score(
                        validation_tail_label,
                        validation_probability,
                    )
                )
                if validation_has_both_classes
                else None
            ),
            "gate_validation_brier": float(
                brier_score_loss(
                    validation_tail_label,
                    validation_probability,
                )
            ),
            "gate_validation_probability_mean": float(
                np.mean(validation_probability)
            ),
            "gate_test_probability_mean": float(
                np.mean(test_probability)
            ),
        }

    anchor_validation_qlike = float(
        np.mean(
            exact_qlike_numpy(
                validation_rv,
                np.exp(validation_anchor_log),
            )
        )
    )
    best: dict[str, Any] | None = None
    grid_rows: list[dict[str, Any]] = []
    for lambda_sum in lambda_grid:
        converged = True
        warning_messages: list[str] = []
        validation_qlike = float("inf")
        candidate: dict[str, Any] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                normal_model = _fit_gamma(
                    normal_core,
                    relative_core[mask],
                    weights,
                    lambda_sum,
                )
                normal_validation_relative = normal_model.predict(
                    normal_validation
                )
                normal_test_relative = normal_model.predict(normal_test)
                if correction_kind == "finbert_normal":
                    validation_relative = normal_validation_relative
                    test_relative = normal_test_relative
                    tail_model = None
                else:
                    assert (
                        tail_core is not None
                        and tail_validation is not None
                        and tail_test is not None
                    )
                    tail_model = _fit_gamma(
                        tail_core[selected_tail],
                        relative_core[mask][selected_tail],
                        weights[selected_tail],
                        lambda_sum,
                    )
                    tail_validation_relative = tail_model.predict(
                        tail_validation
                    )
                    tail_test_relative = tail_model.predict(tail_test)
                    validation_relative = (
                        (1.0 - validation_probability)
                        * normal_validation_relative
                        + validation_probability
                        * tail_validation_relative
                    )
                    test_relative = (
                        (1.0 - test_probability) * normal_test_relative
                        + test_probability * tail_test_relative
                    )
                predicted_validation = (
                    np.exp(validation_anchor_log) * validation_relative
                )
                predicted_test = np.exp(test_anchor_log) * test_relative
                ensure_finite(
                    f"{name} validation predictions",
                    predicted_validation,
                )
                ensure_finite(f"{name} test predictions", predicted_test)
                if np.any(predicted_validation <= 0) or np.any(
                    predicted_test <= 0
                ):
                    raise FloatingPointError("Non-positive Gamma prediction")
                validation_qlike = float(
                    np.mean(
                        exact_qlike_numpy(
                            validation_rv,
                            predicted_validation,
                        )
                    )
                )
                candidate = {
                    "lambda_sum": lambda_sum,
                    "normal_model": normal_model,
                    "tail_model": tail_model,
                    "normal_validation_relative": normal_validation_relative,
                    "normal_test_relative": normal_test_relative,
                    "validation_relative": validation_relative,
                    "test_relative": test_relative,
                    "validation_qlike": validation_qlike,
                }
            except (FloatingPointError, ValueError):
                converged = False
            warning_messages = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            converged = converged and not warning_messages
        grid_rows.append(
            {
                "configuration": name,
                "lambda_sum": lambda_sum,
                "normal_alpha_mean_loss": (
                    2.0 * lambda_sum / int(mask.sum())
                ),
                "tail_alpha_mean_loss": (
                    2.0 * lambda_sum / int(selected_tail.sum())
                    if correction_kind == "finbert_tail_soft_mixture"
                    else None
                ),
                "validation_qlike": validation_qlike,
                "converged": converged,
                "convergence_warnings": json.dumps(warning_messages),
            }
        )
        if not converged or not np.isfinite(validation_qlike):
            continue
        if best is None:
            best = candidate
            continue
        better = validation_qlike < best["validation_qlike"] - min_delta
        tied = (
            abs(validation_qlike - best["validation_qlike"]) <= min_delta
            and lambda_sum > best["lambda_sum"]
        )
        if better or tied:
            best = candidate
    if best is None:
        raise RuntimeError(f"Every lambda failed for {name}")

    correction_selected = bool(
        best["validation_qlike"]
        < anchor_validation_qlike - min_delta
    )
    if correction_selected:
        validation_relative = best["validation_relative"]
        test_relative = best["test_relative"]
    else:
        validation_relative = np.ones(len(validation_dates))
        test_relative = np.ones(len(test_dates))
    validation_frame = _prediction_frame(
        validation_dates,
        validation_rv,
        validation_anchor_log,
        np.log(validation_relative),
    )
    test_frame = _prediction_frame(
        test_dates,
        test_rv,
        test_anchor_log,
        np.log(test_relative),
    )
    for frame, probability, normal_relative in (
        (
            validation_frame,
            validation_probability,
            best["normal_validation_relative"],
        ),
        (test_frame, test_probability, best["normal_test_relative"]),
    ):
        frame["tail_probability"] = probability
        frame["normal_relative_rv"] = normal_relative
        frame["selected_relative_rv"] = (
            validation_relative
            if frame is validation_frame
            else test_relative
        )
        frame["information_cutoff"] = (
            pd.to_datetime(frame["target_date"])
            - pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")

    normal_model = best["normal_model"]
    tail_model = best["tail_model"]
    metadata = {
        "configuration": name,
        "correction_kind": correction_kind,
        "temporal": temporal_metadata,
        "core_tail_quantile": TAIL_QUANTILE,
        "core_tail_threshold": tail_threshold,
        "selected_lambda_sum": best["lambda_sum"],
        "anchor_validation_qlike": anchor_validation_qlike,
        "candidate_validation_qlike": best["validation_qlike"],
        "validation_improvement": (
            anchor_validation_qlike - best["validation_qlike"]
        ),
        "correction_selected": correction_selected,
        "normal_feature_names": list(NORMAL_FEATURE_NAMES),
        "tail_feature_names": (
            list(TAIL_FEATURE_NAMES)
            if correction_kind == "finbert_tail_soft_mixture"
            else []
        ),
        "normal_coefficients": normal_model.coef_.tolist(),
        "normal_intercept": float(normal_model.intercept_),
        "tail_coefficients": (
            tail_model.coef_.tolist() if tail_model is not None else None
        ),
        "tail_intercept": (
            float(tail_model.intercept_)
            if tail_model is not None
            else None
        ),
        "normal_scaler_mean": normal_scaler.mean_.tolist(),
        "normal_scaler_scale": normal_scaler.scale_.tolist(),
        "tail_scaler_mean": (
            tail_scaler.mean_.tolist() if tail_scaler is not None else None
        ),
        "tail_scaler_scale": (
            tail_scaler.scale_.tolist() if tail_scaler is not None else None
        ),
        "selection_rule": (
            "Minimum validation exact QLIKE; within min_delta prefer the "
            "larger lambda. Fall back to the expanding-core HAR anchor unless "
            "the selected correction improves validation by min_delta."
        ),
        **gate_metadata,
    }
    return (
        validation_frame,
        test_frame,
        metadata,
        pd.DataFrame(grid_rows),
    )


def _article_day_summary(
    articles: list[FilteredArticle],
) -> dict[pd.Timestamp, dict[str, Any]]:
    grouped: dict[pd.Timestamp, list[FilteredArticle]] = {}
    for article in articles:
        grouped.setdefault(article.timestamp.normalize(), []).append(article)
    output: dict[pd.Timestamp, dict[str, Any]] = {}
    for day, values in grouped.items():
        families: dict[str, int] = {}
        for article in values:
            family = _event_family_hint(
                article.title,
                article.cleaned_text[:500],
            )
            families[family] = families.get(family, 0) + 1
        ordered = sorted(values, key=lambda article: article.timestamp)
        output[day] = {
            "news_count": len(values),
            "canonical_source_count": len(
                {article.source for article in values}
            ),
            "first_publication_utc": ordered[0].timestamp.isoformat(),
            "last_publication_utc": ordered[-1].timestamp.isoformat(),
            "event_family_counts": json.dumps(
                families,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "titles": json.dumps(
                [article.title for article in ordered[:5]],
                ensure_ascii=False,
            ),
        }
    return output


def _spike_information_rows(
    fold: Fold,
    annotated_anchor: pd.DataFrame,
    features: FoldNewsFeatures,
    articles_by_day: dict[pd.Timestamp, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    empty = {
        "news_count": 0,
        "canonical_source_count": 0,
        "first_publication_utc": None,
        "last_publication_utc": None,
        "event_family_counts": "{}",
        "titles": "[]",
    }
    for item in annotated_anchor.loc[
        annotated_anchor["is_spike"]
    ].itertuples(index=False):
        target = pd.Timestamp(item.target_date, tz="UTC")
        prior = target - pd.Timedelta(days=1)
        prior_summary = articles_by_day.get(prior, empty)
        contemporaneous = articles_by_day.get(target, empty)
        feature_index = int(features.dates.get_loc(prior))
        prior_count = int(prior_summary["news_count"])
        contemporaneous_count = int(contemporaneous["news_count"])
        if prior_count and contemporaneous_count:
            availability = "prior_and_contemporaneous_news"
        elif prior_count:
            availability = "prior_news_only"
        elif contemporaneous_count:
            availability = "contemporaneous_only_no_t_minus_1_news"
        else:
            availability = "no_filtered_news_t_minus_1_or_t"
        rows.append(
            {
                "fold": fold.name,
                "target_date": target.strftime("%Y-%m-%d"),
                "information_cutoff": prior.strftime("%Y-%m-%d"),
                "true_rv": float(item.true_rv),
                "spike_threshold_core_p90": float(item.spike_threshold),
                "har_predicted_rv": float(item.predicted_rv),
                "har_qlike": float(item.qlike),
                "availability_class": availability,
                "prior_news_count": prior_count,
                "prior_canonical_source_count": int(
                    prior_summary["canonical_source_count"]
                ),
                "prior_event_family_counts": prior_summary[
                    "event_family_counts"
                ],
                "prior_first_publication_utc": prior_summary[
                    "first_publication_utc"
                ],
                "prior_last_publication_utc": prior_summary[
                    "last_publication_utc"
                ],
                "prior_titles": prior_summary["titles"],
                "prior_news_intensity_transformed": float(
                    features.daily_scalars[
                        feature_index,
                        DAILY_SCALAR_COLUMNS.index("news_intensity"),
                    ]
                ),
                "prior_semantic_fast_l2": float(
                    np.linalg.norm(features.semantic_fast[feature_index])
                ),
                "prior_sentiment_fast_l2": float(
                    np.linalg.norm(features.sentiment_fast[feature_index])
                ),
                "target_day_posthoc_news_count": contemporaneous_count,
                "target_day_posthoc_source_count": int(
                    contemporaneous["canonical_source_count"]
                ),
                "target_day_posthoc_event_family_counts": contemporaneous[
                    "event_family_counts"
                ],
                "target_day_posthoc_first_publication_utc": contemporaneous[
                    "first_publication_utc"
                ],
                "target_day_posthoc_titles": contemporaneous["titles"],
                "target_day_fields_used_by_model": False,
            }
        )
    return rows


def _selection_screen(
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
            and candidate < anchor - min_delta
        )

    def numeric_delta(candidate: Any, anchor: Any) -> float | None:
        if not (finite(candidate) and finite(anchor)):
            return None
        return float(candidate - anchor)

    anchor_folds = fold_metrics[
        fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    anchor = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    candidates: dict[str, Any] = {}
    for name in CONFIGURATION_NAMES:
        candidate_folds = fold_metrics[
            fold_metrics["model"] == name
        ].set_index("fold")
        pooled = pooled_metrics[pooled_metrics["model"] == name].iloc[0]
        overall_wins = sum(
            better(
                candidate_folds.loc[fold, "mean_qlike"],
                anchor_folds.loc[fold, "mean_qlike"],
            )
            for fold in anchor_folds.index
        )
        spike_wins = sum(
            better(
                candidate_folds.loc[fold, "spike_qlike"],
                anchor_folds.loc[fold, "spike_qlike"],
            )
            for fold in anchor_folds.index
        )
        normal_limit = float(anchor["normal_qlike"]) * 1.01
        candidates[name] = {
            "overall_fold_wins": int(overall_wins),
            "spike_fold_wins": int(spike_wins),
            "pooled_overall_delta": numeric_delta(
                pooled["mean_qlike"],
                anchor["mean_qlike"],
            ),
            "pooled_normal_delta": numeric_delta(
                pooled["normal_qlike"],
                anchor["normal_qlike"],
            ),
            "pooled_spike_delta": numeric_delta(
                pooled["spike_qlike"],
                anchor["spike_qlike"],
            ),
            "normal_not_worse_by_more_than_1pct": bool(
                pooled["normal_qlike"] <= normal_limit
            ),
            "passes_predeclared_screen": bool(
                better(pooled["mean_qlike"], anchor["mean_qlike"])
                and better(pooled["spike_qlike"], anchor["spike_qlike"])
                and spike_wins >= 3
                and pooled["normal_qlike"] <= normal_limit
            ),
        }
    return {
        "scope": "Development Fold 1-4 only; common OOS support.",
        "rule": (
            "Pooled overall and pooled spike QLIKE must beat HAR-QLIKE, "
            "normal QLIKE may be at most 1% worse, and spike QLIKE must beat "
            "HAR in at least 3/4 folds."
        ),
        "candidates": candidates,
        "statistical_claim": "None; development-only diagnostic.",
    }


def _safe_pooled_metrics(
    frame: pd.DataFrame,
    model_name: str,
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


def _run_six_configurations(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: MarketData,
    daily: pd.DataFrame,
    articles: list[FilteredArticle],
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
        name: [] for name in ("har_qlike", *CONFIGURATION_NAMES)
    }
    fold_rows: list[dict[str, Any]] = []
    spike_rows: list[dict[str, Any]] = []
    articles_by_day = _article_day_summary(articles)
    total_runs = len(folds) * len(CONFIGURATION_NAMES)
    run_index = 0
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "TAIL REGIME FOLD %d/%d | %s prepare causal features",
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
        features = _fold_feature(daily, fold, config, logger)
        write_json(
            features_dir / f"{fold.name}_metadata.json",
            features.metadata,
        )
        anchor = fit_har_qlike(market, core_dates)
        core_rv = _target_rv(market, core_dates)
        validation_rv = _target_rv(market, validation_dates)
        test_rv = _target_rv(market, test_dates)
        core_anchor = anchor.predict_log_rv(market, core_dates)
        validation_anchor = anchor.predict_log_rv(
            market, validation_dates
        )
        test_anchor = anchor.predict_log_rv(market, test_dates)
        spike_threshold = float(np.quantile(core_rv, SPIKE_QUANTILE))
        anchor_test = _prediction_frame(
            test_dates,
            test_rv,
            test_anchor,
            np.zeros(len(test_dates)),
        )
        annotated_anchor = _annotate_predictions(
            anchor_test,
            fold.name,
            "har_qlike",
            spike_threshold,
        )
        all_predictions["har_qlike"].append(annotated_anchor)
        anchor_metrics = _diagnostic_metrics(
            anchor_test,
            spike_threshold,
            "har_qlike",
        )
        anchor_metrics.update(
            {
                "fold": fold.name,
                "correction_selected": False,
            }
        )
        fold_rows.append(anchor_metrics)
        spike_rows.extend(
            _spike_information_rows(
                fold,
                annotated_anchor,
                features,
                articles_by_day,
            )
        )
        normal_matrices = tuple(
            _feature_matrix(dates, features, "normal")
            for dates in (core_dates, validation_dates, test_dates)
        )
        tail_matrices = tuple(
            _feature_matrix(dates, features, "tail")
            for dates in (core_dates, validation_dates, test_dates)
        )
        for temporal in TEMPORAL_SPECS:
            for correction_kind in CORRECTION_KINDS:
                run_index += 1
                name = f"{temporal.name}__{correction_kind}"
                logger.info(
                    "TAIL REGIME RUN %02d/%02d | fold=%s config=%s",
                    run_index,
                    total_runs,
                    fold.name,
                    name,
                )
                validation_frame, test_frame, metadata, grid = (
                    _fit_configuration(
                        name=name,
                        correction_kind=correction_kind,
                        temporal_mode=temporal.name,
                        core_dates=core_dates,
                        validation_dates=validation_dates,
                        test_dates=test_dates,
                        core_rv=core_rv,
                        validation_rv=validation_rv,
                        test_rv=test_rv,
                        core_anchor_log=core_anchor,
                        validation_anchor_log=validation_anchor,
                        test_anchor_log=test_anchor,
                        normal_matrices=normal_matrices,
                        tail_matrices=tail_matrices,
                        lambda_grid=lambda_grid,
                        min_delta=config.min_delta,
                    )
                )
                label = f"{fold.name}_{name}"
                validation_frame.to_csv(
                    predictions_dir / f"{label}_validation.csv",
                    index=False,
                )
                test_frame.to_csv(
                    predictions_dir / f"{label}.csv",
                    index=False,
                )
                grid.to_csv(
                    metrics_dir / f"{label}_lambda_grid.csv",
                    index=False,
                )
                metrics = _diagnostic_metrics(
                    test_frame,
                    spike_threshold,
                    name,
                )
                metrics.update({"fold": fold.name, **metadata})
                write_json(metrics_dir / f"{label}.json", metrics)
                fold_rows.append(metrics)
                annotated = _annotate_predictions(
                    test_frame,
                    fold.name,
                    name,
                    spike_threshold,
                )
                all_predictions[name].append(annotated)
                logger.info(
                    "TAIL REGIME OOS | fold=%s config=%s QLIKE=%.8f "
                    "normal=%s spike=%s selected=%s lambda=%s",
                    fold.name,
                    name,
                    metrics["mean_qlike"],
                    metrics["normal_qlike"],
                    metrics["spike_qlike"],
                    metadata["correction_selected"],
                    metadata["selected_lambda_sum"],
                )

    pooled_rows: list[dict[str, Any]] = []
    for name, parts in all_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        pooled.to_csv(
            predictions_dir / f"pooled_{name}.csv",
            index=False,
        )
        pooled_rows.append(_safe_pooled_metrics(pooled, name))
    fold_frame = pd.DataFrame(fold_rows)
    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    selection = _selection_screen(
        fold_frame,
        pooled_frame,
        config.min_delta,
    )
    write_json(metrics_dir / "selection_diagnostic.json", selection)

    spike_frame = pd.DataFrame(spike_rows)
    spike_frame.to_csv(
        output_dir / "audit" / "spike_information_set_audit.csv",
        index=False,
    )
    availability = (
        spike_frame["availability_class"].value_counts().to_dict()
        if len(spike_frame)
        else {}
    )
    spike_audit = {
        "spike_n": int(len(spike_frame)),
        "availability_class_counts": {
            str(key): int(value) for key, value in availability.items()
        },
        "forecast_information_rule": (
            "Only t-1 fields are model inputs. target_day_posthoc_* columns "
            "are outcome-blind availability diagnostics written after OOS "
            "predictions and are never passed to a model."
        ),
        "classification_limitation": (
            "Availability classes show whether filtered news existed; they do "
            "not prove that a specific article caused or anticipated a spike."
        ),
    }
    write_json(
        output_dir / "audit" / "spike_information_set_audit.json",
        spike_audit,
    )
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "selection": selection,
        "spike_information_audit": spike_audit,
    }


def run_development_tail_regime_diagnostic(
    config: MainPilotConfig,
    logger: logging.Logger,
    review_audit_dir: Path,
    silver_path: Path,
    longtext_cache_path: Path,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != PROFILE:
        raise ValueError("Wrong profile for tail-regime diagnostic")
    output_dir = config.output_path
    expanded = review_audit_dir / "stratified_news_filter_review.csv"
    original = (
        review_audit_dir
        / "stratified_news_filter_review_original_366.csv"
    )
    for required in (expanded, original, silver_path):
        if not required.exists():
            raise RuntimeError(
                f"Required audit artifact is missing: {required}"
            )
    base_policy = fit_event_aware_policy(expanded, original)
    policy = refined_policy(base_policy)
    signature = stable_hash(
        {
            "config": config.to_dict(),
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "expanded_review": file_fingerprint(expanded),
            "silver": file_fingerprint(silver_path),
            "policy": asdict(policy),
            "temporal_specs": [asdict(spec) for spec in TEMPORAL_SPECS],
            "correction_kinds": CORRECTION_KINDS,
            "tail_quantile": TAIL_QUANTILE,
            "spike_quantile": SPIKE_QUANTILE,
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
            logger.info("TAIL REGIME RESUME | completed report reused")
            return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The full tail-regime diagnostic is blocked "
            "on CPU; use --smoke locally."
        )
    seed_everything(config.seed)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "audit" / "event_aware_policy.json", asdict(policy))
    silver_evaluation = evaluate_filter_on_silver_holdout(
        silver_path,
        policy,
        output_dir / "audit",
        decision_fn=refined_event_decision,
        warning=(
            "Adaptive GPT-silver development diagnostic; these labels are not "
            "expert ground truth or an independent holdout."
        ),
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "profile": PROFILE,
            "scope": "Development Fold 1-4 only",
            "planned_configurations": list(CONFIGURATION_NAMES),
            "planned_fold_configuration_fits": (
                len(SPIKE_DIAGNOSTIC_FOLDS)
                * len(CONFIGURATION_NAMES)
            ),
            "seed": config.seed,
            "anchor": "Expanding-core HAR-QLIKE, fitted independently by fold",
            "tail_definition_for_gate": "Core-only RV P80",
            "spike_definition_for_reporting": "Core-only RV P90",
            "information_cutoff": "All forecast features end at t-1",
            "excluded": [
                "Fold_5",
                "final_test",
                "COVID_fold",
                "PatchTST_training",
                "chunk_embedding",
                "PCA16",
                "MCS",
                "five_seeds",
                "post_validation_refit",
            ],
            "created_utc": utc_now(),
        },
    )
    logger.info("TAIL REGIME STEP 1/5 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("TAIL REGIME STEP 2/5 | Apply refined event-aware filter")
    articles, filter_audit = load_event_aware_articles(
        Path(config.news_path),
        config.research_start,
        config.development_end,
        policy,
        logger,
        decision_fn=refined_event_decision,
    )
    write_json(
        output_dir / "audit" / "event_filter_data_audit.json",
        filter_audit,
    )
    logger.info(
        "TAIL REGIME STEP 3/5 | Reuse lead BGE/FinBERT embedding cache"
    )
    daily, embedding_audit = _build_tail_daily_frame(
        articles,
        config,
        output_dir,
        torch.device("cuda"),
        logger,
        smoke=False,
        end=config.development_end,
        cache_path=longtext_cache_path,
    )
    write_json(
        output_dir / "audit" / "embedding_audit.json",
        embedding_audit,
    )
    logger.info(
        "TAIL REGIME STEP 4/5 | Audit OOS spikes and fit 6 configurations"
    )
    results = _run_six_configurations(
        config,
        logger,
        market,
        daily,
        articles,
        output_dir,
        SPIKE_DIAGNOSTIC_FOLDS,
    )
    logger.info("TAIL REGIME STEP 5/5 | Pool common support and screen")
    report = {
        "status": "completed",
        "run_signature": signature,
        "policy": asdict(policy),
        "silver_filter_evaluation": silver_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "configurations": list(CONFIGURATION_NAMES),
        **results,
        "statistical_claim": (
            "None; convergence and competitiveness diagnostic on development "
            "Fold 1-4 only."
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("TAIL REGIME DIAGNOSTIC COMPLETED | %s", report_path)
    return report


def run_tail_regime_smoke(
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
    daily, embedding_audit = _build_tail_daily_frame(
        articles,
        smoke_config,
        smoke_dir,
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
    results = _run_six_configurations(
        smoke_config,
        logger,
        market,
        daily,
        articles,
        smoke_dir,
        (fold,),
        lambda_grid=(1.0,),
    )
    report = {
        "status": "passed",
        "backend": "CPU deterministic smoke test double",
        "metrics": str(smoke_dir / "metrics" / "pooled_metrics.csv"),
        "configurations": list(CONFIGURATION_NAMES),
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        **results,
    }
    write_json(
        smoke_dir / "tail_regime_smoke_report.json",
        report,
    )
    return report
