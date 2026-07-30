from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .baselines import fit_har_qlike
from .config import Fold, MainPilotConfig, SPIKE_DIAGNOSTIC_FOLDS
from .data import MarketData, load_market_data, write_market_audit
from .event_aware_longtext_audit import (
    CONCRETE_EVENT,
    EventAwarePolicy,
    _build_variant_daily_frames,
    _feature_matrix,
    _feature_names,
    _fold_feature,
    event_aware_decision,
    evaluate_filter_on_silver_holdout,
    fit_event_aware_policy,
    load_event_aware_articles,
)
from .losses import exact_qlike_numpy
from .metrics import prediction_metrics
from .news import DAILY_SCALAR_COLUMNS, clean_article_text
from .news_filter_labeling import _event_family_hint
from .news_representation_audit import (
    LAMBDA_SUM_GRID,
    _dated_probe_frames,
    _fit_probe,
    _prediction_frame,
    _target_rv,
)
from .pipeline import _block_dates
from .preprocess import FoldNewsFeatures
from .spike_diagnostic import _annotate_predictions, _pooled_metrics
from .utils import (
    ensure_finite,
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


PROFILE = "development-point-in-time-gate-diagnostic"
REFINED_CONTEXT_FAMILIES = (
    "regulation_etf",
    "exchange_custody",
    "macro_liquidity",
    "mining_energy",
)
DEFERRED_CONTEXT_FAMILIES = (
    "stablecoin_defi",
    "security_hack",
    "other_crypto_context",
)
COMPONENT_NAMES = (
    "har_qlike",
    "title_only_pca8_16",
    "chunk_mean_pca8_16",
    "finbert_slow_fast_6",
)
HARD_GATE_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
GATE_LOGISTIC_C = 0.1
GATE_FEATURE_NAMES = (
    "market_logrv_d",
    "market_logrv_w",
    "market_logrv_m",
    "market_logrv_d_minus_w",
    "market_logrv_w_minus_m",
    "market_logrv_change_1d",
    "market_logrv_std_5d",
    *tuple(f"news_{name}" for name in DAILY_SCALAR_COLUMNS),
    "title_semantic_fast_l2",
    "finbert_sentiment_fast_l2",
)


@dataclass(frozen=True)
class GateSpec:
    name: str
    mode: str
    normal_component: str
    spike_component: str = "title_only_pca8_16"


GATE_SPECS = (
    GateSpec("hard_gate_har_title", "hard", "har_qlike"),
    GateSpec(
        "hard_gate_finbert_title", "hard", "finbert_slow_fast_6"
    ),
    GateSpec(
        "hard_gate_chunk_title", "hard", "chunk_mean_pca8_16"
    ),
    GateSpec(
        "soft_gate_finbert_title", "soft", "finbert_slow_fast_6"
    ),
    GateSpec(
        "soft_gate_chunk_title", "soft", "chunk_mean_pca8_16"
    ),
)
GATE_NAMES = tuple(spec.name for spec in GATE_SPECS)


def refined_event_decision(
    title: str,
    cleaned_lead: str,
    policy: EventAwarePolicy,
) -> tuple[bool, int, str, bool]:
    keep, score, reason, price_recap = event_aware_decision(
        title, cleaned_lead, policy
    )
    if keep:
        return keep, score, reason, price_recap
    clean_title = clean_article_text(title)
    clean_lead = clean_article_text(cleaned_lead)[:500]
    family = _event_family_hint(clean_title, clean_lead)
    if (
        family == "direct_bitcoin"
        and CONCRETE_EVENT.search(f"{clean_title} {clean_lead}")
    ):
        return True, 2, "direct_bitcoin_concrete_repair", price_recap
    return keep, score, reason, price_recap


def refined_policy(base: EventAwarePolicy) -> EventAwarePolicy:
    return replace(
        base,
        enabled_context_families=REFINED_CONTEXT_FAMILIES,
        label_source=(
            f"{base.label_source}; family set adaptively locked after the "
            "first GPT-silver development audit"
        ),
    )


def _market_gate_features(
    market: MarketData,
    target: pd.Timestamp,
) -> np.ndarray:
    index = market.date_to_index[target]
    lag = market.log_rv[index - 22 : index]
    if len(lag) != 22 or not np.isfinite(lag).all():
        raise ValueError(f"Invalid gate market lag window for {target}")
    daily = float(lag[-1])
    weekly = float(np.mean(lag[-5:]))
    monthly = float(np.mean(lag))
    return np.asarray(
        [
            daily,
            weekly,
            monthly,
            daily - weekly,
            weekly - monthly,
            float(lag[-1] - lag[-2]),
            float(np.std(lag[-5:])),
        ],
        dtype=np.float64,
    )


def _gate_feature_row(
    market: MarketData,
    features: dict[str, FoldNewsFeatures],
    target: pd.Timestamp,
) -> np.ndarray:
    news_date = target - pd.Timedelta(days=1)
    index = int(features["lead"].dates.get_loc(news_date))
    row = np.concatenate(
        [
            _market_gate_features(market, target),
            features["lead"].daily_scalars[index].astype(np.float64),
            np.asarray(
                [
                    np.linalg.norm(
                        features["title"].semantic_fast[index]
                    ),
                    np.linalg.norm(
                        features["lead"].sentiment_fast[index]
                    ),
                ],
                dtype=np.float64,
            ),
        ]
    )
    if row.shape != (len(GATE_FEATURE_NAMES),):
        raise AssertionError(
            f"Gate feature shape {row.shape} != "
            f"({len(GATE_FEATURE_NAMES)},)"
        )
    ensure_finite("point-in-time gate feature row", row)
    return row


def _gate_feature_matrix(
    market: MarketData,
    features: dict[str, FoldNewsFeatures],
    dates: list[pd.Timestamp],
) -> np.ndarray:
    matrix = np.stack(
        [_gate_feature_row(market, features, date) for date in dates]
    )
    ensure_finite("point-in-time gate feature matrix", matrix)
    return matrix


def _binary_probability_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=bool)
    p = np.asarray(probability, dtype=np.float64)
    ensure_finite("gate probability", p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Gate probabilities must be in [0, 1]")
    both_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "spike_n": int(y.sum()),
        "spike_rate": float(y.mean()),
        "probability_mean": float(p.mean()),
        "probability_p90": float(np.quantile(p, 0.90)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(y, p))
            if y.any()
            else None
        ),
        "brier": float(brier_score_loss(y, p)),
    }


def fit_point_in_time_gate(
    market: MarketData,
    features: dict[str, FoldNewsFeatures],
    core_dates: list[pd.Timestamp],
    validation_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    matrices = {
        "core": _gate_feature_matrix(market, features, core_dates),
        "validation": _gate_feature_matrix(
            market, features, validation_dates
        ),
        "test": _gate_feature_matrix(market, features, test_dates),
    }
    core_rv = _target_rv(market, core_dates)
    threshold = float(np.quantile(core_rv, 0.90))
    labels = {
        "core": core_rv > threshold,
        "validation": _target_rv(market, validation_dates) > threshold,
        "test": _target_rv(market, test_dates) > threshold,
    }
    if len(np.unique(labels["core"])) != 2:
        raise RuntimeError("Core gate labels do not contain both classes")
    scaler = StandardScaler().fit(matrices["core"])
    scaled = {
        name: scaler.transform(values)
        for name, values in matrices.items()
    }
    model = LogisticRegression(
        C=GATE_LOGISTIC_C,
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(scaled["core"], labels["core"])
    if int(model.n_iter_[0]) >= model.max_iter:
        raise RuntimeError("Point-in-time logistic gate did not converge")
    probabilities = {
        name: model.predict_proba(values)[:, 1]
        for name, values in scaled.items()
    }
    for name, probability in probabilities.items():
        ensure_finite(f"{name} gate probabilities", probability)
    metadata = {
        "target_definition": (
            "RV_t above the 90th percentile fitted on fold core only"
        ),
        "spike_threshold": threshold,
        "fit_scope": "fold_core_only",
        "information_set": (
            "market RV lags through t-1 and news aggregates dated t-1"
        ),
        "logistic_c": GATE_LOGISTIC_C,
        "class_weight": "balanced",
        "feature_names": list(GATE_FEATURE_NAMES),
        "feature_scaler_mean": scaler.mean_.tolist(),
        "feature_scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "n_iter": int(model.n_iter_[0]),
        "blocks": {
            name: _binary_probability_metrics(
                labels[name], probabilities[name]
            )
            for name in ("core", "validation", "test")
        },
    }
    return probabilities, metadata


def _route_frame(
    normal: pd.DataFrame,
    spike: pd.DataFrame,
    probability: np.ndarray,
    mode: str,
    threshold: float | None,
    selected: bool,
) -> pd.DataFrame:
    if len(normal) != len(spike) or len(normal) != len(probability):
        raise ValueError("Route inputs have inconsistent lengths")
    if not np.allclose(normal["true_rv"], spike["true_rv"]):
        raise ValueError("Route components do not share targets")
    if mode == "hard":
        if threshold is None:
            raise ValueError("Hard gate requires a threshold")
        raw_weight = (probability >= threshold).astype(np.float64)
    elif mode == "soft":
        if threshold is not None:
            raise ValueError("Soft gate must not receive a threshold")
        raw_weight = np.asarray(probability, dtype=np.float64)
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    normal_log = normal["predicted_log_rv"].to_numpy(dtype=np.float64)
    spike_log = spike["predicted_log_rv"].to_numpy(dtype=np.float64)
    raw_log = (1.0 - raw_weight) * normal_log + raw_weight * spike_log
    anchor_log = normal["har_anchor_log_rv"].to_numpy(dtype=np.float64)
    deployed_log = raw_log if selected else anchor_log
    deployed_weight = raw_weight if selected else np.zeros_like(raw_weight)
    output = normal[
        ["target_date", "true_rv", "true_log_rv", "har_anchor_log_rv"]
    ].copy()
    output["gate_probability"] = probability
    output["raw_gate_weight"] = raw_weight
    output["gate_weight"] = deployed_weight
    output["normal_component_log_rv"] = normal_log
    output["spike_component_log_rv"] = spike_log
    output["raw_gate_predicted_log_rv"] = raw_log
    output["predicted_log_rv"] = deployed_log
    output["predicted_rv"] = np.exp(deployed_log)
    output["delta_log_rv"] = deployed_log - anchor_log
    ensure_finite(
        "point-in-time gate predicted RV",
        output["predicted_rv"].to_numpy(),
    )
    return output


def _validation_qlike(frame: pd.DataFrame) -> float:
    return float(
        np.mean(
            exact_qlike_numpy(
                frame["true_rv"].to_numpy(dtype=np.float64),
                frame["predicted_rv"].to_numpy(dtype=np.float64),
            )
        )
    )


def select_and_apply_gate(
    spec: GateSpec,
    validation_components: dict[str, pd.DataFrame],
    test_components: dict[str, pd.DataFrame],
    validation_probability: np.ndarray,
    test_probability: np.ndarray,
    min_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    anchor_validation = _validation_qlike(
        validation_components["har_qlike"]
    )
    thresholds: tuple[float | None, ...] = (
        tuple(HARD_GATE_THRESHOLDS)
        if spec.mode == "hard"
        else (None,)
    )
    rows: list[dict[str, Any]] = []
    best_threshold: float | None = None
    best_qlike = float("inf")
    best_raw_validation: pd.DataFrame | None = None
    for threshold in thresholds:
        raw = _route_frame(
            validation_components[spec.normal_component],
            validation_components[spec.spike_component],
            validation_probability,
            spec.mode,
            threshold,
            selected=True,
        )
        value = _validation_qlike(raw)
        rows.append(
            {
                "model": spec.name,
                "mode": spec.mode,
                "normal_component": spec.normal_component,
                "spike_component": spec.spike_component,
                "gate_threshold": threshold,
                "validation_qlike": value,
            }
        )
        better = value < best_qlike - min_delta
        tied_conservative = (
            abs(value - best_qlike) <= min_delta
            and spec.mode == "hard"
            and (
                best_threshold is None
                or (
                    threshold is not None
                    and threshold > best_threshold
                )
            )
        )
        if better or tied_conservative:
            best_qlike = value
            best_threshold = threshold
            best_raw_validation = raw
    if best_raw_validation is None:
        raise RuntimeError(f"No valid gate configuration for {spec.name}")
    selected = bool(best_qlike < anchor_validation - min_delta)
    validation = _route_frame(
        validation_components[spec.normal_component],
        validation_components[spec.spike_component],
        validation_probability,
        spec.mode,
        best_threshold,
        selected,
    )
    test = _route_frame(
        test_components[spec.normal_component],
        test_components[spec.spike_component],
        test_probability,
        spec.mode,
        best_threshold,
        selected,
    )
    metadata = {
        "model": spec.name,
        "mode": spec.mode,
        "normal_component": spec.normal_component,
        "spike_component": spec.spike_component,
        "selected_gate_threshold": best_threshold,
        "anchor_validation_qlike": anchor_validation,
        "candidate_validation_qlike": best_qlike,
        "validation_improvement": anchor_validation - best_qlike,
        "correction_selected": selected,
        "validation_raw_gate_weight_mean": float(
            best_raw_validation["raw_gate_weight"].mean()
        ),
        "test_raw_gate_weight_mean": float(
            test["raw_gate_weight"].mean()
        ),
        "selection_rule": (
            "Choose minimum validation exact QLIKE; within 1e-5 prefer "
            "the higher hard threshold. Deploy only if validation beats "
            "HAR by at least 1e-5."
        ),
    }
    return validation, test, metadata, pd.DataFrame(rows)


def _gate_screen(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    min_delta: float,
) -> dict[str, Any]:
    def finite(value: Any) -> bool:
        return value is not None and bool(np.isfinite(value))

    def better(candidate: Any, anchor: Any) -> bool:
        return finite(candidate) and finite(anchor) and candidate < anchor - min_delta

    def not_worse(candidate: Any, anchor: Any) -> bool:
        if not finite(anchor):
            return not finite(candidate)
        return finite(candidate) and candidate <= anchor + min_delta

    def delta(candidate: Any, anchor: Any) -> float | None:
        if not (finite(candidate) and finite(anchor)):
            return None
        return float(candidate - anchor)

    anchor_fold = fold_metrics[
        fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    anchor_pooled = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    candidates: dict[str, Any] = {}
    for name in GATE_NAMES:
        candidate_fold = fold_metrics[
            fold_metrics["model"] == name
        ].set_index("fold")
        pooled = pooled_metrics[pooled_metrics["model"] == name].iloc[0]
        overall_wins = sum(
            better(
                candidate_fold.loc[fold, "mean_qlike"],
                anchor_fold.loc[fold, "mean_qlike"],
            )
            for fold in anchor_fold.index
        )
        normal_wins = sum(
            better(
                candidate_fold.loc[fold, "normal_qlike"],
                anchor_fold.loc[fold, "normal_qlike"],
            )
            for fold in anchor_fold.index
        )
        spike_wins = sum(
            better(
                candidate_fold.loc[fold, "spike_qlike"],
                anchor_fold.loc[fold, "spike_qlike"],
            )
            for fold in anchor_fold.index
        )
        passes = bool(
            overall_wins >= 3
            and better(pooled["mean_qlike"], anchor_pooled["mean_qlike"])
            and not_worse(
                pooled["normal_qlike"], anchor_pooled["normal_qlike"]
            )
            and not_worse(
                pooled["spike_qlike"], anchor_pooled["spike_qlike"]
            )
        )
        candidates[name] = {
            "overall_fold_wins": int(overall_wins),
            "normal_fold_wins": int(normal_wins),
            "spike_fold_wins": int(spike_wins),
            "pooled_overall_delta": delta(
                pooled["mean_qlike"], anchor_pooled["mean_qlike"]
            ),
            "pooled_normal_delta": delta(
                pooled["normal_qlike"], anchor_pooled["normal_qlike"]
            ),
            "pooled_spike_delta": delta(
                pooled["spike_qlike"], anchor_pooled["spike_qlike"]
            ),
            "passes_gate_screen": passes,
            "recommended_for_deep_followup": passes,
        }
    return {
        "scope": "adaptive development-only rolling Fold 1-4, seed 11",
        "rule": (
            "Beat HAR overall in at least 3/4 folds and pooled, while pooled "
            "normal and spike QLIKE are each not worse than HAR."
        ),
        "candidates": candidates,
        "statistical_claim": "None; adaptive development gate diagnostic.",
    }


def _run_gate_folds(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: MarketData,
    daily: dict[str, pd.DataFrame],
    output_dir: Path,
    folds: tuple[Fold, ...],
) -> dict[str, Any]:
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    features_dir = output_dir / "features"
    for directory in (predictions_dir, metrics_dir, features_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model_names = (*COMPONENT_NAMES, *GATE_NAMES)
    all_predictions: dict[str, list[pd.DataFrame]] = {
        name: [] for name in model_names
    }
    fold_rows: list[dict[str, Any]] = []
    gate_diagnostics: dict[str, Any] = {}
    probe_definitions = {
        "title_only_pca8_16": "title_only_pca8_16",
        "chunk_mean_pca8_16": "chunk_mean_pca8_16",
        "finbert_slow_fast_6": "finbert_slow_fast_6",
    }
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "GATE FOLD %d/%d | %s core-only PCA/probes/gate",
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
        if not config.smoke and len(validation_dates) < 60:
            raise RuntimeError(
                f"{fold.name} has only {len(validation_dates)} validation "
                "targets (<60)"
            )
        features = {
            name: _fold_feature(frame, fold, config, logger)
            for name, frame in daily.items()
        }
        write_json(
            features_dir / f"{fold.name}_metadata.json",
            {name: value.metadata for name, value in features.items()},
        )
        core_rv = _target_rv(market, core_dates)
        validation_rv = _target_rv(market, validation_dates)
        test_rv = _target_rv(market, test_dates)
        anchor = fit_har_qlike(market, core_dates)
        core_anchor = anchor.predict_log_rv(market, core_dates)
        validation_anchor = anchor.predict_log_rv(
            market, validation_dates
        )
        test_anchor = anchor.predict_log_rv(market, test_dates)
        threshold = float(np.quantile(core_rv, 0.90))
        validation_components = {
            "har_qlike": _prediction_frame(
                validation_dates,
                validation_rv,
                validation_anchor,
                np.zeros(len(validation_dates)),
            )
        }
        test_components = {
            "har_qlike": _prediction_frame(
                test_dates,
                test_rv,
                test_anchor,
                np.zeros(len(test_dates)),
            )
        }
        component_metadata: dict[str, Any] = {
            "har_qlike": anchor.metadata()
        }
        for name in probe_definitions:
            matrices = [
                _feature_matrix(dates, features, name)
                for dates in (core_dates, validation_dates, test_dates)
            ]
            result = _fit_probe(
                name,
                matrices[0],
                matrices[1],
                matrices[2],
                core_rv,
                validation_rv,
                test_rv,
                core_anchor,
                validation_anchor,
                test_anchor,
                _feature_names(name),
                LAMBDA_SUM_GRID,
                config.min_delta,
            )
            validation, test, metadata, grid = _dated_probe_frames(
                validation_dates, test_dates, result
            )
            validation_components[name] = validation
            test_components[name] = test
            component_metadata[name] = metadata
            grid.to_csv(
                metrics_dir / f"{fold.name}_{name}_lambda_grid.csv",
                index=False,
            )
        probabilities, gate_metadata = fit_point_in_time_gate(
            market,
            features,
            core_dates,
            validation_dates,
            test_dates,
            config.seed,
        )
        gate_diagnostics[fold.name] = {
            "gate_model": gate_metadata,
            "components": component_metadata,
        }
        write_json(
            features_dir / f"{fold.name}_gate_metadata.json",
            gate_diagnostics[fold.name],
        )
        fold_frames: dict[str, pd.DataFrame] = {
            **test_components,
        }
        for spec in GATE_SPECS:
            validation, test, metadata, grid = select_and_apply_gate(
                spec,
                validation_components,
                test_components,
                probabilities["validation"],
                probabilities["test"],
                config.min_delta,
            )
            validation.to_csv(
                predictions_dir
                / f"{fold.name}_{spec.name}_validation.csv",
                index=False,
            )
            test.to_csv(
                predictions_dir / f"{fold.name}_{spec.name}.csv",
                index=False,
            )
            grid.to_csv(
                metrics_dir / f"{fold.name}_{spec.name}_grid.csv",
                index=False,
            )
            fold_frames[spec.name] = test
            gate_diagnostics[fold.name][spec.name] = metadata
            logger.info(
                "GATE SELECT | fold=%s model=%s selected=%s "
                "threshold=%s val_gain=%.8f",
                fold.name,
                spec.name,
                metadata["correction_selected"],
                metadata["selected_gate_threshold"],
                metadata["validation_improvement"],
            )
        for name, frame in fold_frames.items():
            frame.to_csv(
                predictions_dir / f"{fold.name}_{name}.csv",
                index=False,
            )
            metrics = prediction_metrics(frame, threshold, name)
            metrics["fold"] = fold.name
            if name in GATE_NAMES:
                metrics.update(gate_diagnostics[fold.name][name])
            elif name != "har_qlike":
                metrics.update(component_metadata[name])
            else:
                metrics.update(
                    {
                        "correction_selected": False,
                        "validation_improvement": 0.0,
                    }
                )
            write_json(
                metrics_dir / f"{fold.name}_{name}.json", metrics
            )
            fold_rows.append(metrics)
            annotated = _annotate_predictions(
                frame, fold.name, name, threshold
            )
            all_predictions[name].append(annotated)
            logger.info(
                "GATE OOS | fold=%s model=%s QLIKE=%.8f normal=%s "
                "spike=%s",
                fold.name,
                name,
                metrics["mean_qlike"],
                metrics["normal_qlike"],
                metrics["spike_qlike"],
            )
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
    screen = _gate_screen(
        fold_frame, pooled_frame, config.min_delta
    )
    write_json(metrics_dir / "gate_screen.json", screen)
    write_json(metrics_dir / "gate_diagnostics.json", gate_diagnostics)
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "gate_screen": screen,
        "gate_diagnostics": gate_diagnostics,
    }


def run_development_point_in_time_gate_diagnostic(
    config: MainPilotConfig,
    logger: logging.Logger,
    review_audit_dir: Path,
    silver_path: Path,
    longtext_cache_path: Path,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != PROFILE:
        raise ValueError("Wrong profile for point-in-time gate diagnostic")
    output_dir = config.output_path
    expanded = (
        review_audit_dir / "stratified_news_filter_review.csv"
    )
    original = (
        review_audit_dir / "stratified_news_filter_review_original_366.csv"
    )
    for required in (expanded, original, silver_path):
        if not required.exists():
            raise RuntimeError(f"Required audit artifact missing: {required}")
    base_policy = fit_event_aware_policy(expanded, original)
    policy = refined_policy(base_policy)
    signature = stable_hash(
        {
            "config": config.to_dict(),
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "expanded": file_fingerprint(expanded),
            "silver": file_fingerprint(silver_path),
            "policy": asdict(policy),
            "gate_specs": [asdict(spec) for spec in GATE_SPECS],
            "hard_thresholds": HARD_GATE_THRESHOLDS,
            "logistic_c": GATE_LOGISTIC_C,
        }
    )
    report_path = output_dir / "metrics" / "diagnostic_report.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info("GATE RESUME | completed report reused")
            return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full gate diagnostic embedding is blocked "
            "on CPU; use --smoke."
        )
    seed_everything(config.seed)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "audit" / "refined_event_policy.json",
        {
            **asdict(policy),
            "deferred_context_families": DEFERRED_CONTEXT_FAMILIES,
            "direct_bitcoin_repair": "require concrete event language",
            "adaptive_warning": (
                "Family choices were made after inspecting the first "
                "GPT-silver development audit."
            ),
        },
    )
    filter_evaluation = evaluate_filter_on_silver_holdout(
        silver_path,
        policy,
        output_dir / "audit",
        decision_fn=refined_event_decision,
        warning=(
            "Adaptive calibration diagnostic only: these GPT-silver rows "
            "influenced the refined family policy and are no longer an "
            "independent holdout or expert ground truth."
        ),
    )
    logger.info("GATE STEP 1/4 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("GATE STEP 2/4 | Apply refined event-aware filter")
    articles, filter_audit = load_event_aware_articles(
        Path(config.news_path),
        config.research_start,
        config.development_end,
        policy,
        logger,
        decision_fn=refined_event_decision,
    )
    write_json(
        output_dir / "audit" / "refined_filter_data_audit.json",
        filter_audit,
    )
    logger.info(
        "GATE STEP 3/4 | Reuse/cache BGE and FinBERT representations"
    )
    daily, embedding_audit = _build_variant_daily_frames(
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
        "GATE STEP 4/4 | Fit core-only gate and evaluate Fold 1-4"
    )
    results = _run_gate_folds(
        config,
        logger,
        market,
        daily,
        output_dir,
        SPIKE_DIAGNOSTIC_FOLDS,
    )
    report = {
        "status": "completed",
        "run_signature": signature,
        "profile": PROFILE,
        "policy": asdict(policy),
        "adaptive_filter_evaluation": filter_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        **results,
        "excluded": [
            "Fold_5",
            "final_test",
            "PatchTST_retraining",
            "MCS",
            "five_seeds",
            "realized_spike_oracle_as_gate_input",
        ],
        "statistical_claim": (
            "None; adaptive development-only convergence and routing "
            "diagnostic."
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("POINT-IN-TIME GATE COMPLETED | %s", report_path)
    return report


def run_point_in_time_gate_smoke(
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
        enabled_context_families=REFINED_CONTEXT_FAMILIES,
        relevant_rate_threshold=0.0,
        minimum_family_examples=0,
        training_rows=0,
        holdout_rows=0,
        family_statistics=tuple(),
        label_source="synthetic refined smoke policy",
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
    daily, embedding_audit = _build_variant_daily_frames(
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
        "2018-04-30",
        "2018-05-01",
        "2018-05-15",
        "2018-05-16",
        "2018-05-31",
    )
    results = _run_gate_folds(
        smoke_config,
        logger,
        market,
        daily,
        smoke_dir,
        (fold,),
    )
    report_path = smoke_dir / "point_in_time_gate_smoke.json"
    report = {
        "status": "passed",
        "metrics": str(report_path),
        "backend": "CPU deterministic smoke test double",
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        **results,
    }
    write_json(report_path, report)
    return report
