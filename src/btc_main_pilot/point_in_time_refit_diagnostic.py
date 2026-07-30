from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler

from .baselines import fit_har_qlike
from .config import Fold, MainPilotConfig, SPIKE_DIAGNOSTIC_FOLDS
from .data import MarketData, load_market_data, write_market_audit
from .event_aware_longtext_audit import (
    EventAwarePolicy,
    _build_variant_daily_frames,
    _feature_matrix,
    _feature_names,
    _fold_feature,
    evaluate_filter_on_silver_holdout,
    fit_event_aware_policy,
    load_event_aware_articles,
)
from .metrics import prediction_metrics
from .news_representation_audit import _prediction_frame, _target_rv
from .pipeline import _block_dates
from .point_in_time_gate_diagnostic import (
    COMPONENT_NAMES,
    DEFERRED_CONTEXT_FAMILIES,
    GATE_FEATURE_NAMES,
    GATE_LOGISTIC_C,
    GATE_NAMES,
    GATE_SPECS,
    HARD_GATE_THRESHOLDS,
    REFINED_CONTEXT_FAMILIES,
    _binary_probability_metrics,
    _gate_feature_matrix,
    _gate_screen,
    _route_frame,
    _run_gate_folds,
    refined_event_decision,
    refined_policy,
)
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


PROFILE = "development-point-in-time-refit-diagnostic"
COMPARISON_METRICS = (
    "mean_qlike",
    "normal_qlike",
    "spike_qlike",
    "r2_logrv",
    "rmse_logrv",
    "mae_logrv",
)


def _refit_preprocess_fold(fold: Fold) -> Fold:
    """Fit news preprocessing through validation, never through test."""
    return Fold(
        name=f"{fold.name}_core_validation_refit",
        core_start=fold.core_start,
        core_end=fold.validation_end,
        validation_start=fold.test_start,
        validation_end=fold.test_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
    )


def _fit_locked_refit_probe(
    name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_rv: np.ndarray,
    test_rv: np.ndarray,
    train_anchor_log: np.ndarray,
    test_anchor_log: np.ndarray,
    test_dates: list[pd.Timestamp],
    feature_names: list[str],
    locked_selection: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit a validation-selected probe without making another selection."""
    selected = bool(locked_selection["correction_selected"])
    lambda_sum_raw = locked_selection.get("selected_lambda_sum")
    metadata: dict[str, Any] = {
        "model": name,
        "fit_scope": "fold_core_plus_validation",
        "selection_scope": "fold_core_fit_validation_select",
        "selection_locked_before_refit": True,
        "correction_selected": selected,
        "selected_lambda_sum": lambda_sum_raw,
        "feature_dim": int(x_train.shape[1]),
        "ordered_feature_names": feature_names,
        "refit_n": int(len(train_rv)),
    }
    if not selected:
        metadata.update(
            {
                "refit_status": "locked_har_fallback",
                "selected_alpha_mean_loss": None,
                "coefficient_l2": 0.0,
                "coefficients": [],
                "intercept": None,
            }
        )
        return (
            _prediction_frame(
                test_dates,
                test_rv,
                test_anchor_log,
                np.zeros(len(test_rv), dtype=np.float64),
            ),
            metadata,
        )
    if lambda_sum_raw is None:
        raise RuntimeError(
            f"Selected probe {name} has no locked lambda_sum"
        )
    lambda_sum = float(lambda_sum_raw)
    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    test_scaled = scaler.transform(x_test)
    relative_train = train_rv / np.exp(train_anchor_log)
    ensure_finite(f"{name} refit relative target", relative_train)
    alpha = 2.0 * lambda_sum / len(train_rv)
    model = GammaRegressor(
        alpha=alpha,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-8,
        warm_start=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_scaled, relative_train)
    if any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    ):
        raise RuntimeError(f"Refit probe {name} did not converge")
    relative_test = model.predict(test_scaled)
    ensure_finite(f"{name} refit relative prediction", relative_test)
    if np.any(relative_test <= 0):
        raise RuntimeError(
            f"Refit probe {name} produced a nonpositive multiplier"
        )
    correction = np.log(relative_test)
    metadata.update(
        {
            "refit_status": "fitted",
            "selected_alpha_mean_loss": alpha,
            "coefficient_l2": float(np.linalg.norm(model.coef_)),
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "feature_scaler_mean": scaler.mean_.tolist(),
            "feature_scaler_scale": scaler.scale_.tolist(),
        }
    )
    return (
        _prediction_frame(
            test_dates,
            test_rv,
            test_anchor_log,
            correction,
        ),
        metadata,
    )


def _fit_refit_gate(
    market: MarketData,
    features: dict[str, FoldNewsFeatures],
    train_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    locked_spike_threshold: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit the gate on core+validation with its target definition locked."""
    train_matrix = _gate_feature_matrix(market, features, train_dates)
    test_matrix = _gate_feature_matrix(market, features, test_dates)
    train_label = (
        _target_rv(market, train_dates) > locked_spike_threshold
    )
    test_label = _target_rv(market, test_dates) > locked_spike_threshold
    if len(np.unique(train_label)) != 2:
        raise RuntimeError(
            "Core+validation refit gate labels do not contain both classes"
        )
    scaler = StandardScaler().fit(train_matrix)
    train_scaled = scaler.transform(train_matrix)
    test_scaled = scaler.transform(test_matrix)
    model = LogisticRegression(
        C=GATE_LOGISTIC_C,
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(train_scaled, train_label)
    if int(model.n_iter_[0]) >= model.max_iter:
        raise RuntimeError("Core+validation refit gate did not converge")
    train_probability = model.predict_proba(train_scaled)[:, 1]
    test_probability = model.predict_proba(test_scaled)[:, 1]
    ensure_finite("refit train gate probability", train_probability)
    ensure_finite("refit test gate probability", test_probability)
    metadata = {
        "target_definition": (
            "RV_t above the core-only 90th percentile locked during "
            "core/validation selection"
        ),
        "spike_threshold": float(locked_spike_threshold),
        "fit_scope": "fold_core_plus_validation",
        "selection_locked_before_refit": True,
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
            "core_plus_validation": _binary_probability_metrics(
                train_label, train_probability
            ),
            "test": _binary_probability_metrics(
                test_label, test_probability
            ),
        },
    }
    return test_probability, metadata


def _comparison_frame(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> pd.DataFrame:
    before = {
        tuple(row[key] for key in keys): row for row in before_rows
    }
    after = {
        tuple(row[key] for key in keys): row for row in after_rows
    }
    if before.keys() != after.keys():
        raise RuntimeError("Before/after metric keys do not match")
    rows: list[dict[str, Any]] = []
    for key in sorted(before):
        output = {
            name: value for name, value in zip(keys, key, strict=True)
        }
        for metric in COMPARISON_METRICS:
            before_value = before[key].get(metric)
            after_value = after[key].get(metric)
            output[f"before_{metric}"] = before_value
            output[f"after_{metric}"] = after_value
            if before_value is None or after_value is None:
                output[f"delta_{metric}"] = None
                output[f"relative_change_pct_{metric}"] = None
                continue
            before_float = float(before_value)
            after_float = float(after_value)
            output[f"delta_{metric}"] = after_float - before_float
            output[f"relative_change_pct_{metric}"] = (
                100.0 * (after_float - before_float) / abs(before_float)
                if before_float != 0.0
                else None
            )
        rows.append(output)
    return pd.DataFrame(rows)


def _run_refit_folds(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: MarketData,
    daily: dict[str, pd.DataFrame],
    output_dir: Path,
    folds: tuple[Fold, ...],
    selection_results: dict[str, Any],
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
    refit_diagnostics: dict[str, Any] = {}
    selected = selection_results["gate_diagnostics"]
    probe_names = COMPONENT_NAMES[1:]
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "REFIT FOLD %d/%d | %s fit core+validation, test untouched",
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
        train_dates = sorted({*core_dates, *validation_dates})
        if len(train_dates) != len(core_dates) + len(validation_dates):
            raise RuntimeError(
                f"{fold.name} core and validation target dates overlap"
            )
        preprocess_fold = _refit_preprocess_fold(fold)
        features = {
            name: _fold_feature(frame, preprocess_fold, config, logger)
            for name, frame in daily.items()
        }
        write_json(
            features_dir / f"{fold.name}_refit_metadata.json",
            {name: value.metadata for name, value in features.items()},
        )
        train_rv = _target_rv(market, train_dates)
        test_rv = _target_rv(market, test_dates)
        anchor = fit_har_qlike(market, train_dates)
        train_anchor = anchor.predict_log_rv(market, train_dates)
        test_anchor = anchor.predict_log_rv(market, test_dates)
        selection = selected[fold.name]
        locked_threshold = float(
            selection["gate_model"]["spike_threshold"]
        )
        test_components: dict[str, pd.DataFrame] = {
            "har_qlike": _prediction_frame(
                test_dates,
                test_rv,
                test_anchor,
                np.zeros(len(test_dates), dtype=np.float64),
            )
        }
        component_metadata: dict[str, Any] = {
            "har_qlike": {
                **anchor.metadata(),
                "fit_scope": "fold_core_plus_validation",
                "refit_n": len(train_dates),
            }
        }
        for name in probe_names:
            x_train = _feature_matrix(train_dates, features, name)
            x_test = _feature_matrix(test_dates, features, name)
            frame, metadata = _fit_locked_refit_probe(
                name=name,
                x_train=x_train,
                x_test=x_test,
                train_rv=train_rv,
                test_rv=test_rv,
                train_anchor_log=train_anchor,
                test_anchor_log=test_anchor,
                test_dates=test_dates,
                feature_names=_feature_names(name),
                locked_selection=selection["components"][name],
            )
            test_components[name] = frame
            component_metadata[name] = metadata
        test_probability, gate_metadata = _fit_refit_gate(
            market,
            features,
            train_dates,
            test_dates,
            locked_threshold,
            config.seed,
        )
        refit_diagnostics[fold.name] = {
            "train_scope": {
                "core_n": len(core_dates),
                "validation_n": len(validation_dates),
                "core_plus_validation_n": len(train_dates),
                "test_n": len(test_dates),
                "core_start": fold.core_start,
                "validation_end": fold.validation_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
            },
            "gate_model": gate_metadata,
            "components": component_metadata,
            "locked_selection": {
                name: selection[name] for name in GATE_NAMES
            },
        }
        fold_frames = {**test_components}
        for spec in GATE_SPECS:
            locked = selection[spec.name]
            threshold = locked["selected_gate_threshold"]
            if spec.mode == "hard" and threshold is None:
                raise RuntimeError(
                    f"Locked hard gate {spec.name} has no threshold"
                )
            if spec.mode == "soft" and threshold is not None:
                raise RuntimeError(
                    f"Locked soft gate {spec.name} has a threshold"
                )
            frame = _route_frame(
                test_components[spec.normal_component],
                test_components[spec.spike_component],
                test_probability,
                spec.mode,
                threshold,
                selected=bool(locked["correction_selected"]),
            )
            fold_frames[spec.name] = frame
            refit_diagnostics[fold.name][spec.name] = {
                "model": spec.name,
                "mode": spec.mode,
                "normal_component": spec.normal_component,
                "spike_component": spec.spike_component,
                "selected_gate_threshold": threshold,
                "correction_selected": bool(
                    locked["correction_selected"]
                ),
                "selection_locked_before_refit": True,
                "selection_validation_improvement": locked[
                    "validation_improvement"
                ],
                "test_raw_gate_weight_mean": float(
                    frame["raw_gate_weight"].mean()
                ),
                "fit_scope": "fold_core_plus_validation",
            }
        write_json(
            features_dir / f"{fold.name}_refit_gate_metadata.json",
            refit_diagnostics[fold.name],
        )
        for name, frame in fold_frames.items():
            frame.to_csv(
                predictions_dir / f"{fold.name}_{name}.csv",
                index=False,
            )
            metrics = prediction_metrics(
                frame, locked_threshold, name
            )
            metrics.update(
                {
                    "fold": fold.name,
                    "stage": "after_refit",
                    "fit_scope": "fold_core_plus_validation",
                    "selection_locked_before_refit": True,
                }
            )
            if name in GATE_NAMES:
                metrics.update(refit_diagnostics[fold.name][name])
            elif name != "har_qlike":
                metrics.update(component_metadata[name])
            else:
                metrics.update(
                    {
                        "correction_selected": False,
                        "refit_n": len(train_dates),
                    }
                )
            write_json(
                metrics_dir / f"{fold.name}_{name}.json", metrics
            )
            fold_rows.append(metrics)
            annotated = _annotate_predictions(
                frame, fold.name, name, locked_threshold
            )
            all_predictions[name].append(annotated)
            logger.info(
                "REFIT OOS | fold=%s model=%s QLIKE=%.8f normal=%s "
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
    screen["scope"] = (
        "adaptive development-only Fold 1-4 after locked-selection "
        "core+validation refit, seed 11"
    )
    screen["statistical_claim"] = (
        "None; post-hoc development refit diagnostic."
    )
    write_json(metrics_dir / "gate_screen.json", screen)
    write_json(
        metrics_dir / "refit_diagnostics.json", refit_diagnostics
    )
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "gate_screen": screen,
        "refit_diagnostics": refit_diagnostics,
    }


def _write_comparison(
    output_dir: Path,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold = _comparison_frame(
        before["fold_metrics"],
        after["fold_metrics"],
        ("fold", "model"),
    )
    pooled = _comparison_frame(
        before["pooled_metrics"],
        after["pooled_metrics"],
        ("model",),
    )
    fold.to_csv(
        metrics_dir / "before_after_fold_metrics.csv", index=False
    )
    pooled.to_csv(
        metrics_dir / "before_after_pooled_metrics.csv", index=False
    )
    summary = {
        "interpretation": (
            "after_refit minus before_refit; negative QLIKE deltas favor "
            "core+validation refit"
        ),
        "before_gate_screen": before["gate_screen"],
        "after_gate_screen": after["gate_screen"],
        "fold_comparison": fold.to_dict(orient="records"),
        "pooled_comparison": pooled.to_dict(orient="records"),
    }
    write_json(metrics_dir / "before_after_summary.json", summary)
    return summary


def run_development_point_in_time_refit_diagnostic(
    config: MainPilotConfig,
    logger: logging.Logger,
    review_audit_dir: Path,
    silver_path: Path,
    longtext_cache_path: Path,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != PROFILE:
        raise ValueError("Wrong profile for point-in-time refit diagnostic")
    output_dir = config.output_path
    expanded = review_audit_dir / "stratified_news_filter_review.csv"
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
            "refit_rule": (
                "select on core/validation; lock; refit core+validation; "
                "evaluate unchanged test"
            ),
        }
    )
    report_path = output_dir / "metrics" / "diagnostic_report.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info("REFIT RESUME | completed report reused")
            return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full refit diagnostic embedding is blocked "
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
    logger.info("REFIT STEP 1/5 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("REFIT STEP 2/5 | Apply refined event-aware filter")
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
        "REFIT STEP 3/5 | Reuse/cache BGE and FinBERT representations"
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
        "REFIT STEP 4/5 | Select on core/validation and save before-refit OOS"
    )
    before = _run_gate_folds(
        config,
        logger,
        market,
        daily,
        output_dir / "before_refit",
        SPIKE_DIAGNOSTIC_FOLDS,
    )
    logger.info(
        "REFIT STEP 5/5 | Lock selections, refit core+validation, "
        "evaluate same tests"
    )
    after = _run_refit_folds(
        config,
        logger,
        market,
        daily,
        output_dir / "after_refit",
        SPIKE_DIAGNOSTIC_FOLDS,
        before,
    )
    comparison = _write_comparison(output_dir, before, after)
    report = {
        "status": "completed",
        "run_signature": signature,
        "profile": PROFILE,
        "policy": asdict(policy),
        "adaptive_filter_evaluation": filter_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "selection_rule": (
            "Select lambda, correction fallback, gate mode, and hard threshold "
            "using the original core-fit/validation-evaluate stage only."
        ),
        "refit_rule": (
            "After locking selection, refit HAR, fold-local PCA/scalers, "
            "selected Gamma probes, and logistic gate on core+validation. "
            "Use the locked core spike definition and evaluate the unchanged "
            "test once."
        ),
        "before_refit": before,
        "after_refit": after,
        "comparison": comparison,
        "excluded": [
            "Fold_5",
            "final_test",
            "PatchTST_retraining",
            "MCS",
            "five_seeds",
            "realized_spike_oracle_as_gate_input",
            "test_based_model_selection",
        ],
        "statistical_claim": (
            "None; post-hoc adaptive development-only refit diagnostic."
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("POINT-IN-TIME REFIT COMPLETED | %s", report_path)
    return report


def run_point_in_time_refit_smoke(
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
        label_source="synthetic refined refit smoke policy",
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
    before = _run_gate_folds(
        smoke_config,
        logger,
        market,
        daily,
        smoke_dir / "before_refit",
        (fold,),
    )
    after = _run_refit_folds(
        smoke_config,
        logger,
        market,
        daily,
        smoke_dir / "after_refit",
        (fold,),
        before,
    )
    comparison = _write_comparison(smoke_dir, before, after)
    report_path = smoke_dir / "point_in_time_refit_smoke.json"
    report = {
        "status": "passed",
        "metrics": str(report_path),
        "backend": "CPU deterministic smoke test double",
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "before_refit": before,
        "after_refit": after,
        "comparison": comparison,
    }
    write_json(report_path, report)
    return report
