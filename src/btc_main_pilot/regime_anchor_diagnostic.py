from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines import fit_baselines, fit_har_qlike
from .config import (
    Fold,
    MainPilotConfig,
    REGIME_ANCHOR_VARIANTS,
    SPIKE_DIAGNOSTIC_FOLDS,
)
from .data import load_market_data, write_market_audit
from .model import attention_backend_metadata, build_model
from .pipeline import (
    _datasets,
    _initialization_assertion,
    _prepare_news,
    _require_numerically_successful,
)
from .spike_diagnostic import (
    _annotate_predictions,
    _completed_run,
    _diagnostic_metrics,
    _pooled_metrics,
    _spike_threshold,
    _validate_main_scheduler,
)
from .training import predict_dataset, train_model
from .utils import (
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


def _initial_anchor_rv(
    dataset: Any,
    target_mean: float,
    target_scale: float,
    intercept: float,
    coefficients: np.ndarray,
) -> float:
    standardized = dataset[0]["har_scalars"].numpy().astype(np.float64)
    raw_har = standardized * target_scale + target_mean
    return float(np.exp(intercept + raw_har @ coefficients))


def _correction_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    delta = predictions["delta_log_rv"].to_numpy(dtype=np.float64)
    anchor_log = predictions["har_anchor_log_rv"].to_numpy(dtype=np.float64)
    true_log = predictions["true_log_rv"].to_numpy(dtype=np.float64)
    anchor_gap = true_log - anchor_log
    return {
        "correction_mean_logrv": float(np.mean(delta)),
        "correction_std_logrv": float(np.std(delta)),
        "correction_mean_abs_logrv": float(np.mean(np.abs(delta))),
        "correction_max_abs_logrv": float(np.max(np.abs(delta))),
        "correction_target_correlation": (
            float(np.corrcoef(delta, anchor_gap)[0, 1])
            if np.std(delta) > 0 and np.std(anchor_gap) > 0
            else None
        ),
    }


def _anchor_selection(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
) -> dict[str, Any]:
    min_delta = 1e-5
    har_folds = fold_metrics[fold_metrics["model"] == "har_qlike"].set_index(
        "fold"
    )
    har_pooled = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    candidates: dict[str, Any] = {}
    for variant in REGIME_ANCHOR_VARIANTS:
        candidate_folds = fold_metrics[
            fold_metrics["model"] == variant
        ].set_index("fold")
        common = sorted(set(har_folds.index) & set(candidate_folds.index))
        overall_wins = sum(
            candidate_folds.loc[fold, "mean_qlike"]
            < har_folds.loc[fold, "mean_qlike"] - min_delta
            for fold in common
        )
        spike_wins = sum(
            candidate_folds.loc[fold, "spike_qlike"]
            < har_folds.loc[fold, "spike_qlike"] - min_delta
            for fold in common
        )
        pooled = pooled_metrics[pooled_metrics["model"] == variant].iloc[0]
        candidates[variant] = {
            "correction_selected_folds": int(
                sum(
                    bool(value)
                    for value in candidate_folds[
                        "correction_selected"
                    ].tolist()
                    if value is not None and not pd.isna(value)
                )
            ),
            "overall_fold_wins_vs_har_qlike": int(overall_wins),
            "spike_fold_wins_vs_har_qlike": int(spike_wins),
            "folds_compared": len(common),
            "pooled_overall_delta_vs_har_qlike": float(
                pooled["mean_qlike"] - har_pooled["mean_qlike"]
            ),
            "pooled_spike_delta_vs_har_qlike": float(
                pooled["spike_qlike"] - har_pooled["spike_qlike"]
            ),
            "passes_predeclared_screen": bool(
                overall_wins >= 3
                and pooled["mean_qlike"]
                < har_pooled["mean_qlike"] - min_delta
                and pooled["spike_qlike"]
                < har_pooled["spike_qlike"] - min_delta
            ),
        }

    market = pooled_metrics[
        pooled_metrics["model"] == "har_anchor_market"
    ].iloc[0]
    text = pooled_metrics[
        pooled_metrics["model"] == "har_anchor_market_text"
    ].iloc[0]
    market_folds = fold_metrics[
        fold_metrics["model"] == "har_anchor_market"
    ].set_index("fold")
    text_folds = fold_metrics[
        fold_metrics["model"] == "har_anchor_market_text"
    ].set_index("fold")
    common = sorted(set(market_folds.index) & set(text_folds.index))
    text_spike_wins = sum(
        text_folds.loc[fold, "spike_qlike"]
        < market_folds.loc[fold, "spike_qlike"] - min_delta
        for fold in common
    )
    text_contribution = {
        "spike_fold_wins_vs_market_correction": int(text_spike_wins),
        "folds_compared": len(common),
        "pooled_overall_delta_text_minus_market": float(
            text["mean_qlike"] - market["mean_qlike"]
        ),
        "pooled_spike_delta_text_minus_market": float(
            text["spike_qlike"] - market["spike_qlike"]
        ),
        "passes_predeclared_screen": bool(
            text_spike_wins >= 3
            and text["spike_qlike"] < market["spike_qlike"] - min_delta
            and text["mean_qlike"] <= market["mean_qlike"] + min_delta
        ),
    }
    return {
        "scope": "development-only rolling Fold 1-4, seed 11",
        "min_delta": min_delta,
        "epoch_zero_rule": (
            "The exact core-only HAR-QLIKE prediction is checkpoint epoch 0. "
            "A neural correction is selected only when validation QLIKE improves "
            "by at least min_delta."
        ),
        "har_anchor_candidates": candidates,
        "incremental_text_contribution": text_contribution,
        "statistical_claim": "None; this is a development diagnostic only.",
    }


def run_development_regime_anchor_diagnostic(
    config: MainPilotConfig,
    logger: logging.Logger,
    resume: bool,
    confirm_news_filter_reviewed: bool,
    scheduler_path: Path,
) -> dict[str, Any]:
    config.validate()
    if config.profile != "development-regime-anchor-diagnostic":
        raise ValueError("Wrong profile for regime-anchor diagnostic")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The 8 full regime-anchor runs are blocked "
            "on CPU; use --smoke locally and run this profile on a CUDA server."
        )
    device = torch.device("cuda")
    output_dir = config.output_path
    seed_everything(config.seed)
    schedule, scheduler_hash = _validate_main_scheduler(config, scheduler_path)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "run_manifest.json",
        {
            "profile": config.profile,
            "scope": "Fold 1-4 rolling development OOS only",
            "folds": [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS],
            "variants": list(REGIME_ANCHOR_VARIANTS),
            "planned_deep_runs": 8,
            "seed": config.seed,
            "objective": config.training_loss,
            "H_cos": schedule["H_cos"],
            "scheduler_source": str(scheduler_path.resolve()),
            "embedding_cache": config.embedding_cache_path,
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "anchor": (
                "GammaRegressor(alpha=0), fitted on each fold core only; "
                "epoch 0 is exact HAR-QLIKE and the neural head starts at zero"
            ),
            "selection_guard": (
                "retain correction only when validation QLIKE improves by 1e-5"
            ),
            "excluded": [
                "Fold_5_selection",
                "final_test",
                "COVID_fold",
                "MCS",
                "five_seeds",
                "loss_reweighting",
                "spike_oversampling",
            ],
            "created_utc": utc_now(),
        },
    )

    logger.info("REGIME DIAGNOSTIC STEP 1/4 | Load development market data")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info(
        "REGIME DIAGNOSTIC STEP 2/4 | Filter news and reuse embedding cache"
    )
    daily, _, review_path = _prepare_news(
        config,
        config.development_end,
        output_dir,
        logger,
        device,
        smoke=False,
        confirm_review=confirm_news_filter_reviewed,
    )
    if daily is None:
        raise RuntimeError(
            f"Manual development-only news-filter review required: {review_path}"
        )

    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    anchor_dir = output_dir / "anchors"
    for directory in (prediction_dir, metrics_dir, anchor_dir):
        directory.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    model_names = (
        *REGIME_ANCHOR_VARIANTS,
        "random_walk",
        "har_ols",
        "har_qlike",
    )
    all_predictions: dict[str, list[pd.DataFrame]] = {
        name: [] for name in model_names
    }
    run_number = 0
    for fold in SPIKE_DIAGNOSTIC_FOLDS:
        logger.info(
            "REGIME DIAGNOSTIC STEP 3/4 | Prepare %s and fit core-only HAR anchor",
            fold.name,
        )
        core, validation, test, metadata = _datasets(
            market,
            daily,
            fold,
            config,
            logger,
            include_test_diagnostic=True,
            output_dir=output_dir,
        )
        threshold = _spike_threshold(core)
        anchor = fit_har_qlike(market, core.target_dates)
        anchor_payload = anchor.metadata()
        anchor_hash = stable_hash(anchor_payload)
        run_preprocessor_hash = stable_hash(
            {
                "feature_preprocessor_hash": metadata["preprocessor_hash"],
                "har_qlike_anchor": anchor_payload,
            }
        )
        write_json(
            anchor_dir / f"{fold.name}_har_qlike.json",
            {
                **anchor_payload,
                "anchor_hash": anchor_hash,
                "fold": fold.name,
                "core_n": len(core),
                "validation_n": len(validation),
                "development_oos_n": len(test),
            },
        )
        for baseline in fit_baselines(
            market, core.target_dates, test.target_dates
        ):
            annotated = _annotate_predictions(
                baseline.predictions, fold.name, baseline.name, threshold
            )
            all_predictions[baseline.name].append(annotated)
            row = _diagnostic_metrics(
                baseline.predictions, threshold, baseline.name
            )
            row.update(
                {
                    "fold": fold.name,
                    "seed": None,
                    "correction_selected": None,
                }
            )
            fold_rows.append(row)

        for variant in REGIME_ANCHOR_VARIANTS:
            run_number += 1
            run_label = f"{fold.name}_{variant}_seed_11"
            checkpoint_dir = output_dir / "checkpoints" / run_label
            prediction_path = prediction_dir / f"{run_label}.csv"
            validation_path = prediction_dir / f"{run_label}_validation.csv"
            run_metrics_path = metrics_dir / f"{run_label}.json"
            logger.info(
                "REGIME DIAGNOSTIC RUN %02d/08 | fold=%s variant=%s",
                run_number,
                fold.name,
                variant,
            )
            completed = (
                validation_path.exists()
                and _completed_run(
                    checkpoint_dir,
                    prediction_path,
                    run_metrics_path,
                    expected_variant=variant,
                    expected_preprocessor_hash=run_preprocessor_hash,
                    expected_scheduler_hash=scheduler_hash,
                )
            )
            if resume and completed:
                logger.info(
                    "REGIME DIAGNOSTIC RESUME | completed run reused: %s",
                    run_label,
                )
                predictions = pd.read_csv(prediction_path)
                row = json.loads(run_metrics_path.read_text(encoding="utf-8"))
            else:
                seed_everything(config.seed)
                built = build_model(
                    config,
                    metadata["target_mean_logrv"],
                    metadata["target_scale_logrv"],
                    metadata["unconditional_mean_rv"],
                    variant=variant,
                    har_anchor_intercept=anchor.intercept,
                    har_anchor_coefficients=anchor.coefficients.tolist(),
                )
                expected_rv = _initial_anchor_rv(
                    core,
                    metadata["target_mean_logrv"],
                    metadata["target_scale_logrv"],
                    anchor.intercept,
                    anchor.coefficients,
                )
                _initialization_assertion(
                    built.model, core, expected_rv, device
                )
                result = train_model(
                    built.model,
                    core,
                    validation,
                    config,
                    horizon_epochs=int(schedule["H_cos"]),
                    checkpoint_dir=checkpoint_dir,
                    preprocessor_hash=run_preprocessor_hash,
                    scheduler_hash=scheduler_hash,
                    fold_name=run_label,
                    logger=logger,
                    device=device,
                    resume=resume,
                    include_epoch_zero_checkpoint=True,
                )
                _require_numerically_successful(result, run_label)
                validation_predictions = predict_dataset(
                    built.model,
                    validation,
                    config,
                    device,
                    scheduler_path,
                    scheduler_hash,
                )
                validation_predictions.to_csv(validation_path, index=False)
                predictions = predict_dataset(
                    built.model,
                    test,
                    config,
                    device,
                    scheduler_path,
                    scheduler_hash,
                )
                predictions.to_csv(prediction_path, index=False)
                epoch_zero_validation = float(
                    next(
                        item["validation_qlike"]
                        for item in result.history
                        if int(item["epoch"]) == 0
                    )
                )
                correction_selected = bool(result.best_epoch > 0)
                row = _diagnostic_metrics(predictions, threshold, variant)
                row.update(
                    {
                        "fold": fold.name,
                        "seed": config.seed,
                        "best_epoch": result.best_epoch,
                        "epochs_run": result.epochs_run,
                        "early_stopped": result.early_stopped,
                        "numerical_failure": result.numerical_failure,
                        "parameter_count": built.parameter_count,
                        "preprocessor_hash": metadata["preprocessor_hash"],
                        "run_preprocessor_hash": run_preprocessor_hash,
                        "har_anchor_hash": anchor_hash,
                        "epoch_zero_validation_qlike": epoch_zero_validation,
                        "best_validation_qlike": result.best_validation_qlike,
                        "validation_improvement": (
                            epoch_zero_validation
                            - result.best_validation_qlike
                        ),
                        "correction_selected": correction_selected,
                        "validation_metrics": _diagnostic_metrics(
                            validation_predictions, threshold, variant
                        ),
                        **_correction_metrics(predictions),
                    }
                )
                write_json(run_metrics_path, row)
            annotated = _annotate_predictions(
                predictions, fold.name, variant, threshold
            )
            all_predictions[variant].append(annotated)
            fold_rows.append(row)

    logger.info(
        "REGIME DIAGNOSTIC STEP 4/4 | Pool Fold 1-4 and apply HAR/text screens"
    )
    pooled_rows = []
    for model_name, parts in all_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        pooled.to_csv(
            prediction_dir / f"pooled_{model_name}.csv", index=False
        )
        pooled_rows.append(_pooled_metrics(pooled, model_name))
        pooled.nlargest(10, "qlike").to_csv(
            metrics_dir / f"worst_days_{model_name}.csv", index=False
        )
    fold_frame = pd.DataFrame(fold_rows)
    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    selection = _anchor_selection(fold_frame, pooled_frame)
    write_json(metrics_dir / "pooled_metrics.json", pooled_rows)
    write_json(metrics_dir / "selection_diagnostic.json", selection)
    report = {
        "status": "completed",
        "planned_deep_runs": 8,
        "completed_deep_runs": 8,
        "folds": [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS],
        "variants": list(REGIME_ANCHOR_VARIANTS),
        "pooled_metrics": pooled_rows,
        "selection": selection,
        "attention_backend": attention_backend_metadata(device),
        "completed_utc": utc_now(),
    }
    report_path = metrics_dir / "diagnostic_report.json"
    write_json(report_path, report)
    logger.info(
        "REGIME ANCHOR DIAGNOSTIC COMPLETED | report=%s", report_path
    )
    return report


def run_regime_anchor_smoke(
    config: MainPilotConfig,
    logger: logging.Logger,
    resume: bool,
) -> dict[str, Any]:
    smoke_dir = config.output_path / "smoke"
    smoke_config = replace(
        config,
        smoke=True,
        output_dir=str(smoke_dir),
        embedding_cache_path=None,
        physical_batch_size=1,
        num_workers=0,
    )
    device = torch.device("cpu")
    seed_everything(smoke_config.seed)
    write_json(smoke_dir / "config.json", smoke_config.to_dict())
    market = load_market_data(
        Path(smoke_config.market_path),
        smoke_config,
        logger,
        smoke_config.smoke_start,
        smoke_config.smoke_end,
    )
    write_market_audit(market, smoke_dir)
    daily, _, _ = _prepare_news(
        smoke_config,
        smoke_config.smoke_end,
        smoke_dir,
        logger,
        device,
        smoke=True,
        confirm_review=True,
    )
    assert daily is not None
    fold = Fold(
        "smoke_fold",
        "2018-03-02",
        "2018-04-15",
        "2018-04-16",
        "2018-04-30",
        "2018-05-01",
        "2018-05-31",
    )
    core, validation, test, metadata = _datasets(
        market,
        daily,
        fold,
        smoke_config,
        logger,
        include_test_diagnostic=True,
        output_dir=smoke_dir,
    )
    anchor = fit_har_qlike(market, core.target_dates)
    run_preprocessor_hash = stable_hash(
        {
            "feature_preprocessor_hash": metadata["preprocessor_hash"],
            "har_qlike_anchor": anchor.metadata(),
        }
    )
    schedule = {
        "H_cos": smoke_config.provisional_horizon_epochs,
        "config_hash": stable_hash(smoke_config.to_dict()),
        "locked_from": "regime_anchor_diagnostic_smoke_only",
        "created_utc": utc_now(),
    }
    scheduler_path = smoke_dir / "smoke_scheduler_horizon.json"
    write_json(scheduler_path, schedule)
    scheduler_hash = stable_hash(schedule)
    rows = []
    for variant in REGIME_ANCHOR_VARIANTS:
        seed_everything(smoke_config.seed)
        built = build_model(
            smoke_config,
            metadata["target_mean_logrv"],
            metadata["target_scale_logrv"],
            metadata["unconditional_mean_rv"],
            variant=variant,
            har_anchor_intercept=anchor.intercept,
            har_anchor_coefficients=anchor.coefficients.tolist(),
        )
        expected_rv = _initial_anchor_rv(
            core,
            metadata["target_mean_logrv"],
            metadata["target_scale_logrv"],
            anchor.intercept,
            anchor.coefficients,
        )
        _initialization_assertion(built.model, core, expected_rv, device)
        result = train_model(
            built.model,
            core,
            validation,
            smoke_config,
            horizon_epochs=smoke_config.provisional_horizon_epochs,
            checkpoint_dir=smoke_dir / "checkpoints" / variant,
            preprocessor_hash=run_preprocessor_hash,
            scheduler_hash=scheduler_hash,
            fold_name=f"smoke_{variant}",
            logger=logger,
            device=device,
            resume=resume,
            max_epochs_override=smoke_config.smoke_epochs,
            max_train_batches=smoke_config.smoke_max_train_batches,
            max_eval_batches=smoke_config.smoke_max_eval_batches,
            include_epoch_zero_checkpoint=True,
        )
        _require_numerically_successful(result, f"Smoke {variant}")
        predictions = predict_dataset(
            built.model,
            test,
            smoke_config,
            device,
            scheduler_path,
            scheduler_hash,
            max_batches=smoke_config.smoke_max_eval_batches,
        )
        if not {"har_anchor_log_rv", "delta_log_rv"}.issubset(
            predictions.columns
        ):
            raise AssertionError("HAR-anchor prediction diagnostics are absent")
        row = _diagnostic_metrics(
            predictions, _spike_threshold(core), variant
        )
        row.update(
            {
                "best_epoch": result.best_epoch,
                "epoch_zero_validation_qlike": result.history[0][
                    "validation_qlike"
                ],
                "parameter_count": built.parameter_count,
                **_correction_metrics(predictions),
            }
        )
        rows.append(row)
    report = {
        "status": "passed",
        "backend": "CPU",
        "variants": list(REGIME_ANCHOR_VARIANTS),
        "runs": rows,
        "epoch_zero_fallback_verified": True,
        "embedding_backend": "deterministic smoke test double",
    }
    write_json(smoke_dir / "regime_anchor_smoke_report.json", report)
    return report
