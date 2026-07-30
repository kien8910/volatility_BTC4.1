from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines import fit_baselines
from .config import (
    Fold,
    MainPilotConfig,
    SPIKE_DIAGNOSTIC_FOLDS,
    SPIKE_DIAGNOSTIC_VARIANTS,
)
from .data import load_market_data, write_market_audit
from .metrics import log_forecast_diagnostics, prediction_metrics
from .model import attention_backend_metadata, build_model
from .pipeline import (
    _datasets,
    _initialization_assertion,
    _locked_config_hash,
    _prepare_news,
    _require_numerically_successful,
)
from .training import predict_dataset, train_model
from .utils import (
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


def _spike_threshold(dataset: Any) -> float:
    return float(
        np.quantile(
            [
                dataset.market.rv[dataset.market.date_to_index[date]]
                for date in dataset.target_dates
            ],
            0.90,
        )
    )


def _annotate_predictions(
    frame: pd.DataFrame,
    fold_name: str,
    model_name: str,
    threshold: float,
) -> pd.DataFrame:
    output = frame.copy()
    ratio = (
        output["true_rv"].to_numpy(dtype=np.float64)
        / output["predicted_rv"].to_numpy(dtype=np.float64)
    )
    output["fold"] = fold_name
    output["model"] = model_name
    output["spike_threshold"] = threshold
    output["is_spike"] = output["true_rv"] > threshold
    output["predicted_spike"] = output["predicted_rv"] > threshold
    output["qlike"] = ratio - np.log(ratio) - 1.0
    return output


def _diagnostic_metrics(
    frame: pd.DataFrame,
    threshold: float,
    model_name: str,
) -> dict[str, Any]:
    metrics = prediction_metrics(frame, threshold, model_name)
    spike = frame["true_rv"].to_numpy(dtype=np.float64) > threshold
    predicted_spike = (
        frame["predicted_rv"].to_numpy(dtype=np.float64) > threshold
    )
    predicted_log = frame["predicted_log_rv"].to_numpy(dtype=np.float64)
    true_log = frame["true_log_rv"].to_numpy(dtype=np.float64)
    metrics.update(
        {
            "predicted_spike_n": int(predicted_spike.sum()),
            "spike_capture_n": int((spike & predicted_spike).sum()),
            "spike_capture_rate": (
                float((spike & predicted_spike).sum() / spike.sum())
                if spike.any()
                else None
            ),
            "predicted_logrv_std": float(np.std(predicted_log)),
            "true_logrv_std": float(np.std(true_log)),
            "predicted_to_true_logrv_std_ratio": float(
                np.std(predicted_log) / max(np.std(true_log), 1e-12)
            ),
            "mean_spike_predicted_to_true_rv_ratio": (
                float(
                    np.mean(
                        frame.loc[spike, "predicted_rv"].to_numpy()
                        / frame.loc[spike, "true_rv"].to_numpy()
                    )
                )
                if spike.any()
                else None
            ),
        }
    )
    return metrics


def _pooled_metrics(frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
    finite = (
        np.isfinite(frame["true_rv"])
        & np.isfinite(frame["predicted_rv"])
        & np.isfinite(frame["true_log_rv"])
        & np.isfinite(frame["predicted_log_rv"])
        & (frame["true_rv"] > 0)
        & (frame["predicted_rv"] > 0)
    )
    clean = frame.loc[finite].copy()
    spike = clean["is_spike"].to_numpy(dtype=bool)
    predicted_spike = clean["predicted_spike"].to_numpy(dtype=bool)
    qlike = clean["qlike"].to_numpy(dtype=np.float64)
    error = (
        clean["true_log_rv"].to_numpy(dtype=np.float64)
        - clean["predicted_log_rv"].to_numpy(dtype=np.float64)
    )
    true_log = clean["true_log_rv"].to_numpy(dtype=np.float64)
    predicted_log = clean["predicted_log_rv"].to_numpy(dtype=np.float64)
    denominator = float(np.sum((true_log - np.mean(true_log)) ** 2))
    r2 = (
        1.0 - float(np.sum(error**2)) / denominator
        if denominator > 0
        else float("nan")
    )
    output = {
        "model": model_name,
        "folds": int(clean["fold"].nunique()),
        "n_predictions": int(len(frame)),
        "n_nan_inf_or_nonpositive": int((~finite).sum()),
        "mean_qlike": float(np.mean(qlike)),
        "sum_qlike": float(np.sum(qlike)),
        "normal_qlike": float(np.mean(qlike[~spike])),
        "spike_qlike": float(np.mean(qlike[spike])),
        "normal_n": int((~spike).sum()),
        "spike_n": int(spike.sum()),
        "predicted_spike_n": int(predicted_spike.sum()),
        "spike_capture_n": int((spike & predicted_spike).sum()),
        "spike_capture_rate": float(
            (spike & predicted_spike).sum() / max(spike.sum(), 1)
        ),
        "predicted_logrv_std": float(np.std(predicted_log)),
        "true_logrv_std": float(np.std(true_log)),
        "predicted_to_true_logrv_std_ratio": float(
            np.std(predicted_log) / max(np.std(true_log), 1e-12)
        ),
        "mean_spike_predicted_to_true_rv_ratio": float(
            np.mean(
                clean.loc[spike, "predicted_rv"].to_numpy()
                / clean.loc[spike, "true_rv"].to_numpy()
            )
        ),
        "r2_logrv": r2,
        "rmse_logrv": float(np.sqrt(np.mean(error**2))),
        "mae_logrv": float(np.mean(np.abs(error))),
    }
    output.update(log_forecast_diagnostics(clean))
    fold_direction_hits: list[np.ndarray] = []
    for _, group in clean.groupby("fold", sort=False):
        ordered = group.sort_values("target_date")
        group_true = ordered["true_log_rv"].to_numpy(dtype=np.float64)
        group_predicted = ordered["predicted_log_rv"].to_numpy(
            dtype=np.float64
        )
        if len(group_true) > 1:
            fold_direction_hits.append(
                np.sign(np.diff(group_true))
                == np.sign(group_predicted[1:] - group_true[:-1])
            )
    output["directional_accuracy_logrv"] = (
        float(np.mean(np.concatenate(fold_direction_hits)))
        if fold_direction_hits
        else None
    )
    return output


def _validate_main_scheduler(
    config: MainPilotConfig,
    scheduler_path: Path,
) -> tuple[dict[str, Any], str]:
    if not scheduler_path.exists():
        raise RuntimeError(
            f"Locked main-pilot scheduler not found: {scheduler_path}"
        )
    schedule = json.loads(scheduler_path.read_text(encoding="utf-8"))
    expected_hash = _locked_config_hash(replace(config, profile="main-pilot"))
    if schedule.get("config_hash") != expected_hash:
        raise RuntimeError(
            "The development diagnostic must reuse the successful main-pilot "
            "scheduler, but its locked config hash differs"
        )
    if schedule.get("pilot_completed_without_numerical_failure") is not True:
        raise RuntimeError(
            "The referenced scheduler does not certify a numerically successful pilot"
        )
    return schedule, stable_hash(schedule)


def _completed_run(
    checkpoint_dir: Path,
    prediction_path: Path,
    metrics_path: Path,
    expected_variant: str,
    expected_preprocessor_hash: str,
    expected_scheduler_hash: str,
) -> bool:
    result_path = checkpoint_dir / "training_result.json"
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    if not (
        result_path.exists()
        and metadata_path.exists()
        and prediction_path.exists()
        and metrics_path.exists()
    ):
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        not bool(result.get("numerical_failure", True))
        and metadata.get("model_variant") == expected_variant
        and metadata.get("preprocessor_hash") == expected_preprocessor_hash
        and metadata.get("scheduler_hash") == expected_scheduler_hash
    )


def _selection_diagnostic(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
) -> dict[str, Any]:
    min_delta = 1e-5
    main_fold = fold_metrics[fold_metrics["model"] == "main"].set_index("fold")
    main_pooled = pooled_metrics[pooled_metrics["model"] == "main"].iloc[0]
    candidates: dict[str, Any] = {}
    for variant in ("market_only", "hybrid_har"):
        candidate_fold = fold_metrics[
            fold_metrics["model"] == variant
        ].set_index("fold")
        common = sorted(set(main_fold.index) & set(candidate_fold.index))
        spike_wins = sum(
            candidate_fold.loc[fold, "spike_qlike"]
            < main_fold.loc[fold, "spike_qlike"] - min_delta
            for fold in common
        )
        candidate_pooled = pooled_metrics[
            pooled_metrics["model"] == variant
        ].iloc[0]
        candidates[variant] = {
            "spike_fold_wins_vs_main": int(spike_wins),
            "folds_compared": len(common),
            "pooled_spike_improved": bool(
                candidate_pooled["spike_qlike"]
                < main_pooled["spike_qlike"] - min_delta
            ),
            "pooled_overall_not_worse": bool(
                candidate_pooled["mean_qlike"]
                <= main_pooled["mean_qlike"] + min_delta
            ),
            "passes_predeclared_screen": bool(
                spike_wins >= 3
                and candidate_pooled["spike_qlike"]
                < main_pooled["spike_qlike"] - min_delta
                and candidate_pooled["mean_qlike"]
                <= main_pooled["mean_qlike"] + min_delta
            ),
        }
    return {
        "scope": "development-only Fold 1-4, seed 11 screening diagnostic",
        "min_delta": min_delta,
        "rule": (
            "Pass only if spike QLIKE beats main in at least 3/4 folds, "
            "pooled spike QLIKE improves, and pooled overall QLIKE is not worse."
        ),
        "candidates": candidates,
        "statistical_claim": "None; this is not final-test inference.",
    }


def run_development_spike_diagnostic(
    config: MainPilotConfig,
    logger: logging.Logger,
    resume: bool,
    confirm_news_filter_reviewed: bool,
    scheduler_path: Path,
) -> dict[str, Any]:
    config.validate()
    if config.profile != "development-spike-diagnostic":
        raise ValueError("Wrong profile for development spike diagnostic")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The 12 full diagnostic runs are blocked on "
            "CPU; use --smoke locally and run this profile on a CUDA server."
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
            "scope": "Fold 1-4 development OOS only",
            "folds": [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS],
            "variants": list(SPIKE_DIAGNOSTIC_VARIANTS),
            "planned_deep_runs": 12,
            "seed": config.seed,
            "objective": config.training_loss,
            "H_cos": schedule["H_cos"],
            "scheduler_source": str(scheduler_path.resolve()),
            "embedding_cache": config.embedding_cache_path,
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "hybrid_har_scaling": (
                "(logRV_d, logRV_w, logRV_m - core target mean) / "
                "core target standard deviation"
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
    logger.info("DIAGNOSTIC STEP 1/4 | Load development market data")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("DIAGNOSTIC STEP 2/4 | Filter news and reuse embedding cache")
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
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    all_predictions: dict[str, list[pd.DataFrame]] = {
        name: [] for name in (*SPIKE_DIAGNOSTIC_VARIANTS, "random_walk", "har_ols", "har_qlike")
    }
    run_number = 0
    for fold in SPIKE_DIAGNOSTIC_FOLDS:
        logger.info("DIAGNOSTIC STEP 3/4 | Prepare %s features", fold.name)
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
        baselines = fit_baselines(market, core.target_dates, test.target_dates)
        for baseline in baselines:
            annotated = _annotate_predictions(
                baseline.predictions, fold.name, baseline.name, threshold
            )
            all_predictions[baseline.name].append(annotated)
            row = _diagnostic_metrics(
                baseline.predictions, threshold, baseline.name
            )
            row["fold"] = fold.name
            row["seed"] = None
            fold_rows.append(row)

        for variant in SPIKE_DIAGNOSTIC_VARIANTS:
            run_number += 1
            run_label = f"{fold.name}_{variant}_seed_11"
            checkpoint_dir = output_dir / "checkpoints" / run_label
            prediction_path = prediction_dir / f"{run_label}.csv"
            run_metrics_path = metrics_dir / f"{run_label}.json"
            logger.info(
                "DIAGNOSTIC RUN %02d/12 | fold=%s variant=%s",
                run_number,
                fold.name,
                variant,
            )
            if resume and _completed_run(
                checkpoint_dir,
                prediction_path,
                run_metrics_path,
                expected_variant=variant,
                expected_preprocessor_hash=metadata["preprocessor_hash"],
                expected_scheduler_hash=scheduler_hash,
            ):
                logger.info("DIAGNOSTIC RESUME | completed run reused: %s", run_label)
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
                )
                _initialization_assertion(
                    built.model,
                    core,
                    metadata["unconditional_mean_rv"],
                    device,
                )
                result = train_model(
                    built.model,
                    core,
                    validation,
                    config,
                    horizon_epochs=int(schedule["H_cos"]),
                    checkpoint_dir=checkpoint_dir,
                    preprocessor_hash=metadata["preprocessor_hash"],
                    scheduler_hash=scheduler_hash,
                    fold_name=run_label,
                    logger=logger,
                    device=device,
                    resume=resume,
                )
                _require_numerically_successful(result, run_label)
                predictions = predict_dataset(
                    built.model,
                    test,
                    config,
                    device,
                    scheduler_path,
                    scheduler_hash,
                )
                predictions.to_csv(prediction_path, index=False)
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
                    }
                )
                write_json(run_metrics_path, row)
            annotated = _annotate_predictions(
                predictions, fold.name, variant, threshold
            )
            all_predictions[variant].append(annotated)
            fold_rows.append(row)

    logger.info("DIAGNOSTIC STEP 4/4 | Pool Fold 1-4 and apply screen")
    pooled_rows = []
    for model_name, parts in all_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        pooled.to_csv(prediction_dir / f"pooled_{model_name}.csv", index=False)
        pooled_rows.append(_pooled_metrics(pooled, model_name))
        pooled.nlargest(10, "qlike").to_csv(
            metrics_dir / f"worst_days_{model_name}.csv", index=False
        )
    fold_frame = pd.DataFrame(fold_rows)
    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    selection = _selection_diagnostic(fold_frame, pooled_frame)
    write_json(metrics_dir / "pooled_metrics.json", pooled_rows)
    write_json(metrics_dir / "selection_diagnostic.json", selection)
    report = {
        "status": "completed",
        "planned_deep_runs": 12,
        "completed_deep_runs": 12,
        "folds": [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS],
        "variants": list(SPIKE_DIAGNOSTIC_VARIANTS),
        "pooled_metrics": pooled_rows,
        "selection": selection,
        "attention_backend": attention_backend_metadata(device),
        "completed_utc": utc_now(),
    }
    write_json(metrics_dir / "diagnostic_report.json", report)
    return report


def run_spike_diagnostic_smoke(
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
    schedule = {
        "H_cos": smoke_config.provisional_horizon_epochs,
        "config_hash": stable_hash(smoke_config.to_dict()),
        "locked_from": "three_variant_spike_diagnostic_smoke_only",
        "created_utc": utc_now(),
    }
    scheduler_path = smoke_dir / "smoke_scheduler_horizon.json"
    write_json(scheduler_path, schedule)
    scheduler_hash = stable_hash(schedule)
    rows = []
    for variant in SPIKE_DIAGNOSTIC_VARIANTS:
        seed_everything(smoke_config.seed)
        built = build_model(
            smoke_config,
            metadata["target_mean_logrv"],
            metadata["target_scale_logrv"],
            metadata["unconditional_mean_rv"],
            variant=variant,
        )
        _initialization_assertion(
            built.model,
            core,
            metadata["unconditional_mean_rv"],
            device,
        )
        result = train_model(
            built.model,
            core,
            validation,
            smoke_config,
            horizon_epochs=smoke_config.provisional_horizon_epochs,
            checkpoint_dir=smoke_dir / "checkpoints" / variant,
            preprocessor_hash=metadata["preprocessor_hash"],
            scheduler_hash=scheduler_hash,
            fold_name=f"smoke_{variant}",
            logger=logger,
            device=device,
            resume=resume,
            max_epochs_override=smoke_config.smoke_epochs,
            max_train_batches=smoke_config.smoke_max_train_batches,
            max_eval_batches=smoke_config.smoke_max_eval_batches,
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
        row = _diagnostic_metrics(
            predictions, _spike_threshold(core), variant
        )
        row["parameter_count"] = built.parameter_count
        rows.append(row)
    report = {
        "status": "passed",
        "backend": "CPU",
        "variants": list(SPIKE_DIAGNOSTIC_VARIANTS),
        "runs": rows,
        "embedding_backend": "deterministic smoke test double",
    }
    write_json(smoke_dir / "spike_diagnostic_smoke_report.json", report)
    return report
