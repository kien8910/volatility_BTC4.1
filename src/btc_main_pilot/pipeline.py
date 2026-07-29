from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines import fit_baselines
from .config import Fold, MainPilotConfig
from .data import (
    MarketData,
    load_market_data,
    sample_dates_for_block,
    write_market_audit,
)
from .dataset import RVWindowDataset
from .metrics import prediction_metrics
from .model import attention_backend_metadata, build_model
from .news import (
    DeterministicSmokeEncoder,
    EmbeddingCache,
    OfflineBgeFinbertEncoder,
    aggregate_daily_news,
    embed_articles,
    load_filtered_articles,
    write_news_audits,
)
from .preprocess import fit_market_scaler, fit_transform_news_for_fold
from .training import (
    predict_dataset,
    scheduler_payload,
    train_model,
)
from .utils import (
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


def _locked_config_hash(config: MainPilotConfig) -> str:
    payload = config.to_dict()
    for runtime_key in [
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
    ]:
        payload.pop(runtime_key, None)
    return stable_hash(payload)


def _write_filter_review(
    audit: dict[str, Any], output_dir: Path, confirmed: bool
) -> Path:
    review_path = output_dir / "audit" / "manual_news_filter_review.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit["manual_review_examples"]).to_csv(review_path, index=False)
    write_json(
        output_dir / "audit" / "manual_news_filter_review_status.json",
        {
            "development_only": True,
            "sample_path": str(review_path),
            "operator_confirmed": confirmed,
            "threshold": 2,
            "rule_locked_before_embedding": confirmed,
            "created_utc": utc_now(),
        },
    )
    return review_path


def _prepare_news(
    config: MainPilotConfig,
    end: str,
    output_dir: Path,
    logger: logging.Logger,
    device: torch.device,
    smoke: bool,
    confirm_review: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any], Path]:
    articles, filter_audit = load_filtered_articles(
        Path(config.news_path),
        config.research_start,
        end,
        logger,
        smoke_early_stop=smoke,
    )
    review_path = _write_filter_review(filter_audit, output_dir, confirm_review)
    if not smoke and not confirm_review:
        logger.warning(
            "NEWS REVIEW REQUIRED | inspect %s then rerun with "
            "--confirm-news-filter-reviewed",
            review_path,
        )
        return None, filter_audit, review_path
    cache_model_semantic = (
        f"{config.semantic_model}::deterministic-smoke-v1"
        if smoke
        else config.semantic_model
    )
    cache_model_sentiment = (
        f"{config.sentiment_model}::deterministic-smoke-v1"
        if smoke
        else config.sentiment_model
    )
    cache_path = (
        output_dir / "cache" / "article_embeddings.sqlite"
        if smoke or config.embedding_cache_path is None
        else Path(config.embedding_cache_path)
    )
    cache = EmbeddingCache(
        cache_path,
        cache_model_semantic,
        cache_model_sentiment,
    )
    try:
        encoder = (
            DeterministicSmokeEncoder(config.embedding_dim)
            if smoke
            else OfflineBgeFinbertEncoder(config, device)
        )
        semantics, sentiments, cache_stats = embed_articles(
            articles,
            cache,
            encoder,
            config.embedding_batch_size,
            logger,
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
    filter_audit["embedding_cache"] = cache_stats
    filter_audit["embedding_backend"] = (
        "deterministic_smoke_test_double"
        if smoke
        else "offline_frozen_BGE_CLS_L2_and_FinBERT_probabilities"
    )
    write_news_audits(daily, articles, output_dir, filter_audit)
    return daily, filter_audit, review_path


def _block_dates(
    market: MarketData,
    fold: Fold,
    output_dir: Path,
    logger: logging.Logger,
    include_test: bool,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    blocks = {}
    dates = {}
    specifications = [
        ("core", fold.core_start, fold.core_end),
        ("validation", fold.validation_start, fold.validation_end),
    ]
    if include_test:
        specifications.append(("test", fold.test_start, fold.test_end))
    for name, start, end in specifications:
        dates[name], blocks[name] = sample_dates_for_block(market, start, end)
        logger.info(
            "SAMPLE AUDIT | fold=%s block=%s candidates=%d final=%d",
            fold.name,
            name,
            blocks[name]["candidate_target_days"],
            blocks[name]["final_sample_count"],
        )
    write_json(output_dir / "audit" / f"{fold.name}_sample_counts.json", blocks)
    return dates["core"], dates["validation"], dates.get("test", [])


def _datasets(
    market: MarketData,
    daily: pd.DataFrame,
    fold: Fold,
    config: MainPilotConfig,
    logger: logging.Logger,
    include_test_diagnostic: bool,
    output_dir: Path,
) -> tuple[
    RVWindowDataset,
    RVWindowDataset,
    RVWindowDataset,
    dict[str, Any],
]:
    core_dates, validation_dates, test_dates = _block_dates(
        market, fold, output_dir, logger, include_test=include_test_diagnostic
    )
    if not config.smoke and len(validation_dates) < 60:
        raise RuntimeError(
            f"{fold.name} has only {len(validation_dates)} valid validation targets (<60)"
        )
    news_end = fold.test_end if include_test_diagnostic else fold.validation_end
    news_daily = daily.loc[: pd.Timestamp(news_end, tz="UTC")]
    fold_news = fit_transform_news_for_fold(
        news_daily,
        fold.core_start,
        fold.core_end,
        fold.validation_start,
        fold.validation_end,
        fold.test_start if include_test_diagnostic else None,
        fold.test_end if include_test_diagnostic else None,
        config,
        logger,
    )
    market_scaler = fit_market_scaler(market, core_dates, config)
    core_log = np.asarray(
        [market.log_rv[market.date_to_index[date]] for date in core_dates],
        dtype=np.float64,
    )
    target_mean = float(np.mean(core_log))
    target_scale = max(float(np.std(core_log)), config.scaler_epsilon)
    unconditional_mean_rv = float(
        np.mean([market.rv[market.date_to_index[date]] for date in core_dates])
    )
    common = (
        market,
        fold_news,
        market_scaler,
        target_mean,
        target_scale,
        config,
    )
    core_dataset = RVWindowDataset(
        common[0], common[1], core_dates, common[2], common[3], common[4], common[5]
    )
    validation_dataset = RVWindowDataset(
        common[0],
        common[1],
        validation_dates,
        common[2],
        common[3],
        common[4],
        common[5],
    )
    test_dataset = RVWindowDataset(
        common[0], common[1], test_dates, common[2], common[3], common[4], common[5]
    )
    metadata = {
        **fold_news.metadata,
        "target_mean_logrv": target_mean,
        "target_scale_logrv": target_scale,
        "unconditional_mean_rv": unconditional_mean_rv,
        "market_scaler": {
            "mean": market_scaler.mean.tolist(),
            "scale": market_scaler.scale.tolist(),
            "fine_patch_logrv_mean": market_scaler.fine_patch_logrv_mean,
            "fine_patch_logrv_scale": market_scaler.fine_patch_logrv_scale,
            "coarse_patch_logrv_mean": market_scaler.coarse_patch_logrv_mean,
            "coarse_patch_logrv_scale": market_scaler.coarse_patch_logrv_scale,
        },
    }
    write_json(output_dir / "features" / f"{fold.name}_metadata.json", metadata)
    return core_dataset, validation_dataset, test_dataset, metadata


def _initialization_assertion(
    model: torch.nn.Module,
    dataset: RVWindowDataset,
    expected_rv: float,
    device: torch.device,
) -> None:
    model.to(device)
    parameter_devices = {parameter.device for parameter in model.parameters()}
    buffer_devices = {buffer.device for buffer in model.buffers()}

    def on_requested_device(actual: torch.device) -> bool:
        return actual.type == device.type and (
            device.index is None or actual.index == device.index
        )

    if not all(
        on_requested_device(actual)
        for actual in parameter_devices | buffer_devices
    ):
        raise RuntimeError(
            "Initialization check could not place every model parameter and buffer "
            f"on {device}: parameters={parameter_devices}, buffers={buffer_devices}"
        )
    model.eval()
    raw = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device)
        if isinstance(value, torch.Tensor)
        else [value]
        for key, value in raw.items()
    }
    with torch.no_grad():
        output = model(batch)
    actual = float(output["predicted_rv"].item())
    if not np.isclose(actual, expected_rv, rtol=1e-6, atol=1e-14):
        raise AssertionError(
            f"Forecast initialization {actual} != core mean RV {expected_rv}"
        )


def _require_numerically_successful(
    training_result: Any,
    stage: str,
) -> None:
    if training_result.numerical_failure:
        raise RuntimeError(
            f"{stage} failed numerically: {training_result.failure_reason}. "
            "Available diagnostic artifacts/checkpoints were preserved, but no "
            "scheduler lock, OOS prediction, baseline comparison, or metrics "
            "may be produced."
        )


def _report(
    deep_predictions: pd.DataFrame,
    baseline_results: list[Any],
    core_dataset: RVWindowDataset,
    training_result: Any,
    feature_metadata: dict[str, Any],
    parameter_count: int,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    _require_numerically_successful(training_result, "Fold 5 training")
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_frames = {"patchtst_bge_finbert_cross_attention": deep_predictions}
    for baseline in baseline_results:
        all_frames[baseline.name] = baseline.predictions
    common_dates = None
    for frame in all_frames.values():
        dates = set(frame["target_date"])
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates or [])
    if not common_dates:
        raise RuntimeError("No common OOS evaluation dates")
    spike_threshold = float(
        np.quantile(
            [
                core_dataset.market.rv[core_dataset.market.date_to_index[date]]
                for date in core_dataset.target_dates
            ],
            0.90,
        )
    )
    metric_rows = []
    for name, frame in all_frames.items():
        common = frame[frame["target_date"].isin(common_dates)].copy()
        common.sort_values("target_date", inplace=True)
        common.to_csv(prediction_dir / f"{name}.csv", index=False)
        common.to_json(
            prediction_dir / f"{name}.json",
            orient="records",
            indent=2,
        )
        metric_rows.append(prediction_metrics(common, spike_threshold, name))
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(metrics_dir / "metrics.csv", index=False)
    convergence = {
        "best_epoch": training_result.best_epoch,
        "epochs_run": training_result.epochs_run,
        "early_stopped": training_result.early_stopped,
        "best_validation_qlike": training_result.best_validation_qlike,
        "initial_train_qlike": training_result.history[0]["train_qlike"],
        "final_train_qlike": training_result.history[-1]["train_qlike"],
        "initial_validation_qlike": training_result.history[0][
            "validation_qlike"
        ],
        "final_validation_qlike": training_result.history[-1][
            "validation_qlike"
        ],
        "validation_improved": training_result.best_validation_qlike
        < training_result.history[0]["validation_qlike"] - 1e-5,
        "training_seconds": training_result.training_seconds,
        "numerical_failure": training_result.numerical_failure,
        "failure_reason": training_result.failure_reason,
    }
    report = {
        "scope": "main-pilot diagnostic only",
        "statistical_claim": (
            "Convergence and competitiveness against HAR only; not statistical inference."
        ),
        "fold": "fold_5",
        "seed": 11,
        "common_oos_dates": len(common_dates),
        "spike_threshold_core_train_p90_rv": spike_threshold,
        "metrics": metric_rows,
        "convergence": convergence,
        "n_nan_inf_total": int(
            sum(row["n_nan_inf_or_nonpositive"] for row in metric_rows)
        ),
        "parameter_count": parameter_count,
        "pca_diagnostic": {
            key: feature_metadata[key]
            for key in [
                "CVR_core",
                "CVR_validation",
                "CVR_test",
                "relative_drop",
            ]
        },
        "excluded": [
            "final_test",
            "ablation",
            "COVID_fold",
            "MCS",
            "five_seed_protocol",
        ],
    }
    write_json(metrics_dir / "metrics.json", report)
    write_json(metrics_dir / "convergence.json", convergence)
    logger.info("REPORT | common_oos_dates=%d", len(common_dates))
    for row in metric_rows:
        logger.info(
            "METRIC | %-38s QLIKE=%.8f spike=%s normal=%s R2=%.5f RMSE=%.5f NaN/Inf=%d",
            row["model"],
            row["mean_qlike"],
            row["spike_qlike"],
            row["normal_qlike"],
            row["r2_logrv"],
            row["rmse_logrv"],
            row["n_nan_inf_or_nonpositive"],
        )
    return report


def run_review_only(
    config: MainPilotConfig, logger: logging.Logger
) -> Path:
    output_dir = config.output_path
    articles, audit = load_filtered_articles(
        Path(config.news_path),
        config.research_start,
        config.development_end,
        logger,
    )
    del articles
    return _write_filter_review(audit, output_dir, False)


def run_smoke(
    config: MainPilotConfig,
    logger: logging.Logger,
    resume: bool,
) -> dict[str, Any]:
    smoke_dir = config.output_path / "smoke"
    smoke_config = replace(
        config,
        smoke=True,
        output_dir=str(smoke_dir),
        physical_batch_size=1,
        num_workers=0,
    )
    write_json(smoke_dir / "config.json", smoke_config.to_dict())
    device = torch.device("cpu")
    seed_everything(smoke_config.seed)
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
        market, daily, fold, smoke_config, logger, True, smoke_dir
    )
    built = build_model(
        smoke_config,
        metadata["target_mean_logrv"],
        metadata["target_scale_logrv"],
        metadata["unconditional_mean_rv"],
    )
    _initialization_assertion(
        built.model, core, metadata["unconditional_mean_rv"], device
    )
    schedule = {
        "H_cos": smoke_config.provisional_horizon_epochs,
        "E_pilot": None,
        "config_hash": stable_hash(smoke_config.to_dict()),
        "locked_from": "smoke_diagnostic_only_not_main_scheduler",
        "created_utc": utc_now(),
    }
    scheduler_path = smoke_dir / "smoke_scheduler_horizon.json"
    if resume and scheduler_path.exists():
        schedule = json.loads(scheduler_path.read_text(encoding="utf-8"))
    else:
        write_json(scheduler_path, schedule)
    scheduler_hash = stable_hash(schedule)
    result = train_model(
        built.model,
        core,
        validation,
        smoke_config,
        horizon_epochs=smoke_config.provisional_horizon_epochs,
        checkpoint_dir=smoke_dir / "checkpoints" / "smoke_fold",
        preprocessor_hash=metadata["preprocessor_hash"],
        scheduler_hash=scheduler_hash,
        fold_name="smoke_fold",
        logger=logger,
        device=device,
        resume=resume,
        max_epochs_override=smoke_config.smoke_epochs,
        max_train_batches=smoke_config.smoke_max_train_batches,
        max_eval_batches=smoke_config.smoke_max_eval_batches,
    )
    _require_numerically_successful(result, "Smoke training")
    predictions = predict_dataset(
        built.model,
        test,
        smoke_config,
        device,
        scheduler_path,
        scheduler_hash,
        max_batches=smoke_config.smoke_max_eval_batches,
    )
    (smoke_dir / "predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(smoke_dir / "predictions" / "smoke_predictions.csv", index=False)
    predictions.to_json(
        smoke_dir / "predictions" / "smoke_predictions.json",
        orient="records",
        indent=2,
    )
    metric = prediction_metrics(
        predictions,
        float(
            np.quantile(
                [
                    market.rv[market.date_to_index[date]]
                    for date in core.target_dates
                ],
                0.90,
            )
        ),
        "smoke_patchtst",
    )
    report = {
        "status": "passed",
        "backend": "CPU",
        "embedding_backend": "deterministic smoke test double (not research output)",
        "actual_input_files": [
            file_fingerprint(Path(smoke_config.market_path)),
            file_fingerprint(Path(smoke_config.news_path)),
        ],
        "parameter_count": built.parameter_count,
        "forecast_queries": built.forecast_queries,
        "attention_backend": attention_backend_metadata(device),
        "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
        "training": result.__dict__,
        "metrics": metric,
    }
    write_json(smoke_dir / "smoke_report.json", report)
    return report


def run_main_pilot(
    config: MainPilotConfig,
    logger: logging.Logger,
    resume: bool,
    confirm_news_filter_reviewed: bool,
) -> dict[str, Any]:
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Full main-pilot is intentionally blocked on CPU; "
            "run --smoke here or execute the GPU command on a CUDA server."
        )
    device = torch.device("cuda")
    seed_everything(config.seed)
    output_dir = config.output_path
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "run_manifest.json",
        {
            "profile": config.profile,
            "seed": config.seed,
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "device": str(device),
            "excluded": ["final_test", "ablation", "COVID_fold", "MCS", "five_seeds"],
            "created_utc": utc_now(),
        },
    )
    logger.info("STEP 1/7 | Load and validate market data through development end")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("STEP 2/7 | Filter news, verify review lock, embed/cache, aggregate")
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

    config_hash = _locked_config_hash(config)
    scheduler_path = output_dir / "scheduler_horizon.json"
    if scheduler_path.exists():
        schedule = json.loads(scheduler_path.read_text(encoding="utf-8"))
        if schedule.get("config_hash") != config_hash:
            raise RuntimeError(
                "Existing scheduler_horizon.json config hash differs; do not silently relock H_cos"
            )
        if schedule.get("pilot_completed_without_numerical_failure") is not True:
            raise RuntimeError(
                "Existing scheduler_horizon.json does not certify a numerically "
                "successful Fold-1 pilot; it cannot be reused"
            )
        logger.info(
            "STEP 3/7 | Reuse already locked scheduler H_cos=%d", schedule["H_cos"]
        )
        orphaned_pilot_dir = (
            output_dir / "checkpoints" / "scheduler_pilot_fold_1_seed_11"
        )
        if orphaned_pilot_dir.exists():
            shutil.rmtree(orphaned_pilot_dir)
            logger.warning(
                "Recovered after scheduler lock: removed orphaned pilot checkpoint"
            )
    else:
        logger.info(
            "STEP 3/7 | Fold 1 scheduler pilot: core+validation only; test is not opened"
        )
        core_1, validation_1, _, metadata_1 = _datasets(
            market,
            daily,
            config.fold_1,
            config,
            logger,
            include_test_diagnostic=False,
            output_dir=output_dir,
        )
        built_pilot = build_model(
            config,
            metadata_1["target_mean_logrv"],
            metadata_1["target_scale_logrv"],
            metadata_1["unconditional_mean_rv"],
        )
        _initialization_assertion(
            built_pilot.model,
            core_1,
            metadata_1["unconditional_mean_rv"],
            device,
        )
        provisional_payload = {
            "H_cos": 200,
            "status": "provisional_pilot_only",
            "config_hash": config_hash,
        }
        provisional_hash = stable_hash(provisional_payload)
        pilot_dir = output_dir / "checkpoints" / "scheduler_pilot_fold_1_seed_11"
        pilot_result = train_model(
            built_pilot.model,
            core_1,
            validation_1,
            config,
            horizon_epochs=config.provisional_horizon_epochs,
            checkpoint_dir=pilot_dir,
            preprocessor_hash=metadata_1["preprocessor_hash"],
            scheduler_hash=provisional_hash,
            fold_name="fold_1_scheduler_pilot_no_test",
            logger=logger,
            device=device,
            resume=resume,
        )
        _require_numerically_successful(
            pilot_result, "Fold 1 scheduler pilot"
        )
        e_pilot = pilot_result.epochs_run
        h_cos = max(10, 5 * math.ceil(e_pilot / 5))
        if not pilot_result.early_stopped and e_pilot >= config.max_epochs:
            h_cos = 200
        schedule = scheduler_payload(
            e_pilot,
            h_cos,
            config_hash,
            pilot_early_stopped=pilot_result.early_stopped,
        )
        write_json(scheduler_path, schedule)
        shutil.rmtree(pilot_dir)
        if pilot_dir.exists():
            raise RuntimeError("Pilot checkpoint destruction failed")
        logger.info(
            "SCHEDULER LOCKED | E_pilot=%d H_cos=%d; pilot checkpoint destroyed",
            e_pilot,
            h_cos,
        )
    scheduler_hash = stable_hash(schedule)

    logger.info("STEP 4/7 | Prepare Fold 5 core/validation/OOS features")
    core_5, validation_5, test_5, metadata_5 = _datasets(
        market,
        daily,
        config.fold_5,
        config,
        logger,
        include_test_diagnostic=True,
        output_dir=output_dir,
    )
    logger.info("STEP 5/7 | Train primary model Fold 5 seed 11 exact QLIKE")
    built_main = build_model(
        config,
        metadata_5["target_mean_logrv"],
        metadata_5["target_scale_logrv"],
        metadata_5["unconditional_mean_rv"],
    )
    _initialization_assertion(
        built_main.model, core_5, metadata_5["unconditional_mean_rv"], device
    )
    write_json(
        output_dir / "model_architecture.json",
        {
            "parameter_count": built_main.parameter_count,
            "forecast_queries": built_main.forecast_queries,
            "attention_backend": attention_backend_metadata(device),
            "market_query_news": True,
            "fine_tokens": 168,
            "coarse_tokens": 240,
            "news_tokens_plus_null": 61,
        },
    )
    training_result = train_model(
        built_main.model,
        core_5,
        validation_5,
        config,
        horizon_epochs=int(schedule["H_cos"]),
        checkpoint_dir=output_dir / "checkpoints" / "fold_5_seed_11",
        preprocessor_hash=metadata_5["preprocessor_hash"],
        scheduler_hash=scheduler_hash,
        fold_name="fold_5_seed_11",
        logger=logger,
        device=device,
        resume=resume,
    )
    _require_numerically_successful(training_result, "Fold 5 training")
    logger.info("STEP 6/7 | OOS Fold 5 prediction and locked baselines")
    deep_predictions = predict_dataset(
        built_main.model,
        test_5,
        config,
        device,
        scheduler_path,
        scheduler_hash,
    )
    baselines = fit_baselines(market, core_5.target_dates, test_5.target_dates)
    write_json(
        output_dir / "metrics" / "baseline_metadata.json",
        {baseline.name: baseline.metadata for baseline in baselines},
    )
    logger.info("STEP 7/7 | Common-support metrics and convergence diagnosis")
    return _report(
        deep_predictions,
        baselines,
        core_5,
        training_result,
        metadata_5,
        built_main.parameter_count,
        output_dir,
        logger,
    )
