from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from .baselines import (
    ARCH_FAMILY_SPECS,
    fit_arch_family_baselines,
    fit_baselines,
)
from .config import (
    DEVELOPMENT_FOLDS,
    FINAL_HOLDOUT_FOLD,
    Fold,
    MainPilotConfig,
)
from .data import load_market_data, write_market_audit
from .event_aware_longtext_audit import (
    EventAwarePolicy,
    evaluate_filter_on_silver_holdout,
    fit_event_aware_policy,
    load_event_aware_articles,
)
from .pipeline import _block_dates
from .point_in_time_gate_diagnostic import (
    refined_event_decision,
    refined_policy,
)
from .slow_transformer_v2_diagnostic import (
    BLEND_ALPHA_GRID,
    CANDIDATES,
    _run_folds,
    run_slow_transformer_v2_smoke,
)
from .spike_diagnostic import (
    _annotate_predictions,
    _diagnostic_metrics,
    _pooled_metrics,
)
from .utils import (
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)
from .vector_integration_diagnostic import (
    EVENT_FAMILIES,
    _eligible_dates,
    _embed_articles,
    _validate_vector_scheduler,
)


PROFILE = "ml-walk-forward-benchmark"
PRIMARY_SEEDS = (11, 22, 33, 44, 55)
PRIMARY_MODEL = "slow_calendar_control"
PRIMARY_PCA_DIM = 8
BENCHMARK_PHASES = ("development", "final", "all")
ECONOMETRIC_MODELS = (
    "random_walk",
    "har_ols",
    *(spec.name for spec in ARCH_FAMILY_SPECS),
)


def parse_benchmark_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "--benchmark-seeds must be a comma-separated integer list"
        ) from error
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError("--benchmark-seeds must contain positive integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--benchmark-seeds must not contain duplicates")
    return seeds


def _protocol_payload(
    config: MainPilotConfig,
    scheduler_hash: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "profile": PROFILE,
        "development_folds": [asdict(fold) for fold in DEVELOPMENT_FOLDS],
        "final_holdout": asdict(FINAL_HOLDOUT_FOLD),
        "information_cutoff": "t-1",
        "primary_model": PRIMARY_MODEL,
        "primary_pca_dim": PRIMARY_PCA_DIM,
        "locked_representation": {
            "semantic_model": config.semantic_model,
            "sentiment_model": config.sentiment_model,
            "embedding_dim": config.embedding_dim,
            "max_tokens": config.max_tokens,
            "pca_dim": config.pca_dim,
            "slow_alpha": config.slow_alpha,
        },
        "deep_and_text_candidates": list(CANDIDATES),
        "econometric_models": list(ECONOMETRIC_MODELS),
        "arch_family_specs": [
            asdict(spec) for spec in ARCH_FAMILY_SPECS
        ],
        "primary_seeds": list(seeds),
        "required_main_table_seeds": list(PRIMARY_SEEDS),
        "ensemble": "arithmetic mean of predicted RV across seeds",
        "blend_alpha_grid": list(BLEND_ALPHA_GRID),
        "event_families": list(EVENT_FAMILIES),
        "scheduler_hash": scheduler_hash,
        "locked_training": {
            key: getattr(config, key)
            for key in (
                "d_model",
                "attention_heads",
                "news_layers",
                "cross_layers",
                "ffn_dim",
                "dropout",
                "learning_rate",
                "min_learning_rate",
                "weight_decay",
                "warmup_steps",
                "max_epochs",
                "patience",
                "min_delta",
                "gradient_clip_norm",
                "effective_batch_size",
                "training_loss",
                "slow_alpha",
                "news_lookback_days",
            )
        },
        "selection_policy": (
            "Architecture and ablation evidence uses development Fold 1-5 "
            "only. Final outcomes cannot select a model, seed, PCA dimension, "
            "blend, checkpoint rule, or hyperparameter."
        ),
        "metrics": [
            "mean_qlike",
            "sum_qlike",
            "normal_qlike",
            "spike_qlike",
            "rmse_logrv",
            "mae_logrv",
            "mse_logrv",
            "median_ae_logrv",
            "r2_logrv",
            "pearson_logrv",
            "directional_accuracy_logrv",
            "mean_error_logrv",
        ],
        "common_support_required": True,
    }


def _frozen_protocol_path(output_dir: Path) -> Path:
    return output_dir / "frozen_final_protocol.json"


def _freeze_protocol(
    output_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    frozen = {
        "status": "frozen_after_completed_development",
        "protocol_hash": stable_hash(payload),
        "payload": payload,
        "created_utc": utc_now(),
    }
    write_json(_frozen_protocol_path(output_dir), frozen)
    return frozen


def _verify_frozen_protocol(
    output_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    path = _frozen_protocol_path(output_dir)
    if not path.exists():
        raise RuntimeError(
            "Final holdout is blocked because frozen_final_protocol.json is "
            "missing. Complete --benchmark-phase development first."
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    observed_hash = stable_hash(payload)
    if frozen.get("protocol_hash") != observed_hash:
        raise RuntimeError(
            "Final holdout is blocked because the current protocol differs "
            "from the development-frozen protocol. Do not alter the frozen "
            "file; rerun development under a new output directory."
        )
    development_report = (
        output_dir / "development" / "metrics" / "benchmark_report.json"
    )
    if not development_report.exists():
        raise RuntimeError(
            "Final holdout is blocked because the completed development "
            "benchmark report is missing."
        )
    report = json.loads(development_report.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise RuntimeError("Development benchmark is not completed")
    if report.get("primary_model_main_table_eligible") is not True:
        raise RuntimeError(
            "Final holdout is blocked because the frozen primary model did "
            "not complete every development fold for all five primary seeds. "
            "Inspect numerical_failures and resume development first."
        )
    return frozen


def _seed_ensemble(
    phase_dir: Path,
    seeds: tuple[int, ...],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    ensemble_dir = phase_dir / "predictions" / "seed_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    seed_diagnostics: list[dict[str, Any]] = []
    for model in CANDIDATES:
        per_seed: list[pd.DataFrame] = []
        for seed in seeds:
            completion_path = (
                phase_dir
                / f"seed_{seed}"
                / "metrics"
                / "model_completion.json"
            )
            if not completion_path.exists():
                seed_diagnostics.append(
                    {
                        "model": model,
                        "seed": seed,
                        "status": "missing_model_completion_audit",
                        "path": str(completion_path),
                    }
                )
                continue
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            ).get(model, {})
            if not completion.get("eligible_for_seed_ensemble", False):
                seed_diagnostics.append(
                    {
                        "model": model,
                        "seed": seed,
                        "status": "ineligible_incomplete_folds",
                        **completion,
                    }
                )
                continue
            path = (
                phase_dir
                / f"seed_{seed}"
                / "predictions"
                / f"pooled_{model}.csv"
            )
            if not path.exists():
                seed_diagnostics.append(
                    {
                        "model": model,
                        "seed": seed,
                        "status": "missing_prediction",
                        "path": str(path),
                    }
                )
                continue
            frame = pd.read_csv(path)
            frame["target_date"] = frame["target_date"].astype(str)
            frame.sort_values(["fold", "target_date"], inplace=True)
            frame.reset_index(drop=True, inplace=True)
            per_seed.append(frame)
            seed_diagnostics.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "completed",
                    "n_predictions": int(len(frame)),
                }
            )
        if len(per_seed) != len(seeds):
            continue
        keys = per_seed[0][["fold", "target_date"]]
        for frame in per_seed[1:]:
            if not keys.equals(frame[["fold", "target_date"]]):
                raise RuntimeError(
                    f"Seed prediction support differs for model {model}"
                )
            np.testing.assert_allclose(
                per_seed[0]["true_rv"],
                frame["true_rv"],
                rtol=0.0,
                atol=0.0,
            )
        output = per_seed[0].copy()
        predicted_rv = np.mean(
            np.stack(
                [
                    frame["predicted_rv"].to_numpy(dtype=np.float64)
                    for frame in per_seed
                ]
            ),
            axis=0,
        )
        output["predicted_rv"] = predicted_rv
        output["predicted_log_rv"] = np.log(predicted_rv)
        output["predicted_spike"] = (
            output["predicted_rv"] > output["spike_threshold"]
        )
        ratio = output["true_rv"] / output["predicted_rv"]
        output["qlike"] = ratio - np.log(ratio) - 1.0
        output["model"] = model
        output["ensemble_seed_count"] = len(seeds)
        output.to_csv(ensemble_dir / f"{model}.csv", index=False)
        frames[model] = output
    return frames, seed_diagnostics


def _development_seed_summary(
    seed_results: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        [
            {"seed": seed, **row}
            for seed, result in seed_results.items()
            for row in result["pooled_metrics"]
        ]
    )
    if "eligible_for_seed_ensemble" in frame.columns:
        frame = frame[frame["eligible_for_seed_ensemble"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for model, group in frame.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model,
            "seed_count": int(group["seed"].nunique()),
            "seeds": sorted(int(seed) for seed in group["seed"].unique()),
        }
        for metric in (
            "mean_qlike",
            "normal_qlike",
            "spike_qlike",
            "rmse_logrv",
            "mae_logrv",
            "r2_logrv",
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_seed_mean"] = float(values.mean())
            row[f"{metric}_seed_std"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def _common_keys(frames: dict[str, pd.DataFrame]) -> set[tuple[str, str]]:
    supports = []
    for frame in frames.values():
        supports.append(
            set(
                zip(
                    frame["fold"].astype(str),
                    frame["target_date"].astype(str),
                )
            )
        )
    return set.intersection(*supports) if supports else set()


def _filter_to_keys(
    frame: pd.DataFrame,
    keys: set[tuple[str, str]],
) -> pd.DataFrame:
    mask = [
        (str(fold), str(date)) in keys
        for fold, date in zip(frame["fold"], frame["target_date"])
    ]
    output = frame.loc[mask].copy()
    output.sort_values(["fold", "target_date"], inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


def _metric_table(
    frames: dict[str, pd.DataFrame],
    common_only: bool,
) -> pd.DataFrame:
    keys = _common_keys(frames) if common_only else None
    rows = []
    for model, frame in frames.items():
        evaluated = _filter_to_keys(frame, keys) if keys is not None else frame
        rows.append(_pooled_metrics(evaluated, model))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    anchor_rows = result[result["model"] == "har_qlike"]
    if len(anchor_rows) == 1:
        anchor = anchor_rows.iloc[0]
        for metric in ("mean_qlike", "rmse_logrv", "mae_logrv"):
            result[f"delta_{metric}_vs_har_qlike"] = (
                result[metric] - anchor[metric]
            )
    result["qlike_rank"] = result["mean_qlike"].rank(
        method="min", ascending=True
    )
    return result.sort_values(["qlike_rank", "model"])


def _run_econometric_and_common_support(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: Any,
    daily: pd.DataFrame,
    folds: tuple[Fold, ...],
    phase_dir: Path,
    ensemble_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    prediction_dir = phase_dir / "predictions" / "econometric"
    metrics_dir = phase_dir / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pooled: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ECONOMETRIC_MODELS
    }
    metadata: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "ECONOMETRIC FOLD %d/%d | %s fixed core fit",
            fold_index,
            len(folds),
            fold.name,
        )
        core_dates, _, test_dates = _block_dates(
            market,
            fold,
            phase_dir,
            logger,
            include_test=True,
            config=config,
            sampling_rule="har_text",
        )
        core_dates = _eligible_dates(
            core_dates, daily.index, config.news_lookback_days
        )
        test_dates = _eligible_dates(
            test_dates, daily.index, config.news_lookback_days
        )
        true_core = np.asarray(
            [market.rv[market.date_to_index[date]] for date in core_dates],
            dtype=np.float64,
        )
        spike_threshold = float(np.quantile(true_core, 0.90))
        standard = fit_baselines(market, core_dates, test_dates)
        arch_results, arch_failures = fit_arch_family_baselines(
            market, fold, test_dates, logger
        )
        failures.extend(arch_failures)
        for result in [
            item
            for item in standard
            if item.name in {"random_walk", "har_ols"}
        ] + arch_results:
            annotated = _annotate_predictions(
                result.predictions,
                fold.name,
                result.name,
                spike_threshold,
            )
            annotated.to_csv(
                prediction_dir / f"{fold.name}_{result.name}.csv",
                index=False,
            )
            pooled[result.name].append(annotated)
            fold_rows.append(
                {
                    "fold": fold.name,
                    **_diagnostic_metrics(
                        annotated, spike_threshold, result.name
                    ),
                }
            )
            metadata[f"{fold.name}:{result.name}"] = result.metadata

    all_frames = dict(ensemble_frames)
    for model, parts in pooled.items():
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(prediction_dir / f"pooled_{model}.csv", index=False)
        all_frames[model] = frame
    expected_models = set(CANDIDATES) | set(ECONOMETRIC_MODELS)
    completed_models = set(all_frames)
    missing_models = sorted(expected_models - completed_models)
    common_keys = _common_keys(all_frames)
    if not common_keys:
        raise RuntimeError("No common OOS support across completed models")
    common_dir = phase_dir / "predictions" / "common_support"
    common_dir.mkdir(parents=True, exist_ok=True)
    for model, frame in all_frames.items():
        _filter_to_keys(frame, common_keys).to_csv(
            common_dir / f"{model}.csv", index=False
        )
    full_metrics = _metric_table(all_frames, common_only=False)
    common_metrics = _metric_table(all_frames, common_only=True)
    full_metrics.to_csv(
        metrics_dir / "all_model_metrics_full_support.csv", index=False
    )
    common_metrics.to_csv(
        metrics_dir / "all_model_metrics_common_support.csv", index=False
    )
    all_fold_rows: list[dict[str, Any]] = []
    for model, frame in all_frames.items():
        common_frame = _filter_to_keys(frame, common_keys)
        for fold_name, group in common_frame.groupby("fold", sort=False):
            thresholds = group["spike_threshold"].unique()
            if len(thresholds) != 1:
                raise RuntimeError(
                    f"{model}/{fold_name} has inconsistent spike thresholds"
                )
            all_fold_rows.append(
                {
                    "fold": fold_name,
                    **_diagnostic_metrics(
                        group,
                        float(thresholds[0]),
                        model,
                    ),
                }
            )
    all_fold_metrics = pd.DataFrame(all_fold_rows)
    all_fold_metrics.to_csv(
        metrics_dir / "all_model_fold_metrics_common_support.csv",
        index=False,
    )
    stability_rows: list[dict[str, Any]] = []
    anchor_fold = all_fold_metrics[
        all_fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    for model, group in all_fold_metrics.groupby("model", sort=False):
        indexed = group.set_index("fold")
        common_folds = sorted(set(indexed.index) & set(anchor_fold.index))
        row: dict[str, Any] = {
            "model": model,
            "fold_count": len(common_folds),
            "qlike_fold_wins_vs_har": int(
                sum(
                    indexed.loc[name, "mean_qlike"]
                    < anchor_fold.loc[name, "mean_qlike"]
                    for name in common_folds
                )
            ),
        }
        for metric in (
            "mean_qlike",
            "normal_qlike",
            "spike_qlike",
            "rmse_logrv",
            "mae_logrv",
            "r2_logrv",
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_fold_mean"] = float(values.mean())
            row[f"{metric}_fold_std"] = float(values.std(ddof=1))
        stability_rows.append(row)
    pd.DataFrame(stability_rows).to_csv(
        metrics_dir / "all_model_fold_stability.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        metrics_dir / "econometric_fold_metrics.csv", index=False
    )
    write_json(metrics_dir / "econometric_fit_metadata.json", metadata)
    write_json(metrics_dir / "econometric_failures.json", failures)
    return {
        "completed_models": sorted(completed_models),
        "missing_models": missing_models,
        "main_table_eligible_models": sorted(completed_models),
        "supplemental_or_failed_models": missing_models,
        "common_support_n": len(common_keys),
        "common_support_fold_count": int(
            all_fold_metrics["fold"].nunique()
        ),
        "econometric_failures": failures,
        "main_table_eligible_models_complete": not missing_models,
        "common_support_metrics": common_metrics.to_dict(orient="records"),
    }


def _phase_signature(
    config: MainPilotConfig,
    phase: str,
    end: str,
    seeds: tuple[int, ...],
    scheduler_hash: str,
    required_files: tuple[Path, ...],
) -> str:
    return stable_hash(
        {
            "phase": phase,
            "end": end,
            "config": {
                key: value
                for key, value in config.to_dict().items()
                if key not in {"output_dir", "seed"}
            },
            "seeds": seeds,
            "scheduler_hash": scheduler_hash,
            "candidates": CANDIDATES,
            "econometric": [asdict(spec) for spec in ARCH_FAMILY_SPECS],
            "files": [file_fingerprint(path) for path in required_files],
        }
    )


def _run_phase(
    config: MainPilotConfig,
    logger: logging.Logger,
    phase: Literal["development", "final"],
    folds: tuple[Fold, ...],
    end: str,
    seeds: tuple[int, ...],
    policy: EventAwarePolicy,
    scheduler_path: Path,
    scheduler_hash: str,
    scheduler_validation: dict[str, Any],
    review_paths: tuple[Path, Path],
    silver_path: Path,
    longtext_cache_path: Path,
    resume: bool,
) -> dict[str, Any]:
    phase_dir = config.output_path / phase
    (phase_dir / "metrics").mkdir(parents=True, exist_ok=True)
    report_path = phase_dir / "metrics" / "benchmark_report.json"
    signature = _phase_signature(
        config,
        phase,
        end,
        seeds,
        scheduler_hash,
        (
            Path(config.market_path),
            Path(config.news_path),
            *review_paths,
            silver_path,
            scheduler_path,
        ),
    )
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info(
                "ML BENCHMARK RESUME | phase=%s completed report reused",
                phase,
            )
            return report

    write_json(phase_dir / "config.json", config.to_dict())
    write_json(
        phase_dir / "audit" / "scheduler_compatibility.json",
        scheduler_validation,
    )
    write_json(phase_dir / "audit" / "event_aware_policy.json", asdict(policy))
    silver_evaluation = evaluate_filter_on_silver_holdout(
        silver_path,
        policy,
        phase_dir / "audit",
        decision_fn=refined_event_decision,
        warning=(
            "GPT-silver development audit; not expert ground truth or an "
            "independent market-outcome holdout."
        ),
    )
    logger.info(
        "ML BENCHMARK %s STEP 1/5 | Load market through %s",
        phase.upper(),
        end,
    )
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        end,
    )
    write_market_audit(market, phase_dir)
    logger.info(
        "ML BENCHMARK %s STEP 2/5 | Apply frozen event-aware policy",
        phase.upper(),
    )
    articles, filter_audit = load_event_aware_articles(
        Path(config.news_path),
        config.research_start,
        end,
        policy,
        logger,
        decision_fn=refined_event_decision,
    )
    write_json(
        phase_dir / "audit" / "event_filter_data_audit.json", filter_audit
    )
    logger.info(
        "ML BENCHMARK %s STEP 3/5 | Reuse BGE/FinBERT cache",
        phase.upper(),
    )
    daily, semantics, embedding_audit = _embed_articles(
        articles,
        config,
        longtext_cache_path,
        torch.device("cuda"),
        logger,
        smoke=False,
        end=end,
    )
    write_json(
        phase_dir / "audit" / "embedding_audit.json", embedding_audit
    )
    logger.info(
        "ML BENCHMARK %s STEP 4/5 | Deep/text runs seeds=%s",
        phase.upper(),
        ",".join(str(seed) for seed in seeds),
    )
    seed_results: dict[int, dict[str, Any]] = {}
    for seed_index, seed in enumerate(seeds, start=1):
        logger.info(
            "ML BENCHMARK SEED %d/%d | phase=%s seed=%d",
            seed_index,
            len(seeds),
            phase,
            seed,
        )
        seed_config = replace(
            config,
            seed=seed,
            output_dir=str(phase_dir / f"seed_{seed}"),
        )
        seed_everything(seed)
        seed_results[seed] = _run_folds(
            seed_config,
            logger,
            market,
            daily,
            articles,
            semantics,
            Path(seed_config.output_dir),
            folds,
            torch.device("cuda"),
            int(
                json.loads(scheduler_path.read_text(encoding="utf-8"))[
                    "H_cos"
                ]
            ),
            scheduler_path,
            scheduler_hash,
            resume,
            selection_mode=(
                "development_screen"
                if phase == "development"
                else "confirmatory_evaluation"
            ),
        )
    ensemble_frames, seed_diagnostics = _seed_ensemble(
        phase_dir, seeds
    )
    seed_summary = _development_seed_summary(seed_results)
    pd.DataFrame(seed_summary).to_csv(
        phase_dir / "metrics" / "deep_seed_summary.csv", index=False
    )
    write_json(
        phase_dir / "metrics" / "seed_completion.json", seed_diagnostics
    )
    logger.info(
        "ML BENCHMARK %s STEP 5/5 | Econometric baselines and common support",
        phase.upper(),
    )
    comparison = _run_econometric_and_common_support(
        config,
        logger,
        market,
        daily,
        folds,
        phase_dir,
        ensemble_frames,
    )
    exact_primary_seeds = seeds == PRIMARY_SEEDS
    all_numerical_failures = [
        failure
        for result in seed_results.values()
        for failure in result.get("numerical_failures", [])
    ]
    eligible_models = set(comparison["main_table_eligible_models"])
    report = {
        "status": "completed",
        "run_signature": signature,
        "phase": phase,
        "folds": [asdict(fold) for fold in folds],
        "seeds": list(seeds),
        "primary_seeds_complete": exact_primary_seeds,
        "main_table_eligible": bool(
            exact_primary_seeds
            and comparison["main_table_eligible_models_complete"]
        ),
        "primary_model_main_table_eligible": bool(
            exact_primary_seeds and PRIMARY_MODEL in eligible_models
        ),
        "primary_model": PRIMARY_MODEL,
        "primary_pca_dim": PRIMARY_PCA_DIM,
        "scheduler_validation": scheduler_validation,
        "silver_filter_evaluation": silver_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "deep_seed_summary": seed_summary,
        "numerical_failures": all_numerical_failures,
        "numerical_failure_count": len(all_numerical_failures),
        "comparison": comparison,
        "interpretation": (
            "Development results are architecture-selection diagnostics."
            if phase == "development"
            else (
                "Final results are confirmatory only; no post-hoc model or "
                "hyperparameter selection is permitted."
            )
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    return report


def run_ml_walk_forward_benchmark(
    config: MainPilotConfig,
    logger: logging.Logger,
    review_audit_dir: Path,
    silver_path: Path,
    longtext_cache_path: Path,
    scheduler_path: Path,
    resume: bool,
    phase: Literal["development", "final", "all"],
    seeds: tuple[int, ...],
    confirm_open_final_holdout: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != PROFILE:
        raise ValueError("Wrong profile for ML walk-forward benchmark")
    if phase not in BENCHMARK_PHASES:
        raise ValueError(f"Unsupported benchmark phase: {phase}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full ML benchmark is blocked on CPU; use "
            "--smoke locally and run the full profile on a CUDA server."
        )
    expanded = review_audit_dir / "stratified_news_filter_review.csv"
    original = (
        review_audit_dir / "stratified_news_filter_review_original_366.csv"
    )
    required = (expanded, original, silver_path, scheduler_path)
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Required artifact is missing: {path}")
    schedule, scheduler_hash, scheduler_validation = (
        _validate_vector_scheduler(config, scheduler_path)
    )
    del schedule
    policy = refined_policy(fit_event_aware_policy(expanded, original))
    payload = _protocol_payload(config, scheduler_hash, seeds)
    payload["input_fingerprints"] = {
        "market": file_fingerprint(Path(config.market_path)),
        "news": file_fingerprint(Path(config.news_path)),
        "expanded_review": file_fingerprint(expanded),
        "original_366_review": file_fingerprint(original),
        "silver_holdout": file_fingerprint(silver_path),
        "scheduler": file_fingerprint(scheduler_path),
    }
    write_json(config.output_path / "config.json", config.to_dict())
    write_json(
        config.output_path / "run_manifest.json",
        {
            **payload,
            "requested_phase": phase,
            "final_requires_explicit_confirmation": True,
            "created_utc": utc_now(),
        },
    )
    reports: dict[str, Any] = {}
    if phase in {"development", "all"}:
        reports["development"] = _run_phase(
            config,
            logger,
            "development",
            DEVELOPMENT_FOLDS,
            config.development_end,
            seeds,
            policy,
            scheduler_path,
            scheduler_hash,
            scheduler_validation,
            (expanded, original),
            silver_path,
            longtext_cache_path,
            resume,
        )
        _freeze_protocol(config.output_path, payload)
        logger.info(
            "ML BENCHMARK DEVELOPMENT COMPLETED | final protocol frozen at %s",
            _frozen_protocol_path(config.output_path),
        )
    if phase in {"final", "all"}:
        if not confirm_open_final_holdout:
            raise RuntimeError(
                "Opening 2024-04-17..2025-06-30 requires "
                "--confirm-open-final-holdout. Run development first, inspect "
                "and freeze the protocol, then invoke the final phase once."
            )
        if seeds != PRIMARY_SEEDS:
            raise RuntimeError(
                "Final main-table evaluation requires exactly the locked "
                f"primary seeds {PRIMARY_SEEDS}; received {seeds}."
            )
        frozen = _verify_frozen_protocol(config.output_path, payload)
        logger.warning(
            "FINAL HOLDOUT OPENED | 2024-04-17..2025-06-30 | "
            "protocol_hash=%s",
            frozen["protocol_hash"],
        )
        reports["final"] = _run_phase(
            config,
            logger,
            "final",
            (FINAL_HOLDOUT_FOLD,),
            config.research_end,
            seeds,
            policy,
            scheduler_path,
            scheduler_hash,
            scheduler_validation,
            (expanded, original),
            silver_path,
            longtext_cache_path,
            resume,
        )
    summary = {
        "status": "completed",
        "profile": PROFILE,
        "requested_phase": phase,
        "reports": {
            name: str(
                config.output_path
                / name
                / "metrics"
                / "benchmark_report.json"
            )
            for name in reports
        },
        "protocol_hash": stable_hash(payload),
        "completed_utc": utc_now(),
    }
    write_json(config.output_path / "benchmark_summary.json", summary)
    return summary


def run_ml_benchmark_smoke(
    config: MainPilotConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    smoke_report = run_slow_transformer_v2_smoke(config, logger)
    smoke_dir = config.output_path / "smoke"
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.smoke_start,
        config.smoke_end,
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
    har_path = smoke_dir / "predictions" / "pooled_har_qlike.csv"
    har_frame = pd.read_csv(har_path)
    test_dates = [
        pd.Timestamp(value, tz="UTC") for value in har_frame["target_date"]
    ]
    arch_results, failures = fit_arch_family_baselines(
        market,
        fold,
        test_dates,
        logger,
        minimum_core_returns=30,
    )
    if not arch_results:
        raise RuntimeError(
            "Smoke test produced no finite ARCH-family forecast: "
            f"{failures}"
        )
    report = {
        "status": "passed",
        "metrics": str(
            smoke_dir / "metrics" / "pooled_metrics.csv"
        ),
        "deep_smoke": smoke_report,
        "arch_models_completed": [result.name for result in arch_results],
        "arch_failures": failures,
        "final_holdout_opened": False,
    }
    write_json(smoke_dir / "ml_benchmark_smoke_report.json", report)
    return report
