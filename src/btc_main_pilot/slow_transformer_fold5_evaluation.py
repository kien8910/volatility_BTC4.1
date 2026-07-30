from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import Fold, MainPilotConfig
from .data import load_market_data, write_market_audit
from .event_aware_longtext_audit import (
    EventAwarePolicy,
    evaluate_filter_on_silver_holdout,
    fit_event_aware_policy,
    load_event_aware_articles,
)
from .point_in_time_gate_diagnostic import (
    refined_event_decision,
    refined_policy,
)
from .slow_transformer_v2_diagnostic import (
    BLEND_ALPHA_GRID,
    CANDIDATES,
    _run_folds,
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
    _embed_articles,
    _validate_vector_scheduler,
)


PROFILE = "slow-transformer-fold5-evaluation"
PCA_DIMS = (8, 16)
PRIMARY_MODEL = "slow_calendar_control"
PRIMARY_PCA_DIM = 8


def _comparison_rows(
    branches: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_dim: dict[int, dict[str, dict[str, Any]]] = {}
    for branch in branches.values():
        pca_dim = int(branch["pca_dim"])
        by_dim[pca_dim] = {
            str(row["model"]): row for row in branch["pooled_metrics"]
        }
    anchor = by_dim[PRIMARY_PCA_DIM]
    rows: list[dict[str, Any]] = []
    for pca_dim in PCA_DIMS:
        for model, metrics in by_dim[pca_dim].items():
            reference = anchor[model]
            row = {
                "pca_dim": pca_dim,
                **metrics,
                "delta_mean_qlike_vs_pca8": float(
                    metrics["mean_qlike"] - reference["mean_qlike"]
                ),
                "delta_normal_qlike_vs_pca8": float(
                    metrics["normal_qlike"] - reference["normal_qlike"]
                ),
                "delta_spike_qlike_vs_pca8": (
                    float(
                        metrics["spike_qlike"]
                        - reference["spike_qlike"]
                    )
                    if metrics.get("spike_qlike") is not None
                    and reference.get("spike_qlike") is not None
                    else None
                ),
                "delta_r2_logrv_vs_pca8": float(
                    metrics["r2_logrv"] - reference["r2_logrv"]
                ),
                "delta_rmse_logrv_vs_pca8": float(
                    metrics["rmse_logrv"] - reference["rmse_logrv"]
                ),
            }
            rows.append(row)
    return rows


def _fold5_manifest(
    config: MainPilotConfig,
    scheduler_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": PROFILE,
        "scope": "Fold 5 OOS evaluation with PCA8 primary and PCA16 sensitivity",
        "fold": asdict(config.fold_5),
        "candidates": list(CANDIDATES),
        "pca_dims": list(PCA_DIMS),
        "primary_model": PRIMARY_MODEL,
        "primary_pca_dim": PRIMARY_PCA_DIM,
        "seed": config.seed,
        "scheduler_validation": scheduler_validation,
        "information_cutoff": "t-1",
        "prototype_fit_scope": "Fold 5 core only",
        "selection_scope": (
            "Checkpoint, Gamma penalty, and blend alpha use Fold 5 validation "
            "only; no model or PCA choice uses Fold 5 OOS outcomes."
        ),
        "pca16_status": "exploratory sensitivity, not a new primary",
        "prior_access_disclosure": (
            "Fold 5 was previously opened by main-pilot; this is a temporal "
            "generalization diagnostic, not a pristine confirmatory test."
        ),
        "excluded": [
            "post_2024_04_16_final_test",
            "COVID_fold",
            "MCS",
            "five_seeds",
            "post_validation_refit",
            "realized_spike_routing",
        ],
        "created_utc": utc_now(),
    }


def run_slow_transformer_fold5_evaluation(
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
        raise ValueError("Wrong profile for slow-Transformer Fold 5 evaluation")
    if config.pca_dim != PRIMARY_PCA_DIM:
        raise ValueError(
            "The top-level Fold 5 profile must remain PCA8; it runs PCA16 "
            "internally as a predeclared sensitivity branch."
        )
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
            "CUDA is unavailable. Full Fold 5 PCA8/PCA16 evaluation is "
            "blocked on CPU; use --smoke locally."
        )

    policy = refined_policy(fit_event_aware_policy(expanded, original))
    schedule, scheduler_hash, scheduler_validation = (
        _validate_vector_scheduler(config, scheduler_path)
    )
    signature = stable_hash(
        {
            "config": config.to_dict(),
            "fold": asdict(config.fold_5),
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "expanded_review": file_fingerprint(expanded),
            "silver": file_fingerprint(silver_path),
            "policy": asdict(policy),
            "scheduler": schedule,
            "candidates": CANDIDATES,
            "pca_dims": PCA_DIMS,
            "primary_model": PRIMARY_MODEL,
            "blend_alpha_grid": BLEND_ALPHA_GRID,
            "event_families": EVENT_FAMILIES,
        }
    )
    report_path = output_dir / "metrics" / "fold5_pca_report.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info("FOLD 5 PCA RESUME | completed report reused")
            return report

    seed_everything(config.seed)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "audit" / "scheduler_compatibility.json",
        scheduler_validation,
    )
    write_json(output_dir / "audit" / "event_aware_policy.json", asdict(policy))
    write_json(
        output_dir / "run_manifest.json",
        _fold5_manifest(config, scheduler_validation),
    )
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

    logger.info("FOLD 5 PCA STEP 1/5 | Load market through Fold 5 OOS end")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.fold_5.test_end,
    )
    write_market_audit(market, output_dir)
    logger.info("FOLD 5 PCA STEP 2/5 | Apply frozen event-aware news policy")
    articles, filter_audit = load_event_aware_articles(
        Path(config.news_path),
        config.research_start,
        config.fold_5.test_end,
        policy,
        logger,
        decision_fn=refined_event_decision,
    )
    write_json(
        output_dir / "audit" / "event_filter_data_audit.json",
        filter_audit,
    )
    logger.info("FOLD 5 PCA STEP 3/5 | Reuse cached BGE and FinBERT vectors")
    daily, semantics, embedding_audit = _embed_articles(
        articles,
        config,
        longtext_cache_path,
        torch.device("cuda"),
        logger,
        smoke=False,
        end=config.fold_5.test_end,
    )
    write_json(
        output_dir / "audit" / "embedding_audit.json",
        embedding_audit,
    )

    branches: dict[str, dict[str, Any]] = {}
    for branch_index, pca_dim in enumerate(PCA_DIMS, start=1):
        branch_name = f"pca{pca_dim}"
        branch_dir = output_dir / branch_name
        branch_config = replace(
            config,
            pca_dim=pca_dim,
            output_dir=str(branch_dir),
        )
        write_json(branch_dir / "config.json", branch_config.to_dict())
        logger.info(
            "FOLD 5 PCA STEP 4/5 | branch=%d/%d pca_dim=%d",
            branch_index,
            len(PCA_DIMS),
            pca_dim,
        )
        seed_everything(branch_config.seed)
        results = _run_folds(
            branch_config,
            logger,
            market,
            daily,
            articles,
            semantics,
            branch_dir,
            (config.fold_5,),
            torch.device("cuda"),
            int(schedule["H_cos"]),
            scheduler_path,
            scheduler_hash,
            resume,
            selection_mode="single_fold_evaluation",
        )
        branches[branch_name] = {
            "pca_dim": pca_dim,
            "output_dir": str(branch_dir),
            **results,
        }

    logger.info("FOLD 5 PCA STEP 5/5 | Compare PCA16 against frozen PCA8")
    comparison = _comparison_rows(branches)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(
        metrics_dir / "pca_comparison.csv",
        index=False,
    )
    report = {
        "status": "completed",
        "run_signature": signature,
        "fold": asdict(config.fold_5),
        "primary_model": PRIMARY_MODEL,
        "primary_pca_dim": PRIMARY_PCA_DIM,
        "pca16_status": "exploratory sensitivity",
        "scheduler_validation": scheduler_validation,
        "silver_filter_evaluation": silver_evaluation,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        "branches": branches,
        "pca_comparison": comparison,
        "statistical_claim": (
            "None; Fold 5 was previously opened and PCA16 is an exploratory "
            "single-fold sensitivity analysis."
        ),
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("FOLD 5 PCA COMPLETED | %s", report_path)
    return report


def run_slow_transformer_fold5_smoke(
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
    schedule = {"H_cos": 200, "smoke": True, "created_utc": utc_now()}
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
    branches: dict[str, dict[str, Any]] = {}
    for pca_dim in PCA_DIMS:
        branch_name = f"pca{pca_dim}"
        branch_dir = smoke_dir / branch_name
        branch_config = replace(
            smoke_config,
            pca_dim=pca_dim,
            output_dir=str(branch_dir),
        )
        seed_everything(branch_config.seed)
        results = _run_folds(
            branch_config,
            logger,
            market,
            daily,
            articles,
            semantics,
            branch_dir,
            (fold,),
            torch.device("cpu"),
            200,
            scheduler_path,
            scheduler_hash,
            resume=False,
            lambda_grid=(1.0,),
            smoke=True,
            selection_mode="single_fold_evaluation",
        )
        branches[branch_name] = {
            "pca_dim": pca_dim,
            "output_dir": str(branch_dir),
            **results,
        }
    comparison = _comparison_rows(branches)
    report = {
        "status": "passed",
        "backend": "CPU deterministic smoke encoder",
        "metrics": str(smoke_dir / "pca_comparison.csv"),
        "pca_dims": list(PCA_DIMS),
        "branches": branches,
        "pca_comparison": comparison,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
    }
    pd.DataFrame(comparison).to_csv(
        smoke_dir / "pca_comparison.csv",
        index=False,
    )
    write_json(smoke_dir / "fold5_pca_smoke_report.json", report)
    return report
