from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import MainPilotConfig
from .pipeline import run_main_pilot, run_review_only, run_smoke
from .news_representation_audit import (
    run_development_news_representation_audit,
    run_news_representation_review,
    run_news_representation_smoke,
)
from .news_filter_labeling import run_gpt_news_filter_audit
from .news_filter_labeling import run_gpt_silver_holdout
from .event_aware_longtext_audit import (
    run_development_event_aware_longtext_audit,
    run_event_aware_longtext_smoke,
)
from .point_in_time_gate_diagnostic import (
    run_development_point_in_time_gate_diagnostic,
    run_point_in_time_gate_smoke,
)
from .point_in_time_refit_diagnostic import (
    run_development_point_in_time_refit_diagnostic,
    run_point_in_time_refit_smoke,
)
from .regime_anchor_diagnostic import (
    run_development_regime_anchor_diagnostic,
    run_regime_anchor_smoke,
)
from .spike_diagnostic import (
    run_development_spike_diagnostic,
    run_spike_diagnostic_smoke,
)
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc-main-pilot",
        description=(
            "Run the locked main-pilot or a separate development-only "
            "diagnostic."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=[
            "main-pilot",
            "development-spike-diagnostic",
            "development-regime-anchor-diagnostic",
            "development-news-representation-audit",
            "development-event-aware-longtext-audit",
            "development-point-in-time-gate-diagnostic",
            "development-point-in-time-refit-diagnostic",
        ],
        default="main-pilot",
        help="Development diagnostics run Fold 1-4 only and never open final test.",
    )
    parser.add_argument(
        "--market",
        default="data/BTCUSDT_5min_2018_2025_present.csv",
    )
    parser.add_argument("--news", default="data/news_clusters.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help=(
            "Optional shared article-embedding SQLite cache. Diagnostics "
            "defaults to outputs/main_pilot/cache/article_embeddings.sqlite."
        ),
    )
    parser.add_argument(
        "--scheduler-path",
        default="outputs/main_pilot/scheduler_horizon.json",
        help="Successful locked main-pilot scheduler reused by the diagnostic.",
    )
    parser.add_argument(
        "--physical-batch-size",
        type=int,
        default=32,
        help="Physical batch only; gradient accumulation preserves effective batch 32.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible checkpoints and reuse embedding cache.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU-safe small-data smoke test; never produces research results.",
    )
    parser.add_argument(
        "--review-news-only",
        action="store_true",
        help="Export the development-only manual filter-review sample and stop.",
    )
    parser.add_argument(
        "--confirm-news-filter-reviewed",
        action="store_true",
        help="Operator attests the exported development-only filter sample was reviewed.",
    )
    parser.add_argument(
        "--label-news-review",
        action="store_true",
        help=(
            "Expand and label the development-only news-filter review with the "
            "OpenAI API, then stop. No market outcomes are sent."
        ),
    )
    parser.add_argument(
        "--prepare-news-review-only",
        action="store_true",
        help="Expand the news-filter review sample without making API calls.",
    )
    parser.add_argument("--review-target-size", type=int, default=1200)
    parser.add_argument("--review-batch-size", type=int, default=16)
    parser.add_argument(
        "--review-model",
        default="gpt-5.6-terra",
        help="OpenAI model used for weak-label annotation.",
    )
    parser.add_argument(
        "--review-key-dpapi",
        default=None,
        help=(
            "Windows DPAPI credential file. OPENAI_API_KEY takes precedence. "
            "The key is never written to outputs or logs."
        ),
    )
    parser.add_argument(
        "--review-audit-dir",
        default="outputs/news_representation_audit/audit",
        help="Directory containing the expanded 1200-row review and original 366 holdout.",
    )
    parser.add_argument(
        "--label-silver-holdout",
        action="store_true",
        help=(
            "Blindly relabel the locked original 366-row holdout and optionally "
            "adjudicate disagreements, then stop."
        ),
    )
    parser.add_argument("--silver-model", default="gpt-5.6-sol")
    parser.add_argument("--silver-batch-size", type=int, default=12)
    parser.add_argument(
        "--silver-adjudicate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--silver-holdout-path",
        default=(
            "outputs/event_aware_longtext_audit/audit/"
            "gpt_silver_holdout_366.csv"
        ),
        help="GPT-silver calibration CSV used by the adaptive gate diagnostic.",
    )
    parser.add_argument(
        "--longtext-cache",
        default=(
            "outputs/event_aware_longtext_audit/cache/"
            "longtext_embeddings.sqlite"
        ),
        help="Shared long-text embedding cache reused by the gate diagnostic.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    default_outputs = {
        "main-pilot": "outputs/main_pilot",
        "development-spike-diagnostic": "outputs/spike_diagnostic",
        "development-regime-anchor-diagnostic": (
            "outputs/regime_anchor_diagnostic"
        ),
        "development-news-representation-audit": (
            "outputs/news_representation_audit"
        ),
        "development-event-aware-longtext-audit": (
            "outputs/event_aware_longtext_audit"
        ),
        "development-point-in-time-gate-diagnostic": (
            "outputs/point_in_time_gate_diagnostic"
        ),
        "development-point-in-time-refit-diagnostic": (
            "outputs/point_in_time_refit_diagnostic"
        ),
    }
    output_dir = args.output_dir or default_outputs[args.profile]
    embedding_cache = args.embedding_cache
    if (
        args.profile != "main-pilot"
        and embedding_cache is None
    ):
        embedding_cache = "outputs/main_pilot/cache/article_embeddings.sqlite"
    config = MainPilotConfig(
        profile=args.profile,
        market_path=args.market,
        news_path=args.news,
        output_dir=output_dir,
        embedding_cache_path=embedding_cache,
        physical_batch_size=args.physical_batch_size,
        num_workers=args.num_workers,
    )
    config.validate()
    logger = setup_logging(Path(output_dir))
    if args.label_silver_holdout:
        if args.profile != "development-event-aware-longtext-audit":
            raise ValueError(
                "Silver holdout labeling requires "
                "--profile development-event-aware-longtext-audit"
            )
        review_dir = Path(args.review_audit_dir)
        default_dpapi = (
            Path(os.environ["LOCALAPPDATA"])
            / "btc-news-audit"
            / "openai_api_key.dpapi"
            if os.name == "nt" and "LOCALAPPDATA" in os.environ
            else None
        )
        run_gpt_silver_holdout(
            expanded_review_path=(
                review_dir / "stratified_news_filter_review.csv"
            ),
            original_review_path=(
                review_dir
                / "stratified_news_filter_review_original_366.csv"
            ),
            output_dir=Path(output_dir) / "audit",
            model=args.silver_model,
            batch_size=args.silver_batch_size,
            dpapi_path=(
                Path(args.review_key_dpapi)
                if args.review_key_dpapi
                else default_dpapi
            ),
            logger=logger,
            adjudicate=args.silver_adjudicate,
        )
        return 0
    if args.label_news_review or args.prepare_news_review_only:
        if args.profile != "development-news-representation-audit":
            raise ValueError(
                "News review labeling requires "
                "--profile development-news-representation-audit"
            )
        review_path = (
            Path(output_dir)
            / "audit"
            / "stratified_news_filter_review.csv"
        )
        default_dpapi = (
            Path(os.environ["LOCALAPPDATA"])
            / "btc-news-audit"
            / "openai_api_key.dpapi"
            if os.name == "nt" and "LOCALAPPDATA" in os.environ
            else None
        )
        run_gpt_news_filter_audit(
            news_path=Path(config.news_path),
            review_path=review_path,
            start=config.research_start,
            end=config.development_end,
            target_size=args.review_target_size,
            model=args.review_model,
            batch_size=args.review_batch_size,
            dpapi_path=(
                Path(args.review_key_dpapi)
                if args.review_key_dpapi
                else default_dpapi
            ),
            logger=logger,
            prepare_only=args.prepare_news_review_only,
        )
        return 0
    if args.review_news_only:
        review_path = (
            run_news_representation_review(config, logger)
            if args.profile == "development-news-representation-audit"
            else run_review_only(config, logger)
        )
        logger.info("REVIEW SAMPLE READY | %s", review_path)
        return 0
    if args.smoke:
        if args.profile == "development-spike-diagnostic":
            report = run_spike_diagnostic_smoke(
                config, logger, resume=args.resume
            )
        elif args.profile == "development-regime-anchor-diagnostic":
            report = run_regime_anchor_smoke(
                config, logger, resume=args.resume
            )
        elif args.profile == "development-news-representation-audit":
            report = run_news_representation_smoke(config, logger)
        elif args.profile == "development-event-aware-longtext-audit":
            report = run_event_aware_longtext_smoke(config, logger)
        elif (
            args.profile
            == "development-point-in-time-gate-diagnostic"
        ):
            report = run_point_in_time_gate_smoke(config, logger)
        elif (
            args.profile
            == "development-point-in-time-refit-diagnostic"
        ):
            report = run_point_in_time_refit_smoke(config, logger)
        else:
            report = run_smoke(config, logger, resume=args.resume)
        logger.info(
            "SMOKE %s | %s",
            report["status"].upper(),
            report.get("metrics", report.get("runs")),
        )
        return 0
    if args.profile == "development-spike-diagnostic":
        run_development_spike_diagnostic(
            config,
            logger,
            resume=args.resume,
            confirm_news_filter_reviewed=args.confirm_news_filter_reviewed,
            scheduler_path=Path(args.scheduler_path),
        )
    elif args.profile == "development-regime-anchor-diagnostic":
        run_development_regime_anchor_diagnostic(
            config,
            logger,
            resume=args.resume,
            confirm_news_filter_reviewed=args.confirm_news_filter_reviewed,
            scheduler_path=Path(args.scheduler_path),
        )
    elif args.profile == "development-news-representation-audit":
        run_development_news_representation_audit(
            config,
            logger,
            resume=args.resume,
            confirm_news_filter_reviewed=args.confirm_news_filter_reviewed,
        )
    elif args.profile == "development-event-aware-longtext-audit":
        run_development_event_aware_longtext_audit(
            config,
            logger,
            review_audit_dir=Path(args.review_audit_dir),
            resume=args.resume,
            silver_path=Path(args.silver_holdout_path),
            longtext_cache_path=Path(args.longtext_cache),
        )
    elif args.profile == "development-point-in-time-gate-diagnostic":
        run_development_point_in_time_gate_diagnostic(
            config,
            logger,
            review_audit_dir=Path(args.review_audit_dir),
            silver_path=Path(args.silver_holdout_path),
            longtext_cache_path=Path(args.longtext_cache),
            resume=args.resume,
        )
    elif args.profile == "development-point-in-time-refit-diagnostic":
        run_development_point_in_time_refit_diagnostic(
            config,
            logger,
            review_audit_dir=Path(args.review_audit_dir),
            silver_path=Path(args.silver_holdout_path),
            longtext_cache_path=Path(args.longtext_cache),
            resume=args.resume,
        )
    else:
        run_main_pilot(
            config,
            logger,
            resume=args.resume,
            confirm_news_filter_reviewed=args.confirm_news_filter_reviewed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
