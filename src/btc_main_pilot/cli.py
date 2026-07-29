from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import MainPilotConfig
from .pipeline import run_main_pilot, run_review_only, run_smoke
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc-main-pilot",
        description=(
            "Run only the locked main-pilot profile from "
            "BTC_PATCHTST_FINBERT_MODEL_SPEC.md"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["main-pilot"],
        default="main-pilot",
        help="Only the requested main-pilot profile is available.",
    )
    parser.add_argument(
        "--market",
        default="data/BTCUSDT_5min_2018_2025_present.csv",
    )
    parser.add_argument("--news", default="data/news_clusters.json")
    parser.add_argument("--output-dir", default="outputs/main_pilot")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = MainPilotConfig(
        profile=args.profile,
        market_path=args.market,
        news_path=args.news,
        output_dir=args.output_dir,
        physical_batch_size=args.physical_batch_size,
        num_workers=args.num_workers,
    )
    config.validate()
    logger = setup_logging(Path(args.output_dir))
    if args.review_news_only:
        review_path = run_review_only(config, logger)
        logger.info("REVIEW SAMPLE READY | %s", review_path)
        return 0
    if args.smoke:
        report = run_smoke(config, logger, resume=args.resume)
        logger.info("SMOKE %s | %s", report["status"].upper(), report["metrics"])
        return 0
    run_main_pilot(
        config,
        logger,
        resume=args.resume,
        confirm_news_filter_reviewed=args.confirm_news_filter_reviewed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

