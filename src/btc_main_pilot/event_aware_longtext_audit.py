from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .baselines import fit_har_qlike
from .config import Fold, MainPilotConfig, SPIKE_DIAGNOSTIC_FOLDS
from .data import load_market_data, write_market_audit
from .metrics import prediction_metrics
from .news import (
    BITCOIN_CASH_PATTERNS,
    PRIMARY_PATTERNS,
    DeterministicSmokeEncoder,
    FilteredArticle,
    OfflineBgeFinbertEncoder,
    aggregate_daily_news,
    clean_article_text,
    iter_json_array,
    relevance_score,
)
from .news_filter_labeling import _event_family_hint
from .news_representation_audit import (
    LAMBDA_SUM_GRID,
    _dated_probe_frames,
    _fit_probe,
    _prediction_frame,
    _target_rv,
)
from .pipeline import _block_dates
from .preprocess import FoldNewsFeatures, fit_transform_news_for_fold
from .spike_diagnostic import _annotate_predictions, _pooled_metrics
from .utils import (
    ensure_finite,
    file_fingerprint,
    seed_everything,
    stable_hash,
    utc_now,
    write_json,
)


PROFILE = "development-event-aware-longtext-audit"
REPRESENTATION_NAMES = (
    "title_only_pca8_16",
    "title_lead_512_pca8_16",
    "chunk_mean_pca8_16",
    "title_content_separate_pca8_32",
    "finbert_slow_fast_6",
    "surprise_norms_2",
)
CONTEXT_FAMILIES = (
    "regulation_etf",
    "exchange_custody",
    "macro_liquidity",
    "mining_energy",
    "stablecoin_defi",
    "security_hack",
    "other_crypto_context",
)
CRYPTO_ANCHOR = re.compile(
    r"\b(bitcoin(?!\s+cash)|btc|xbt|crypto(?:currency|currencies)?|"
    r"digital asset|blockchain|ethereum|ether|stablecoin|tether|usdt|usdc|"
    r"coinbase|binance|kraken|bitfinex|bitmex|defi)\b",
    re.I,
)
CONCRETE_EVENT = re.compile(
    r"\b(approve|approved|approval|reject|rejected|ban|banned|launch|"
    r"launched|listing|listed|suspend|suspended|shutdown|outage|hack|"
    r"hacked|exploit|breach|stolen|lawsuit|court|regulat|investigat|"
    r"adopt|partnership|acquire|acquisition|filed|filing|default|"
    r"bankrupt|bankruptcy|liquidat|seiz|fine|charged|indicted)\w*\b",
    re.I,
)


@dataclass(frozen=True)
class EventAwarePolicy:
    enabled_context_families: tuple[str, ...]
    relevant_rate_threshold: float
    minimum_family_examples: int
    training_rows: int
    holdout_rows: int
    family_statistics: tuple[dict[str, Any], ...]
    label_source: str


class VariantVectorCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_features (
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                representation TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (content_hash, model, representation)
            )
            """
        )
        self.connection.commit()

    def get(
        self,
        content_hash: str,
        model: str,
        representation: str,
        dimension: int,
    ) -> np.ndarray | None:
        row = self.connection.execute(
            """
            SELECT dimension, vector FROM vector_features
            WHERE content_hash=? AND model=? AND representation=?
            """,
            (content_hash, model, representation),
        ).fetchone()
        if row is None:
            return None
        if int(row[0]) != dimension:
            raise RuntimeError(
                f"Cached {representation} dimension {row[0]} != {dimension}"
            )
        return np.frombuffer(row[1], dtype=np.float32).copy()

    def put(
        self,
        content_hash: str,
        model: str,
        representation: str,
        vector: np.ndarray,
    ) -> None:
        value = np.asarray(vector, dtype=np.float32)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO vector_features
            (content_hash, model, representation, dimension, vector)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                content_hash,
                model,
                representation,
                int(value.size),
                value.tobytes(),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def fit_event_aware_policy(
    expanded_review_path: Path,
    original_holdout_path: Path,
    relevant_rate_threshold: float = 0.50,
    minimum_family_examples: int = 10,
) -> EventAwarePolicy:
    review = pd.read_csv(
        expanded_review_path, dtype=str, keep_default_na=False
    )
    holdout = pd.read_csv(
        original_holdout_path, dtype=str, keep_default_na=False
    )
    holdout_ids = set(holdout["news_cluster_id"])
    training = review[~review["news_cluster_id"].isin(holdout_ids)].copy()
    training = training[
        training["gpt_relevance_label"].isin(["relevant", "irrelevant"])
    ].copy()
    if training.empty:
        raise RuntimeError("No GPT weak labels available outside the holdout")
    training["weight"] = pd.to_numeric(
        training.get("sampling_weight", 1.0), errors="coerce"
    ).fillna(1.0)
    statistics: list[dict[str, Any]] = []
    enabled: list[str] = []
    for family in CONTEXT_FAMILIES:
        group = training[training["event_family_hint"] == family]
        relevant_weight = float(
            group.loc[
                group["gpt_relevance_label"] == "relevant", "weight"
            ].sum()
        )
        total_weight = float(group["weight"].sum())
        rate = relevant_weight / total_weight if total_weight else float("nan")
        active = bool(
            len(group) >= minimum_family_examples
            and np.isfinite(rate)
            and rate >= relevant_rate_threshold
        )
        if active:
            enabled.append(family)
        statistics.append(
            {
                "event_family": family,
                "sample_n": int(len(group)),
                "weighted_relevant_rate": rate,
                "enabled": active,
            }
        )
    return EventAwarePolicy(
        enabled_context_families=tuple(enabled),
        relevant_rate_threshold=relevant_rate_threshold,
        minimum_family_examples=minimum_family_examples,
        training_rows=int(len(training)),
        holdout_rows=int(len(holdout)),
        family_statistics=tuple(statistics),
        label_source=(
            "GPT weak labels outside the locked original 366-row holdout"
        ),
    )


def event_aware_decision(
    title: str,
    cleaned_lead: str,
    policy: EventAwarePolicy,
) -> tuple[bool, int, str, bool]:
    title = clean_article_text(title)
    lead = clean_article_text(cleaned_lead)[:500]
    direct_keep, direct_score = relevance_score(title, lead)
    family = _event_family_hint(title, lead)
    text = f"{title} {lead}"
    price_recap = family == "price_recap_ta"
    cash_in_title = any(pattern.search(title) for pattern in BITCOIN_CASH_PATTERNS)
    independent_title = any(pattern.search(title) for pattern in PRIMARY_PATTERNS)
    if cash_in_title and not independent_title and price_recap:
        return False, 0, "bitcoin_cash_price_recap", True
    if direct_keep:
        score = max(1, direct_score)
        if price_recap:
            score = 1
        return True, score, "direct_bitcoin_rule", price_recap

    enabled = family in policy.enabled_context_families
    crypto_anchor = bool(CRYPTO_ANCHOR.search(text))
    concrete = bool(CONCRETE_EVENT.search(text))
    if (
        enabled
        and crypto_anchor
        and (family != "other_crypto_context" or concrete)
    ):
        return True, 2, f"context_event:{family}", price_recap
    return False, 0, f"no_enabled_event:{family}", price_recap


def load_event_aware_articles(
    path: Path,
    start: str,
    end: str,
    policy: EventAwarePolicy,
    logger: logging.Logger,
    smoke_early_stop: bool = False,
    decision_fn: Callable[
        [str, str, EventAwarePolicy], tuple[bool, int, str, bool]
    ] = event_aware_decision,
) -> tuple[list[FilteredArticle], dict[str, Any]]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    articles: list[FilteredArticle] = []
    counts: dict[str, int] = {
        "records_scanned": 0,
        "outside_date_range": 0,
        "invalid_timestamp": 0,
        "retained": 0,
        "removed": 0,
        "price_recap_retained": 0,
    }
    reasons: dict[str, int] = {}
    previous: pd.Timestamp | None = None
    monotonic = True
    last_report = time.monotonic()
    for raw in iter_json_array(path):
        counts["records_scanned"] += 1
        try:
            timestamp = pd.to_datetime(
                raw.get("canonical_publication_time"),
                utc=True,
                errors="raise",
            )
        except Exception:
            counts["invalid_timestamp"] += 1
            continue
        if previous is not None and timestamp < previous:
            monotonic = False
        previous = timestamp
        if timestamp < start_ts or timestamp >= end_ts:
            counts["outside_date_range"] += 1
            if smoke_early_stop and monotonic and timestamp >= end_ts:
                break
            continue
        title = clean_article_text(raw.get("canonical_title", ""))
        cleaned = clean_article_text(raw.get("canonical_article_text", ""))
        keep, score, reason, price_recap = decision_fn(
            title, cleaned[:500], policy
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if not keep:
            counts["removed"] += 1
            continue
        counts["retained"] += 1
        counts["price_recap_retained"] += int(price_recap)
        articles.append(
            FilteredArticle(
                cluster_id=str(raw.get("news_cluster_id", "")),
                timestamp=timestamp,
                title=title,
                cleaned_text=cleaned,
                source=clean_article_text(raw.get("canonical_source", "")),
                relevance=max(score, 1),
            )
        )
        now = time.monotonic()
        if now - last_report >= 10:
            logger.info(
                "EVENT FILTER | scanned=%d retained=%d removed=%d",
                counts["records_scanned"],
                counts["retained"],
                counts["removed"],
            )
            last_report = now
    audit = {
        **counts,
        "monotonic_timestamp_order": monotonic,
        "enabled_context_families": list(policy.enabled_context_families),
        "decision_reasons": reasons,
        "point_in_time_rule": (
            "Filter uses canonical title and first 500 cleaned characters only; "
            "no market outcome is consulted."
        ),
    }
    logger.info(
        "EVENT FILTER DONE | retained=%d removed=%d outside=%d",
        counts["retained"],
        counts["removed"],
        counts["outside_date_range"],
    )
    return articles, audit


def _token_budgeted_text(
    article: FilteredArticle,
    tokenizer: Any | None,
    max_tokens: int,
    include_title: bool,
) -> str:
    prefix = (
        f"Title: {article.title}\nContent:"
        if include_title
        else "Content:"
    )
    if tokenizer is None:
        prefix_tokens = prefix.split()
        budget = max(max_tokens - len(prefix_tokens) - 2, 1)
        content = " ".join(article.cleaned_text.split()[:budget])
        return f"{prefix} {content}".strip()
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    budget = max(max_tokens - len(prefix_ids) - special, 1)
    content_ids = tokenizer.encode(
        article.cleaned_text, add_special_tokens=False
    )[:budget]
    content = tokenizer.decode(
        content_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return f"{prefix} {content}".strip()


def _article_chunks(
    article: FilteredArticle,
    tokenizer: Any | None,
    max_tokens: int,
) -> list[str]:
    prefix = f"Title: {article.title}\nContent:"
    if tokenizer is None:
        prefix_tokens = prefix.split()
        budget = max(max_tokens - len(prefix_tokens) - 2, 1)
        words = article.cleaned_text.split()
        chunks = [
            " ".join(words[offset : offset + budget])
            for offset in range(0, len(words), budget)
        ]
        return [f"{prefix} {chunk}".strip() for chunk in chunks] or [prefix]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    budget = max(max_tokens - len(prefix_ids) - special, 1)
    content_ids = tokenizer.encode(
        article.cleaned_text, add_special_tokens=False
    )
    chunks = [
        content_ids[offset : offset + budget]
        for offset in range(0, len(content_ids), budget)
    ] or [[]]
    return [
        (
            f"{prefix} "
            + tokenizer.decode(
                chunk,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ).strip()
        for chunk in chunks
    ]


def _variant_hash(
    article: FilteredArticle, representation: str, text: str
) -> str:
    return stable_hash(
        {
            "cluster_id": article.cluster_id,
            "timestamp": article.timestamp.isoformat(),
            "representation": representation,
            "text": text,
        }
    )


def _embed_fixed_variant(
    articles: list[FilteredArticle],
    texts: list[str],
    cache: VariantVectorCache,
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder,
    model: str,
    representation: str,
    dimension: int,
    batch_size: int,
    logger: logging.Logger,
    sentiment: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    vectors = np.empty((len(articles), dimension), dtype=np.float32)
    missing: list[int] = []
    hits = 0
    hashes: list[str] = []
    for index, (article, text) in enumerate(zip(articles, texts, strict=True)):
        content_hash = _variant_hash(article, representation, text)
        hashes.append(content_hash)
        cached = cache.get(content_hash, model, representation, dimension)
        if cached is None:
            missing.append(index)
        else:
            vectors[index] = cached
            hits += 1
    logger.info(
        "VARIANT CACHE | representation=%s total=%d hits=%d missing=%d",
        representation,
        len(articles),
        hits,
        len(missing),
    )
    started = time.monotonic()
    for offset in range(0, len(missing), batch_size):
        indices = missing[offset : offset + batch_size]
        batch_texts = [texts[index] for index in indices]
        batch = (
            encoder.encode_sentiment(batch_texts)
            if sentiment
            else encoder.encode_semantic(batch_texts)
        )
        ensure_finite(f"{representation} embeddings", batch)
        if batch.shape != (len(indices), dimension):
            raise AssertionError(
                f"Unexpected {representation} shape {batch.shape}"
            )
        for local, index in enumerate(indices):
            vectors[index] = batch[local]
            cache.put(hashes[index], model, representation, batch[local])
        cache.commit()
        completed = min(offset + len(indices), len(missing))
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed / elapsed
        eta = (len(missing) - completed) / max(rate, 1e-9)
        batch_number = offset // batch_size + 1
        if batch_number % 20 == 0 or completed == len(missing):
            logger.info(
                "VARIANT EMBED | representation=%s %d/%d (%.1f%%) "
                "rate=%.2f/s ETA=%.1f min",
                representation,
                completed,
                len(missing),
                100.0 * completed / max(len(missing), 1),
                rate,
                eta / 60.0,
            )
    return vectors, {"cache_hits": hits, "cache_misses": len(missing)}


def _embed_chunk_mean(
    articles: list[FilteredArticle],
    cache: VariantVectorCache,
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder,
    tokenizer: Any | None,
    model: str,
    max_tokens: int,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any]]:
    representation = "chunk_mean_all_nonoverlap_512"
    vectors = np.empty((len(articles), 768), dtype=np.float32)
    missing: list[int] = []
    hashes: dict[int, str] = {}
    hits = 0
    for index, article in enumerate(articles):
        content_hash = _variant_hash(
            article, representation, article.encoder_text
        )
        hashes[index] = content_hash
        cached = cache.get(content_hash, model, representation, 768)
        if cached is None:
            missing.append(index)
        else:
            vectors[index] = cached
            hits += 1
    logger.info(
        "VARIANT CACHE | representation=%s total=%d hits=%d missing=%d",
        representation,
        len(articles),
        hits,
        len(missing),
    )
    sums = {index: np.zeros(768, dtype=np.float64) for index in missing}
    counts = {index: 0 for index in missing}
    buffer_texts: list[str] = []
    buffer_indices: list[int] = []
    article_chunk_counts: list[int] = []
    chunks_encoded = 0
    started = time.monotonic()

    def flush() -> None:
        nonlocal chunks_encoded
        if not buffer_texts:
            return
        batch = encoder.encode_semantic(buffer_texts)
        ensure_finite("chunk embeddings", batch)
        for local, article_index in enumerate(buffer_indices):
            sums[article_index] += batch[local].astype(np.float64)
            counts[article_index] += 1
        chunks_encoded += len(buffer_texts)
        buffer_texts.clear()
        buffer_indices.clear()

    for position, index in enumerate(missing, start=1):
        chunks = _article_chunks(articles[index], tokenizer, max_tokens)
        article_chunk_counts.append(len(chunks))
        for chunk in chunks:
            buffer_texts.append(chunk)
            buffer_indices.append(index)
            if len(buffer_texts) >= batch_size:
                flush()
        if position % 250 == 0 or position == len(missing):
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = position / elapsed
            eta = (len(missing) - position) / max(rate, 1e-9)
            logger.info(
                "CHUNK EMBED | articles=%d/%d chunks=%d ETA=%.1f min",
                position,
                len(missing),
                chunks_encoded,
                eta / 60.0,
            )
    flush()
    for index in missing:
        vector = sums[index] / max(counts[index], 1)
        norm = max(float(np.linalg.norm(vector)), 1e-12)
        vector = (vector / norm).astype(np.float32)
        vectors[index] = vector
        cache.put(hashes[index], model, representation, vector)
    cache.commit()
    return vectors, {
        "cache_hits": hits,
        "cache_misses": len(missing),
        "embedded_chunks": int(sum(article_chunk_counts)),
        "chunk_count_p50": (
            float(np.quantile(article_chunk_counts, 0.50))
            if article_chunk_counts
            else 0.0
        ),
        "chunk_count_p90": (
            float(np.quantile(article_chunk_counts, 0.90))
            if article_chunk_counts
            else 0.0
        ),
        "aggregation": "L2-normalized mean of all non-overlapping chunks",
    }


def _build_variant_daily_frames(
    articles: list[FilteredArticle],
    config: MainPilotConfig,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    smoke: bool,
    end: str,
    cache_path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder = (
        DeterministicSmokeEncoder(config.embedding_dim)
        if smoke
        else OfflineBgeFinbertEncoder(config, device)
    )
    tokenizer = (
        None if smoke else getattr(encoder, "semantic_tokenizer", None)
    )
    cache = VariantVectorCache(
        cache_path
        if cache_path is not None
        else output_dir / "cache" / "longtext_embeddings.sqlite"
    )
    try:
        title_texts = [f"Title: {article.title}" for article in articles]
        lead_texts = [
            _token_budgeted_text(
                article, tokenizer, config.max_tokens, include_title=True
            )
            for article in articles
        ]
        content_texts = [
            _token_budgeted_text(
                article, tokenizer, config.max_tokens, include_title=False
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
        title_vectors, title_stats = _embed_fixed_variant(
            articles,
            title_texts,
            cache,
            encoder,
            semantic_model,
            "title_only",
            768,
            config.embedding_batch_size,
            logger,
        )
        lead_vectors, lead_stats = _embed_fixed_variant(
            articles,
            lead_texts,
            cache,
            encoder,
            semantic_model,
            "title_lead_token_budget_512",
            768,
            config.embedding_batch_size,
            logger,
        )
        content_vectors, content_stats = _embed_fixed_variant(
            articles,
            content_texts,
            cache,
            encoder,
            semantic_model,
            "content_only_token_budget_512",
            768,
            config.embedding_batch_size,
            logger,
        )
        sentiments, sentiment_stats = _embed_fixed_variant(
            articles,
            lead_texts,
            cache,
            encoder,
            sentiment_model,
            "finbert_title_lead_token_budget_512",
            3,
            config.embedding_batch_size,
            logger,
            sentiment=True,
        )
        chunk_vectors, chunk_stats = _embed_chunk_mean(
            articles,
            cache,
            encoder,
            tokenizer,
            semantic_model,
            config.max_tokens,
            config.embedding_batch_size,
            logger,
        )
    finally:
        cache.close()
    daily = {
        "title": aggregate_daily_news(
            articles,
            title_vectors,
            sentiments,
            config.research_start,
            end,
        ),
        "lead": aggregate_daily_news(
            articles,
            lead_vectors,
            sentiments,
            config.research_start,
            end,
        ),
        "chunk": aggregate_daily_news(
            articles,
            chunk_vectors,
            sentiments,
            config.research_start,
            end,
        ),
        "content": aggregate_daily_news(
            articles,
            content_vectors,
            sentiments,
            config.research_start,
            end,
        ),
    }
    return daily, {
        "title_only": title_stats,
        "title_lead_512": lead_stats,
        "content_only_512": content_stats,
        "finbert_title_lead_512": sentiment_stats,
        "chunk_mean": chunk_stats,
    }


def _fold_feature(
    daily: pd.DataFrame,
    fold: Fold,
    config: MainPilotConfig,
    logger: logging.Logger,
) -> FoldNewsFeatures:
    return fit_transform_news_for_fold(
        daily,
        fold.core_start,
        fold.core_end,
        fold.validation_start,
        fold.validation_end,
        fold.test_start,
        fold.test_end,
        replace(config, pca_dim=8),
        logger,
    )


def _feature_names(name: str) -> list[str]:
    semantic = [
        f"semantic_{state}_pc_{index:02d}"
        for state in ("slow", "fast")
        for index in range(1, 9)
    ]
    if name == "title_content_separate_pca8_32":
        return [
            f"{stream}_{feature}"
            for stream in ("title", "content")
            for feature in semantic
        ]
    if name == "finbert_slow_fast_6":
        return [
            f"sentiment_{state}_{label}"
            for state in ("slow", "fast")
            for label in ("positive", "negative", "neutral")
        ]
    if name == "surprise_norms_2":
        return ["semantic_fast_l2", "sentiment_fast_l2"]
    return semantic


def _feature_row(
    target: pd.Timestamp,
    features: dict[str, FoldNewsFeatures],
    name: str,
) -> np.ndarray:
    index = int(
        features["lead"].dates.get_loc(target - pd.Timedelta(days=1))
    )
    if name == "title_only_pca8_16":
        source = features["title"]
        return np.concatenate(
            [source.semantic_slow[index], source.semantic_fast[index]]
        ).astype(np.float64)
    if name == "title_lead_512_pca8_16":
        source = features["lead"]
        return np.concatenate(
            [source.semantic_slow[index], source.semantic_fast[index]]
        ).astype(np.float64)
    if name == "chunk_mean_pca8_16":
        source = features["chunk"]
        return np.concatenate(
            [source.semantic_slow[index], source.semantic_fast[index]]
        ).astype(np.float64)
    if name == "title_content_separate_pca8_32":
        title = features["title"]
        content = features["content"]
        return np.concatenate(
            [
                title.semantic_slow[index],
                title.semantic_fast[index],
                content.semantic_slow[index],
                content.semantic_fast[index],
            ]
        ).astype(np.float64)
    if name == "finbert_slow_fast_6":
        source = features["lead"]
        return np.concatenate(
            [source.sentiment_slow[index], source.sentiment_fast[index]]
        ).astype(np.float64)
    if name == "surprise_norms_2":
        source = features["lead"]
        return np.asarray(
            [
                np.linalg.norm(source.semantic_fast[index]),
                np.linalg.norm(source.sentiment_fast[index]),
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown representation: {name}")


def _feature_matrix(
    dates: list[pd.Timestamp],
    features: dict[str, FoldNewsFeatures],
    name: str,
) -> np.ndarray:
    matrix = np.stack(
        [_feature_row(date, features, name) for date in dates]
    )
    ensure_finite(f"{name} feature matrix", matrix)
    return matrix


def _screen(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    min_delta: float,
) -> dict[str, Any]:
    anchor_fold = fold_metrics[
        fold_metrics["model"] == "har_qlike"
    ].set_index("fold")
    anchor_pooled = pooled_metrics[
        pooled_metrics["model"] == "har_qlike"
    ].iloc[0]
    candidates: dict[str, Any] = {}
    for name in REPRESENTATION_NAMES:
        candidate_fold = fold_metrics[
            fold_metrics["model"] == name
        ].set_index("fold")
        pooled = pooled_metrics[pooled_metrics["model"] == name].iloc[0]
        overall_wins = sum(
            candidate_fold.loc[fold, "mean_qlike"]
            < anchor_fold.loc[fold, "mean_qlike"] - min_delta
            for fold in anchor_fold.index
        )
        normal_wins = sum(
            candidate_fold.loc[fold, "normal_qlike"]
            < anchor_fold.loc[fold, "normal_qlike"] - min_delta
            for fold in anchor_fold.index
        )
        spike_wins = sum(
            candidate_fold.loc[fold, "spike_qlike"]
            < anchor_fold.loc[fold, "spike_qlike"] - min_delta
            for fold in anchor_fold.index
        )
        overall_screen = bool(
            overall_wins >= 3
            and pooled["mean_qlike"]
            < anchor_pooled["mean_qlike"] - min_delta
        )
        normal_screen = bool(
            name == "finbert_slow_fast_6"
            and normal_wins >= 3
            and pooled["normal_qlike"]
            < anchor_pooled["normal_qlike"] - min_delta
            and pooled["mean_qlike"]
            <= anchor_pooled["mean_qlike"] + min_delta
        )
        spike_screen = bool(
            name == "surprise_norms_2"
            and spike_wins >= 3
            and pooled["spike_qlike"]
            < anchor_pooled["spike_qlike"] - min_delta
            and pooled["mean_qlike"]
            <= anchor_pooled["mean_qlike"] + min_delta
        )
        candidates[name] = {
            "overall_fold_wins": int(overall_wins),
            "normal_fold_wins": int(normal_wins),
            "spike_fold_wins": int(spike_wins),
            "pooled_overall_delta": float(
                pooled["mean_qlike"] - anchor_pooled["mean_qlike"]
            ),
            "pooled_normal_delta": float(
                pooled["normal_qlike"] - anchor_pooled["normal_qlike"]
            ),
            "pooled_spike_delta": float(
                pooled["spike_qlike"] - anchor_pooled["spike_qlike"]
            ),
            "passes_overall_screen": overall_screen,
            "passes_normal_screen": normal_screen,
            "passes_spike_screen": spike_screen,
            "recommended_for_deep_followup": bool(
                overall_screen or normal_screen or spike_screen
            ),
        }
    return {
        "scope": "development-only rolling Fold 1-4, seed 11",
        "overall_rule": "Beat HAR overall in at least 3/4 folds and pooled.",
        "finbert_normal_rule": (
            "FinBERT normal QLIKE beats HAR in at least 3/4 folds and pooled; "
            "pooled overall is not worse."
        ),
        "surprise_spike_rule": (
            "Surprise spike QLIKE beats HAR in at least 3/4 folds and pooled; "
            "pooled overall is not worse."
        ),
        "candidates": candidates,
        "statistical_claim": "None; development representation screening only.",
    }


def _run_probes(
    config: MainPilotConfig,
    logger: logging.Logger,
    market: Any,
    daily: dict[str, pd.DataFrame],
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
        name: [] for name in ("har_qlike", *REPRESENTATION_NAMES)
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        logger.info(
            "LONGTEXT FOLD %d/%d | %s fit PCA8 on core only",
            fold_index,
            len(folds),
            fold.name,
        )
        core_dates, validation_dates, test_dates = _block_dates(
            market, fold, output_dir, logger, include_test=True
        )
        features = {
            name: _fold_feature(frame, fold, config, logger)
            for name, frame in daily.items()
        }
        write_json(
            features_dir / f"{fold.name}_metadata.json",
            {name: value.metadata for name, value in features.items()},
        )
        anchor = fit_har_qlike(market, core_dates)
        core_rv = _target_rv(market, core_dates)
        validation_rv = _target_rv(market, validation_dates)
        test_rv = _target_rv(market, test_dates)
        core_anchor = anchor.predict_log_rv(market, core_dates)
        validation_anchor = anchor.predict_log_rv(market, validation_dates)
        test_anchor = anchor.predict_log_rv(market, test_dates)
        threshold = float(np.quantile(core_rv, 0.90))
        anchor_test = _prediction_frame(
            test_dates,
            test_rv,
            test_anchor,
            np.zeros(len(test_dates)),
        )
        all_predictions["har_qlike"].append(
            _annotate_predictions(
                anchor_test, fold.name, "har_qlike", threshold
            )
        )
        anchor_metrics = prediction_metrics(
            anchor_test, threshold, "har_qlike"
        )
        anchor_metrics.update(
            {
                "fold": fold.name,
                "correction_selected": False,
                "validation_improvement": 0.0,
            }
        )
        fold_rows.append(anchor_metrics)
        for probe_index, name in enumerate(REPRESENTATION_NAMES, start=1):
            logger.info(
                "LONGTEXT PROBE %d/%d | fold=%s representation=%s",
                (fold_index - 1) * len(REPRESENTATION_NAMES) + probe_index,
                len(folds) * len(REPRESENTATION_NAMES),
                fold.name,
                name,
            )
            matrices = [
                _feature_matrix(dates, features, name)
                for dates in (core_dates, validation_dates, test_dates)
            ]
            fitted = _fit_probe(
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
                lambda_grid,
                config.min_delta,
            )
            validation_frame, test_frame, metadata, grid = (
                _dated_probe_frames(
                    validation_dates, test_dates, fitted
                )
            )
            label = f"{fold.name}_{name}"
            validation_frame.to_csv(
                predictions_dir / f"{label}_validation.csv", index=False
            )
            test_frame.to_csv(
                predictions_dir / f"{label}.csv", index=False
            )
            grid.to_csv(
                metrics_dir / f"{label}_lambda_grid.csv", index=False
            )
            metrics = prediction_metrics(test_frame, threshold, name)
            metrics.update({"fold": fold.name, **metadata})
            write_json(metrics_dir / f"{label}.json", metrics)
            fold_rows.append(metrics)
            all_predictions[name].append(
                _annotate_predictions(
                    test_frame, fold.name, name, threshold
                )
            )
            logger.info(
                "LONGTEXT OOS | fold=%s representation=%s QLIKE=%.8f "
                "normal=%s spike=%s selected=%s",
                fold.name,
                name,
                metrics["mean_qlike"],
                metrics["normal_qlike"],
                metrics["spike_qlike"],
                metadata["correction_selected"],
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
    fold_frame.to_csv(output_dir / "metrics" / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(
        output_dir / "metrics" / "pooled_metrics.csv", index=False
    )
    screen = _screen(fold_frame, pooled_frame, config.min_delta)
    write_json(output_dir / "metrics" / "representation_screen.json", screen)
    return {
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "screen": screen,
    }


def _weighted_binary_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, Any]:
    valid = frame[
        frame["silver_relevance_label"].isin(["relevant", "irrelevant"])
    ].copy()
    truth = valid["silver_relevance_label"] == "relevant"
    predicted = valid[prediction_column].astype(bool)
    weight = pd.to_numeric(valid["holdout_sampling_weight"])
    tp = float(weight[truth & predicted].sum())
    fp = float(weight[~truth & predicted].sum())
    fn = float(weight[truth & ~predicted].sum())
    tn = float(weight[~truth & ~predicted].sum())
    return {
        "n": int(len(valid)),
        "weighted_tp": tp,
        "weighted_fp": fp,
        "weighted_fn": fn,
        "weighted_tn": tn,
        "precision_proxy": tp / (tp + fp) if tp + fp else None,
        "recall_proxy": tp / (tp + fn) if tp + fn else None,
        "specificity_proxy": tn / (tn + fp) if tn + fp else None,
    }


def evaluate_filter_on_silver_holdout(
    silver_path: Path,
    policy: EventAwarePolicy,
    output_dir: Path,
    decision_fn: Callable[
        [str, str, EventAwarePolicy], tuple[bool, int, str, bool]
    ] = event_aware_decision,
    warning: str | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(silver_path, dtype=str, keep_default_na=False)
    strata = [
        "decision",
        "year",
        "evidence_type",
        "score_band",
    ]
    sample_counts = frame.groupby(strata)["news_cluster_id"].transform("count")
    population = pd.to_numeric(frame["stratum_population_n"])
    frame["holdout_sampling_weight"] = population / sample_counts
    decisions = [
        decision_fn(
            str(row.canonical_title),
            str(row.cleaned_lead),
            policy,
        )
        for row in frame.itertuples(index=False)
    ]
    frame["current_filter_keep"] = frame["decision"] == "retained"
    frame["event_aware_keep"] = [value[0] for value in decisions]
    frame["event_aware_score"] = [value[1] for value in decisions]
    frame["event_aware_reason"] = [value[2] for value in decisions]
    frame["event_aware_price_recap"] = [value[3] for value in decisions]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_dir / "silver_holdout_filter_predictions.csv", index=False
    )
    overall = {
        "current_filter": _weighted_binary_metrics(
            frame, "current_filter_keep"
        ),
        "event_aware_filter": _weighted_binary_metrics(
            frame, "event_aware_keep"
        ),
    }
    group_rows: list[dict[str, Any]] = []
    for group_column in ("year", "decision", "score_band", "evidence_type"):
        for group_value, group in frame.groupby(group_column):
            for filter_name, column in (
                ("current_filter", "current_filter_keep"),
                ("event_aware_filter", "event_aware_keep"),
            ):
                group_rows.append(
                    {
                        "group_column": group_column,
                        "group_value": group_value,
                        "filter": filter_name,
                        **_weighted_binary_metrics(group, column),
                    }
                )
    pd.DataFrame(group_rows).to_csv(
        output_dir / "silver_filter_metrics_by_group.csv", index=False
    )
    report = {
        "status": "completed",
        "holdout_n": int(len(frame)),
        "overall": overall,
        "policy": asdict(policy),
        "warning": warning
        or (
            "Precision/recall are GPT-silver proxies, not expert-ground-truth "
            "estimates."
        ),
        "completed_utc": utc_now(),
    }
    write_json(output_dir / "silver_filter_evaluation.json", report)
    return report


def run_development_event_aware_longtext_audit(
    config: MainPilotConfig,
    logger: logging.Logger,
    review_audit_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    if config.profile != PROFILE:
        raise ValueError("Wrong profile for event-aware long-text audit")
    output_dir = config.output_path
    expanded = (
        review_audit_dir / "stratified_news_filter_review.csv"
    )
    original = (
        review_audit_dir / "stratified_news_filter_review_original_366.csv"
    )
    silver = output_dir / "audit" / "gpt_silver_holdout_366.csv"
    for required in (expanded, original, silver):
        if not required.exists():
            raise RuntimeError(
                f"Required audit artifact is missing: {required}. Run the "
                "GPT-silver holdout labeling stage first."
            )
    policy = fit_event_aware_policy(expanded, original)
    signature = stable_hash(
        {
            "config": config.to_dict(),
            "market": file_fingerprint(Path(config.market_path)),
            "news": file_fingerprint(Path(config.news_path)),
            "expanded_review": file_fingerprint(expanded),
            "silver": file_fingerprint(silver),
            "policy": asdict(policy),
            "representations": REPRESENTATION_NAMES,
        }
    )
    report_path = output_dir / "metrics" / "diagnostic_report.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "completed"
            and report.get("run_signature") == signature
        ):
            logger.info("LONGTEXT RESUME | completed report reused")
            return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full long-text embedding audit is blocked "
            "on CPU; use --smoke."
        )
    seed_everything(config.seed)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "audit" / "event_aware_policy.json", asdict(policy))
    silver_eval = evaluate_filter_on_silver_holdout(
        silver, policy, output_dir / "audit"
    )
    logger.info("LONGTEXT STEP 1/4 | Load development market")
    market = load_market_data(
        Path(config.market_path),
        config,
        logger,
        config.research_start,
        config.development_end,
    )
    write_market_audit(market, output_dir)
    logger.info("LONGTEXT STEP 2/4 | Apply locked event-aware filter")
    articles, filter_audit = load_event_aware_articles(
        Path(config.news_path),
        config.research_start,
        config.development_end,
        policy,
        logger,
    )
    write_json(output_dir / "audit" / "event_filter_data_audit.json", filter_audit)
    logger.info("LONGTEXT STEP 3/4 | Embed four representations")
    daily, embedding_audit = _build_variant_daily_frames(
        articles,
        config,
        output_dir,
        torch.device("cuda"),
        logger,
        smoke=False,
        end=config.development_end,
    )
    write_json(output_dir / "audit" / "embedding_audit.json", embedding_audit)
    logger.info("LONGTEXT STEP 4/4 | Run Fold 1-4 HAR-anchored probes")
    results = _run_probes(
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
        "policy": asdict(policy),
        "silver_filter_evaluation": silver_eval,
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        **results,
        "excluded": [
            "Fold_5",
            "final_test",
            "deep_model_training",
            "PCA16",
            "legacy_combined_PCA_probes",
            "MCS",
            "five_seeds",
        ],
        "statistical_claim": "None; development-only diagnostic.",
        "completed_utc": utc_now(),
    }
    write_json(report_path, report)
    logger.info("EVENT-AWARE LONGTEXT AUDIT COMPLETED | %s", report_path)
    return report


def run_event_aware_longtext_smoke(
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
        enabled_context_families=CONTEXT_FAMILIES,
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
        "2018-04-15",
        "2018-04-16",
        "2018-04-30",
        "2018-05-01",
        "2018-05-31",
    )
    results = _run_probes(
        smoke_config,
        logger,
        market,
        daily,
        smoke_dir,
        (fold,),
        lambda_grid=(1.0,),
    )
    report_path = smoke_dir / "event_aware_longtext_smoke.json"
    report = {
        "status": "passed",
        "backend": "CPU deterministic smoke test double",
        "metrics": str(report_path),
        "filter_audit": filter_audit,
        "embedding_audit": embedding_audit,
        **results,
    }
    write_json(report_path, report)
    return report
