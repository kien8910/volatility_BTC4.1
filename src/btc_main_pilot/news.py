from __future__ import annotations

import html
import json
import logging
import math
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import MainPilotConfig
from .utils import ensure_finite, file_fingerprint, stable_hash, write_json


ALLOWED_FEATURE_FIELDS = {
    "canonical_publication_time",
    "canonical_title",
    "canonical_article_text",
    "canonical_source",
    "news_cluster_id",
}
FORBIDDEN_FEATURE_FIELDS = {
    "source_count",
    "member_count",
    "republication_offsets_minutes",
    "all_sources",
    "all_urls",
    "all_source_urls",
    "all_title_variants",
    "member_original_row_ids",
}
PRIMARY_PATTERNS = [
    re.compile(r"\bbitcoin\b(?!\s+cash)", re.I),
    re.compile(r"\bbtc\b", re.I),
    re.compile(r"\bxbt\b", re.I),
]
BITCOIN_CASH_PATTERNS = [
    re.compile(r"\bbitcoin cash\b", re.I),
    re.compile(r"\bbch\b", re.I),
]
SPECIFIC_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bsatoshi nakamoto\b",
        r"\bsatoshis?\b",
        r"\blightning network\b",
        r"\bbitcoin core\b",
        r"\bsegwit\b",
        r"\btaproot\b",
        r"\bbitcoin halv(?:ing|en)\b",
        r"\bbitcoin miners?\b",
        r"\bbitcoin mining\b",
        r"\bbitcoin etf\b",
        r"\bspot bitcoin etf\b",
        r"\bbitcoin futures?\b",
        r"\bmt\.?\s*gox\b",
    ]
]
FOOTER_MARKERS = [
    re.compile(r"more from the motley fool", re.I),
    re.compile(r"more from fortune\.com", re.I),
    re.compile(r"see original article", re.I),
    re.compile(r"\brelated stor(?:y|ies)\b", re.I),
    re.compile(r"\bdisclosure\b", re.I),
    re.compile(r"\babout the author\b", re.I),
    re.compile(r"\bauthor biography\b", re.I),
    re.compile(r"\brecommended (?:articles|stories)\b", re.I),
]
HTML_TAG = re.compile(r"<[^>]+>")
BAD_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]+")
WHITESPACE = re.compile(r"\s+")


@dataclass
class FilteredArticle:
    cluster_id: str
    timestamp: pd.Timestamp
    title: str
    cleaned_text: str
    source: str
    relevance: int

    @property
    def encoder_text(self) -> str:
        return f"Title: {self.title}\nContent: {self.cleaned_text}"

    @property
    def content_hash(self) -> str:
        return stable_hash(
            {
                "id": self.cluster_id,
                "timestamp": self.timestamp.isoformat(),
                "text": self.encoder_text,
            }
        )


def clean_article_text(text: Any) -> str:
    value = html.unescape(str(text or ""))
    cut_positions = []
    for marker in FOOTER_MARKERS:
        match = marker.search(value)
        if match:
            cut_positions.append(match.start())
    if cut_positions:
        value = value[: min(cut_positions)]
    value = HTML_TAG.sub(" ", value)
    value = BAD_CHARS.sub(" ", value)
    return WHITESPACE.sub(" ", value).strip()


def relevance_score(title: str, cleaned_text: str) -> tuple[bool, int]:
    title = title or ""
    lead = cleaned_text[:2000]
    lead500 = cleaned_text[:500]
    title_hits = sum(len(pattern.findall(title)) for pattern in PRIMARY_PATTERNS)
    content_matches = [m for pattern in PRIMARY_PATTERNS for m in pattern.finditer(lead)]
    content_hits = len(content_matches)
    title_specific = any(pattern.search(title) for pattern in SPECIFIC_PATTERNS)
    content_specific = any(pattern.search(lead) for pattern in SPECIFIC_PATTERNS)
    independent_hits = title_hits + content_hits
    cash_only = (
        any(pattern.search(f"{title} {lead}") for pattern in BITCOIN_CASH_PATTERNS)
        and independent_hits == 0
        and not title_specific
        and not content_specific
    )
    if cash_only:
        return False, 0

    keep_rule = (
        title_hits > 0
        or title_specific
        or content_hits >= 2
        or (title_hits > 0 and content_hits > 0)
        or content_specific
    )
    score = 0
    if title_hits:
        score += 3
    if any(pattern.search(lead500) for pattern in PRIMARY_PATTERNS):
        score += 2
    if title_specific or content_specific:
        score += 2
    if independent_hits >= 3:
        score += 1
    return bool(keep_rule and score >= 2), score


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without loading the 541 MB file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        return
                    continue
                if buffer[position] != "[":
                    raise ValueError("news_clusters.json must contain a top-level JSON array")
                position += 1
                started = True
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    yield value
                    if position > chunk_size:
                        buffer = buffer[position:]
                        position = 0
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    remainder = buffer[position:]
                    more = handle.read(chunk_size)
                    eof = not more
                    buffer = remainder + more
                    position = 0


def load_filtered_articles(
    path: Path,
    start: str,
    end: str,
    logger: logging.Logger,
    smoke_early_stop: bool = False,
) -> tuple[list[FilteredArticle], dict[str, Any]]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    retained: list[FilteredArticle] = []
    counts: dict[str, int] = {
        "records_scanned": 0,
        "outside_date_range": 0,
        "invalid_timestamp": 0,
        "filtered_irrelevant": 0,
        "retained": 0,
    }
    previous_ts: pd.Timestamp | None = None
    monotonic = True
    strange_timestamps: list[dict[str, str]] = []
    retained_examples: list[dict[str, Any]] = []
    rejected_examples: list[dict[str, Any]] = []
    last_report = time.monotonic()
    for raw in iter_json_array(path):
        counts["records_scanned"] += 1
        try:
            timestamp = pd.to_datetime(
                raw.get("canonical_publication_time"), utc=True, errors="raise"
            )
        except Exception:
            counts["invalid_timestamp"] += 1
            continue
        if previous_ts is not None and timestamp < previous_ts:
            monotonic = False
        previous_ts = timestamp
        if timestamp.hour == 0 and timestamp.minute == 0 and len(str(raw.get("canonical_publication_time", ""))) <= 10:
            strange_timestamps.append(
                {
                    "news_cluster_id": str(raw.get("news_cluster_id", "")),
                    "canonical_publication_time": str(
                        raw.get("canonical_publication_time", "")
                    ),
                    "reason": "date_only_or_midnight_timestamp",
                }
            )
        if timestamp < start_ts or timestamp >= end_ts:
            counts["outside_date_range"] += 1
            if smoke_early_stop and monotonic and timestamp >= end_ts:
                break
            continue
        title = clean_article_text(raw.get("canonical_title", ""))
        cleaned = clean_article_text(raw.get("canonical_article_text", ""))
        keep, score = relevance_score(title, cleaned)
        if not keep:
            counts["filtered_irrelevant"] += 1
            if len(rejected_examples) < 100:
                rejected_examples.append(
                    {
                        "news_cluster_id": str(raw.get("news_cluster_id", "")),
                        "canonical_publication_time": str(timestamp),
                        "canonical_title": title,
                        "cleaned_lead": cleaned[:500],
                        "decision": "removed",
                        "relevance_score": score,
                    }
                )
            continue
        retained.append(
            FilteredArticle(
                cluster_id=str(raw.get("news_cluster_id", "")),
                timestamp=timestamp,
                title=title,
                cleaned_text=cleaned,
                source=clean_article_text(raw.get("canonical_source", "")),
                relevance=score,
            )
        )
        counts["retained"] += 1
        if len(retained_examples) < 100:
            retained_examples.append(
                {
                    "news_cluster_id": str(raw.get("news_cluster_id", "")),
                    "canonical_publication_time": str(timestamp),
                    "canonical_title": title,
                    "cleaned_lead": cleaned[:500],
                    "decision": "retained",
                    "relevance_score": score,
                }
            )
        now = time.monotonic()
        if now - last_report >= 10:
            logger.info(
                "NEWS FILTER | scanned=%d retained=%d removed=%d",
                counts["records_scanned"],
                counts["retained"],
                counts["filtered_irrelevant"],
            )
            last_report = now
    audit: dict[str, Any] = {
        **counts,
        "input": file_fingerprint(path),
        "monotonic_timestamp_order": monotonic,
        "timestamp_limitation": (
            "canonical_publication_time may be publication, crawl, or updated-content "
            "time; the schema cannot prove point-in-time snapshots."
        ),
        "canonical_cluster_limitation": (
            "Canonical title/text and clustering may be retrospective. Forbidden member "
            "and count metadata are excluded from every feature."
        ),
        "strange_timestamp_count": len(strange_timestamps),
        "strange_timestamp_examples": strange_timestamps[:100],
        "manual_review_examples": retained_examples + rejected_examples,
    }
    logger.info(
        "NEWS FILTER DONE | kept=%d removed=%d outside=%d invalid_time=%d",
        counts["retained"],
        counts["filtered_irrelevant"],
        counts["outside_date_range"],
        counts["invalid_timestamp"],
    )
    return retained, audit


class EmbeddingCache:
    def __init__(self, path: Path, semantic_model: str, sentiment_model: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.semantic_model = semantic_model
        self.sentiment_model = sentiment_model
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS article_features (
                content_hash TEXT NOT NULL,
                semantic_model TEXT NOT NULL,
                sentiment_model TEXT NOT NULL,
                semantic BLOB NOT NULL,
                sentiment BLOB NOT NULL,
                created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (content_hash, semantic_model, sentiment_model)
            )
            """
        )
        self.connection.commit()

    def get(self, content_hash: str) -> tuple[np.ndarray, np.ndarray] | None:
        row = self.connection.execute(
            """
            SELECT semantic, sentiment FROM article_features
            WHERE content_hash=? AND semantic_model=? AND sentiment_model=?
            """,
            (content_hash, self.semantic_model, self.sentiment_model),
        ).fetchone()
        if row is None:
            return None
        semantic = np.frombuffer(row[0], dtype=np.float32).copy()
        sentiment = np.frombuffer(row[1], dtype=np.float32).copy()
        return semantic, sentiment

    def put(
        self, content_hash: str, semantic: np.ndarray, sentiment: np.ndarray
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO article_features
            (content_hash, semantic_model, sentiment_model, semantic, sentiment)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                content_hash,
                self.semantic_model,
                self.sentiment_model,
                np.asarray(semantic, dtype=np.float32).tobytes(),
                np.asarray(sentiment, dtype=np.float32).tobytes(),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class OfflineBgeFinbertEncoder:
    def __init__(self, config: MainPilotConfig, device: torch.device):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.device = device
        self.semantic_tokenizer = AutoTokenizer.from_pretrained(
            config.semantic_model, local_files_only=True
        )
        self.semantic_model = AutoModel.from_pretrained(
            config.semantic_model, local_files_only=True
        ).to(device)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            config.sentiment_model, local_files_only=True
        )
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            config.sentiment_model, local_files_only=True
        ).to(device)
        self.semantic_model.eval()
        self.sentiment_model.eval()
        for parameter in list(self.semantic_model.parameters()) + list(
            self.sentiment_model.parameters()
        ):
            parameter.requires_grad_(False)
        labels = {
            int(index): str(label).lower()
            for index, label in self.sentiment_model.config.id2label.items()
        }
        self.output_order = [
            next(index for index, label in labels.items() if "positive" in label),
            next(index for index, label in labels.items() if "negative" in label),
            next(index for index, label in labels.items() if "neutral" in label),
        ]

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        semantic_inputs = self.semantic_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_tokens,
            return_tensors="pt",
        )
        semantic_inputs = {key: value.to(self.device) for key, value in semantic_inputs.items()}
        semantic_output = self.semantic_model(**semantic_inputs).last_hidden_state[:, 0]
        semantic_output = torch.nn.functional.normalize(semantic_output, p=2, dim=-1)
        if semantic_output.shape[1] != self.config.embedding_dim:
            raise AssertionError(
                f"BGE output dimension {semantic_output.shape[1]} != 768"
            )
        sentiment_inputs = self.sentiment_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_tokens,
            return_tensors="pt",
        )
        sentiment_inputs = {
            key: value.to(self.device) for key, value in sentiment_inputs.items()
        }
        logits = self.sentiment_model(**sentiment_inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[:, self.output_order]
        return (
            semantic_output.float().cpu().numpy(),
            probabilities.float().cpu().numpy(),
        )


class DeterministicSmokeEncoder:
    """Clearly labelled test double; never used by a non-smoke run."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        semantics = []
        sentiments = []
        for text in texts:
            seed = int(stable_hash(text)[:16], 16) % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.embedding_dim).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            logits = rng.normal(size=3)
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            semantics.append(vector)
            sentiments.append(probs.astype(np.float32))
        return np.stack(semantics), np.stack(sentiments)


def embed_articles(
    articles: list[FilteredArticle],
    cache: EmbeddingCache,
    encoder: OfflineBgeFinbertEncoder | DeterministicSmokeEncoder,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    semantics = np.empty((len(articles), 768), dtype=np.float32)
    sentiments = np.empty((len(articles), 3), dtype=np.float32)
    missing_indices = []
    cache_hits = 0
    for index, article in enumerate(articles):
        cached = cache.get(article.content_hash)
        if cached is None:
            missing_indices.append(index)
        else:
            semantics[index], sentiments[index] = cached
            cache_hits += 1
    logger.info(
        "EMBED CACHE | total=%d hits=%d missing=%d",
        len(articles),
        cache_hits,
        len(missing_indices),
    )
    started = time.monotonic()
    for offset in range(0, len(missing_indices), batch_size):
        batch_indices = missing_indices[offset : offset + batch_size]
        batch_texts = [articles[index].encoder_text for index in batch_indices]
        semantic_batch, sentiment_batch = encoder.encode(batch_texts)
        ensure_finite("semantic embedding", semantic_batch)
        ensure_finite("sentiment probabilities", sentiment_batch)
        if semantic_batch.shape != (len(batch_indices), 768):
            raise AssertionError(f"Unexpected BGE shape {semantic_batch.shape}")
        if sentiment_batch.shape != (len(batch_indices), 3):
            raise AssertionError(f"Unexpected FinBERT shape {sentiment_batch.shape}")
        for local_index, article_index in enumerate(batch_indices):
            semantics[article_index] = semantic_batch[local_index]
            sentiments[article_index] = sentiment_batch[local_index]
            cache.put(
                articles[article_index].content_hash,
                semantic_batch[local_index],
                sentiment_batch[local_index],
            )
        cache.commit()
        completed = min(offset + len(batch_indices), len(missing_indices))
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed / elapsed
        eta = (len(missing_indices) - completed) / max(rate, 1e-9)
        logger.info(
            "EMBED PROGRESS | %d/%d (%.1f%%) rate=%.2f article/s ETA=%.1f min",
            completed,
            len(missing_indices),
            100.0 * completed / max(len(missing_indices), 1),
            rate,
            eta / 60.0,
        )
    return semantics, sentiments, {
        "cache_hits": cache_hits,
        "cache_misses": len(missing_indices),
    }


DAILY_SCALAR_COLUMNS = [
    "news_intensity",
    "log1p_canonical_source_count",
    "negative_ratio",
    "log1p_negative_count_070",
    "negative_probability_max",
    "negative_probability_std",
    "positive_probability_max",
    "sentiment_entropy_mean",
    "semantic_dispersion",
    "mean_relevance",
    "no_news_dummy",
]
UNDEFINED_NO_NEWS_COLUMNS = [
    "negative_ratio",
    "negative_probability_max",
    "negative_probability_std",
    "positive_probability_max",
    "sentiment_entropy_mean",
    "semantic_dispersion",
    "mean_relevance",
]


def aggregate_daily_news(
    articles: list[FilteredArticle],
    semantics: np.ndarray,
    sentiments: np.ndarray,
    start: str,
    end: str,
) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="D", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["news_count"] = 0
    frame["canonical_source_count"] = 0
    frame["no_news_dummy"] = 1.0
    frame["semantic"] = pd.Series([None] * len(frame), index=index, dtype=object)
    frame["sentiment"] = pd.Series([None] * len(frame), index=index, dtype=object)
    for column in UNDEFINED_NO_NEWS_COLUMNS + ["negative_count_070"]:
        frame[column] = np.nan
    grouped: dict[pd.Timestamp, list[int]] = {}
    for idx, article in enumerate(articles):
        grouped.setdefault(article.timestamp.normalize(), []).append(idx)
    for day, indices in grouped.items():
        if day not in frame.index:
            continue
        weights = np.asarray(
            [articles[index].relevance for index in indices], dtype=np.float64
        )
        weights /= weights.sum()
        day_semantics = semantics[indices].astype(np.float64)
        day_sentiments = sentiments[indices].astype(np.float64)
        centroid = np.sum(day_semantics * weights[:, None], axis=0)
        sentiment = np.sum(day_sentiments * weights[:, None], axis=0)
        labels = np.argmax(day_sentiments, axis=1)
        entropy = -np.sum(
            day_sentiments * np.log(np.clip(day_sentiments, 1e-12, 1.0)), axis=1
        )
        centroid_norm = max(float(np.linalg.norm(centroid)), 1e-12)
        cosine = (day_semantics @ centroid) / np.maximum(
            np.linalg.norm(day_semantics, axis=1) * centroid_norm, 1e-12
        )
        frame.at[day, "news_count"] = len(indices)
        frame.at[day, "canonical_source_count"] = len(
            {articles[index].source for index in indices}
        )
        frame.at[day, "no_news_dummy"] = 0.0
        frame.at[day, "semantic"] = centroid.astype(np.float32)
        frame.at[day, "sentiment"] = sentiment.astype(np.float32)
        frame.at[day, "negative_ratio"] = float(np.mean(labels == 1))
        frame.at[day, "negative_count_070"] = int(
            np.sum(day_sentiments[:, 1] > 0.70)
        )
        frame.at[day, "negative_probability_max"] = float(
            np.max(day_sentiments[:, 1])
        )
        frame.at[day, "negative_probability_std"] = float(
            np.std(day_sentiments[:, 1], ddof=0)
        )
        frame.at[day, "positive_probability_max"] = float(
            np.max(day_sentiments[:, 0])
        )
        frame.at[day, "sentiment_entropy_mean"] = float(np.sum(entropy * weights))
        frame.at[day, "semantic_dispersion"] = float(
            np.sum((1.0 - np.clip(cosine, -1.0, 1.0)) * weights)
        )
        frame.at[day, "mean_relevance"] = float(
            np.mean([articles[index].relevance for index in indices])
        )
    log_count = np.log1p(frame["news_count"].astype(float))
    rolling = log_count.rolling(window=365, min_periods=30).median()
    expanding = log_count.expanding(min_periods=1).median()
    causal_median = rolling.where(rolling.notna(), expanding)
    frame["news_intensity"] = log_count - causal_median
    frame["log1p_canonical_source_count"] = np.log1p(
        frame["canonical_source_count"].astype(float)
    )
    frame["log1p_negative_count_070"] = np.log1p(
        frame["negative_count_070"].fillna(0.0).astype(float)
    )
    return frame


def write_news_audits(
    frame: pd.DataFrame,
    articles: list[FilteredArticle],
    output_dir: Path,
    filter_audit: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    daily = frame[["news_count", "canonical_source_count", "no_news_dummy"]].copy()
    daily.to_csv(audit_dir / "news_daily_counts.csv", index_label="date")
    monthly = daily.resample("MS").agg(
        news_count_median=("news_count", "median"),
        news_count_mean=("news_count", "mean"),
        news_count_p90=("news_count", lambda x: x.quantile(0.90)),
        news_count_max=("news_count", "max"),
        no_news_rate=("no_news_dummy", "mean"),
        canonical_source_occurrence_sum=("canonical_source_count", "sum"),
    )
    yearly = daily.resample("YS").agg(
        news_count_median=("news_count", "median"),
        news_count_mean=("news_count", "mean"),
        news_count_p90=("news_count", lambda x: x.quantile(0.90)),
        news_count_max=("news_count", "max"),
        no_news_rate=("no_news_dummy", "mean"),
        canonical_source_occurrence_sum=("canonical_source_count", "sum"),
    )
    monthly.to_csv(audit_dir / "news_monthly_stats.csv", index_label="date")
    yearly.to_csv(audit_dir / "news_yearly_stats.csv", index_label="date")
    source_rows = [
        {
            "date": article.timestamp.normalize(),
            "month": article.timestamp.tz_localize(None)
            .to_period("M")
            .strftime("%Y-%m"),
            "year": article.timestamp.year,
            "canonical_source": article.source,
        }
        for article in articles
    ]
    if source_rows:
        sources = pd.DataFrame(source_rows)
        for grouping, filename in [
            (["month", "canonical_source"], "canonical_source_monthly.csv"),
            (["year", "canonical_source"], "canonical_source_yearly.csv"),
        ]:
            table = sources.groupby(grouping).size().rename("count").reset_index()
            base = table.groupby(grouping[0])["count"].transform("sum")
            table["share"] = table["count"] / base
            table.to_csv(audit_dir / filename, index=False)
        distinct_month = (
            sources.groupby("month")["canonical_source"]
            .nunique()
            .rename("distinct_canonical_sources")
        )
        distinct_year = (
            sources.groupby("year")["canonical_source"]
            .nunique()
            .rename("distinct_canonical_sources")
        )
        distinct_month.to_csv(
            audit_dir / "distinct_canonical_sources_monthly.csv"
        )
        distinct_year.to_csv(
            audit_dir / "distinct_canonical_sources_yearly.csv"
        )
    semantic_days = frame[frame["semantic"].notna()][["semantic"]].copy()
    drift_rows = []
    previous: np.ndarray | None = None
    month_keys = semantic_days.index.tz_localize(None).to_period("M")
    for month, group in semantic_days.groupby(month_keys):
        centroid = np.mean(np.stack(group["semantic"].to_list()), axis=0)
        drift = np.nan
        if previous is not None:
            drift = 1.0 - float(
                np.dot(previous, centroid)
                / max(np.linalg.norm(previous) * np.linalg.norm(centroid), 1e-12)
            )
        drift_rows.append({"month": str(month), "cosine_drift": drift})
        previous = centroid
    pd.DataFrame(drift_rows).to_csv(
        audit_dir / "monthly_semantic_centroid_drift.csv", index=False
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].plot(frame.index, frame["news_count"], linewidth=0.6)
    axes[0].set_title("Relevant Bitcoin canonical clusters per UTC day")
    axes[1].plot(
        frame["news_count"].resample("MS").mean().index,
        frame["news_count"].resample("MS").mean(),
    )
    axes[1].set_title("Mean relevant clusters per month")
    fig.savefig(audit_dir / "news_count_daily_monthly.png", dpi=150)
    plt.close(fig)
    write_json(audit_dir / "news_filter_timestamp_audit.json", filter_audit)
