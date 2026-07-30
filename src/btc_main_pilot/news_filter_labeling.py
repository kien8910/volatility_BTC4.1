from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from .news import load_filtered_articles
from .utils import stable_hash, utc_now, write_json


PROMPT_VERSION = "btc-news-relevance-v1"
SILVER_PASS2_PROMPT_VERSION = "btc-news-silver-blind-v1"
SILVER_ADJUDICATION_PROMPT_VERSION = "btc-news-silver-blind-adjudication-v1"
LABEL_COLUMNS = (
    "gpt_relevance_label",
    "gpt_forecast_relevance",
    "gpt_event_types",
    "gpt_is_price_recap",
    "gpt_confidence",
    "gpt_reason",
    "gpt_model",
    "gpt_prompt_version",
    "gpt_labeled_utc",
)
EVENT_TYPES = (
    "direct_bitcoin",
    "etf",
    "regulation",
    "exchange_custody",
    "security_hack",
    "macro_liquidity",
    "mining_energy",
    "stablecoin_defi",
    "institutional_adoption",
    "price_recap_technical_analysis",
    "other_crypto_context",
    "none",
)


class NewsLabel(BaseModel):
    news_cluster_id: str
    relevance_label: Literal["relevant", "irrelevant", "uncertain"]
    forecast_relevance: Literal[
        "direct", "contextual", "weak", "none", "uncertain"
    ]
    event_types: list[
        Literal[
            "direct_bitcoin",
            "etf",
            "regulation",
            "exchange_custody",
            "security_hack",
            "macro_liquidity",
            "mining_energy",
            "stablecoin_defi",
            "institutional_adoption",
            "price_recap_technical_analysis",
            "other_crypto_context",
            "none",
        ]
    ]
    is_price_recap: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)


class NewsLabelBatch(BaseModel):
    labels: list[NewsLabel]


def _event_family_hint(title: str, lead: str) -> str:
    text = f" {title} {lead} ".lower()
    patterns = (
        (
            "security_hack",
            (
                " hack",
                " hacked",
                " exploit",
                " breach",
                " stolen",
                " ransomware",
                " cyberattack",
            ),
        ),
        (
            "regulation_etf",
            (
                " sec ",
                " regulator",
                " regulation",
                " lawsuit",
                " court ",
                " legal ",
                " ban ",
                " etf ",
                " exchange-traded fund",
            ),
        ),
        (
            "exchange_custody",
            (
                " binance",
                " coinbase",
                " kraken",
                " bitfinex",
                " bitmex",
                " mt. gox",
                " mt gox",
                " exchange",
                " custody",
                " custodian",
            ),
        ),
        (
            "macro_liquidity",
            (
                " federal reserve",
                " fed ",
                " interest rate",
                " inflation",
                " cpi ",
                " monetary policy",
                " liquidity",
                " recession",
                " treasury",
                " dollar",
            ),
        ),
        (
            "mining_energy",
            (
                " miner",
                " mining",
                " hashrate",
                " hash rate",
                " energy",
                " electricity",
                " halving",
            ),
        ),
        (
            "stablecoin_defi",
            (
                " stablecoin",
                " tether",
                " usdt",
                " usdc",
                " defi ",
                " decentralized finance",
            ),
        ),
        (
            "price_recap_ta",
            (
                " price analysis",
                " technical analysis",
                " support level",
                " resistance level",
                " price prediction",
                " market recap",
            ),
        ),
        (
            "direct_bitcoin",
            (" bitcoin", " btc ", " xbt ", " satoshi", " lightning network"),
        ),
        (
            "other_crypto_context",
            (
                " crypto",
                " blockchain",
                " ethereum",
                " ether ",
                " digital asset",
                " cryptocurrency",
            ),
        ),
    )
    for family, needles in patterns:
        if any(needle in text for needle in needles):
            return family
    return "other"


def _stratum_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["decision"]),
        int(row["year"]),
        str(row["evidence_type"]),
        str(row["score_band"]),
    )


def _choose_decision_quotas(
    population_counts: dict[tuple[str, int, str, str], int],
    target_size: int,
    minimum_per_cell: int = 4,
    maximum_per_cell: int = 100,
) -> dict[str, int]:
    if target_size < 1:
        raise ValueError("target_size must be positive")
    decisions = ("retained", "removed")
    best: tuple[tuple[int, int, int], dict[str, int]] | None = None
    for retained_quota in range(minimum_per_cell, maximum_per_cell + 1):
        for removed_quota in range(minimum_per_cell, maximum_per_cell + 1):
            quotas = {
                "retained": retained_quota,
                "removed": removed_quota,
            }
            totals = {
                decision: sum(
                    min(count, quotas[decision])
                    for key, count in population_counts.items()
                    if key[0] == decision
                )
                for decision in decisions
            }
            total = sum(totals.values())
            objective = (
                abs(total - target_size),
                abs(totals["retained"] - totals["removed"]),
                retained_quota + removed_quota,
            )
            if best is None or objective < best[0]:
                best = (objective, quotas)
    if best is None:
        raise RuntimeError("Could not choose audit quotas")
    return best[1]


def _population_counts(audit: dict[str, Any]) -> dict[
    tuple[str, int, str, str], int
]:
    counts: dict[tuple[str, int, str, str], int] = {}
    for item in audit.get("stratified_review_strata", []):
        key = (
            str(item["decision"]),
            int(item["year"]),
            str(item["evidence_type"]),
            str(item["score_band"]),
        )
        counts[key] = int(item["population_n"])
    if not counts:
        raise RuntimeError(
            "News audit did not return stratum population counts"
        )
    return counts


def prepare_expanded_review(
    news_path: Path,
    review_path: Path,
    start: str,
    end: str,
    target_size: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    original = (
        pd.read_csv(review_path, dtype=str, keep_default_na=False)
        if review_path.exists()
        else pd.DataFrame()
    )
    backup_path = review_path.with_name(
        "stratified_news_filter_review_original_366.csv"
    )
    if review_path.exists() and not backup_path.exists():
        shutil.copy2(review_path, backup_path)
    baseline = (
        pd.read_csv(backup_path, dtype=str, keep_default_na=False)
        if backup_path.exists()
        else original
    )
    original_ids = set(
        baseline.get("news_cluster_id", pd.Series(dtype=str))
    )

    logger.info(
        "GPT AUDIT SAMPLE | scanning development news; target=%d",
        target_size,
    )
    _, audit = load_filtered_articles(
        news_path,
        start,
        end,
        logger,
        stratified_review_per_cell=100,
    )
    population = _population_counts(audit)
    quotas = _choose_decision_quotas(population, target_size)
    candidates = audit["stratified_review_examples"]
    selected: list[dict[str, Any]] = []
    per_stratum_selected: Counter[tuple[str, int, str, str]] = Counter()
    for row in candidates:
        key = _stratum_key(row)
        if per_stratum_selected[key] >= quotas[key[0]]:
            continue
        item = dict(row)
        item["event_family_hint"] = _event_family_hint(
            str(item["canonical_title"]), str(item["cleaned_lead"])
        )
        item["sample_source"] = (
            "original_stratified_366"
            if str(item["news_cluster_id"]) in original_ids
            else "expanded_balanced_stratified"
        )
        selected.append(item)
        per_stratum_selected[key] += 1

    frame = pd.DataFrame(selected)
    sample_counts = Counter(_stratum_key(row) for row in selected)
    frame["stratum_population_n"] = [
        population[_stratum_key(row)] for row in selected
    ]
    frame["stratum_sample_n"] = [
        sample_counts[_stratum_key(row)] for row in selected
    ]
    frame["sampling_weight"] = (
        frame["stratum_population_n"].astype(float)
        / frame["stratum_sample_n"].astype(float)
    )
    for column in ("manual_relevant", "review_notes", "labels", *LABEL_COLUMNS):
        if column not in frame:
            frame[column] = ""

    if not original.empty:
        old_by_id = original.set_index("news_cluster_id", drop=False)
        for index, cluster_id in frame["news_cluster_id"].items():
            if cluster_id not in old_by_id.index:
                continue
            old = old_by_id.loc[cluster_id]
            if isinstance(old, pd.DataFrame):
                old = old.iloc[0]
            for column in (
                "manual_relevant",
                "review_notes",
                "labels",
                *LABEL_COLUMNS,
            ):
                if column in old and str(old[column]).strip():
                    frame.at[index, column] = old[column]

    sort_columns = [
        "decision",
        "year",
        "evidence_type",
        "score_band",
        "news_cluster_id",
    ]
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    temporary = review_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(review_path)
    summary = {
        "target_size": target_size,
        "actual_size": int(len(frame)),
        "original_rows_preserved": int(
            frame["news_cluster_id"].isin(original_ids).sum()
        ),
        "decision_quotas_per_stratum": quotas,
        "decision_counts": {
            str(key): int(value)
            for key, value in frame["decision"].value_counts().items()
        },
        "year_counts": {
            str(key): int(value)
            for key, value in frame["year"].value_counts().sort_index().items()
        },
        "event_family_counts": {
            str(key): int(value)
            for key, value in frame["event_family_hint"]
            .value_counts()
            .items()
        },
        "selection": (
            "Deterministic hash-ranked sampling within "
            "decision×year×evidence_type×score_band strata. Decision-specific "
            "per-stratum quotas are selected to approach the requested size "
            "and retained/removed balance. No market outcome is used."
        ),
        "backup_path": str(backup_path) if backup_path.exists() else None,
        "created_utc": utc_now(),
    }
    write_json(review_path.with_name("expanded_review_sampling.json"), summary)
    logger.info(
        "GPT AUDIT SAMPLE READY | rows=%d retained=%d removed=%d "
        "original_preserved=%d",
        len(frame),
        summary["decision_counts"].get("retained", 0),
        summary["decision_counts"].get("removed", 0),
        summary["original_rows_preserved"],
    )
    return frame, summary


def _decrypt_dpapi_key(path: Path) -> str:
    if os.name != "nt":
        raise RuntimeError(
            "DPAPI credential files are supported only on Windows; set "
            "OPENAI_API_KEY instead."
        )
    if not path.exists():
        raise FileNotFoundError(f"DPAPI credential file not found: {path}")
    script = (
        "$cipher = (Get-Content -LiteralPath "
        "$env:BTC_NEWS_AUDIT_DPAPI_PATH -Raw).Trim(); "
        "$value = ConvertTo-SecureString -String $cipher "
        "-ErrorAction Stop; "
        "[System.Net.NetworkCredential]::new('', $value).Password"
    )
    child_environment = os.environ.copy()
    child_environment["BTC_NEWS_AUDIT_DPAPI_PATH"] = str(path)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    key = completed.stdout.strip()
    if not key:
        raise RuntimeError("DPAPI credential decrypted to an empty value")
    return key


def _load_api_key(dpapi_path: Path | None) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if dpapi_path is None:
        raise RuntimeError(
            "OPENAI_API_KEY is unset and no DPAPI credential path was provided"
        )
    return _decrypt_dpapi_key(dpapi_path)


def _cache_key(row: pd.Series, model: str) -> str:
    return stable_hash(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "news_cluster_id": str(row["news_cluster_id"]),
            "title": str(row["canonical_title"]),
            "lead": str(row["cleaned_lead"]),
            "source": str(row["canonical_source"]),
            "timestamp": str(row["canonical_publication_time"]),
        }
    )


def _load_label_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cached[str(item["cache_key"])] = item
    return cached


def _append_cache(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _label_prompt(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    return f"""
Label every supplied news record independently using only information that was
available in its title and lead at publication time. Do not infer later market
outcomes and do not use realized volatility, future prices, or spike labels.

Goal: audit whether a high-recall Bitcoin-news filter should keep the record for
next-day Bitcoin volatility research.

relevance_label:
- relevant: directly about Bitcoin, or a concrete crypto-system, regulatory,
  macro-liquidity, security, exchange, mining, stablecoin, ETF, or institutional
  event with a plausible transmission channel to Bitcoin volatility.
- irrelevant: unrelated, merely contains a boilerplate/tangential mention,
  Bitcoin Cash only, or lacks a plausible Bitcoin transmission channel.
- uncertain: the supplied title/lead is insufficient or genuinely ambiguous.

forecast_relevance:
- direct: concrete Bitcoin-specific information.
- contextual: a concrete event with a plausible Bitcoin transmission channel.
- weak: topically related but mostly generic commentary, evergreen education,
  promotion, price recap, or technical analysis without a new event.
- none: no meaningful Bitcoin-volatility relevance.
- uncertain: insufficient evidence.

Use one or more event_types from: {", ".join(EVENT_TYPES)}. Use "none" alone
when no event type applies. Mark is_price_recap true when the item mainly
describes already-observed price movements, predictions, support/resistance, or
technical analysis without a new causal event. Give a concise evidence-based
reason, not investment advice. Return each news_cluster_id exactly once.

Records:
{payload}
""".strip()


def _validate_labels(
    labels: list[NewsLabel], expected_ids: list[str]
) -> None:
    returned = [label.news_cluster_id for label in labels]
    if len(returned) != len(set(returned)):
        raise ValueError("Model returned duplicate news_cluster_id values")
    if set(returned) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(returned))
        extra = sorted(set(returned) - set(expected_ids))
        raise ValueError(
            f"Model ID mismatch; missing={missing[:5]} extra={extra[:5]}"
        )
    for label in labels:
        if "none" in label.event_types and len(label.event_types) != 1:
            raise ValueError("'none' event type must be used alone")


def _call_label_batch(
    client: Any,
    rows: list[dict[str, str]],
    model: str,
    max_retries: int,
) -> tuple[list[NewsLabel], dict[str, int]]:
    expected_ids = [row["news_cluster_id"] for row in rows]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.parse(
                model=model,
                instructions=(
                    "You are a conservative financial-news annotation system. "
                    "Follow the supplied rubric and schema exactly."
                ),
                input=_label_prompt(rows),
                text_format=NewsLabelBatch,
                reasoning={"effort": "low"},
                store=False,
                max_output_tokens=max(3000, 260 * len(rows)),
                timeout=120.0,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("OpenAI response did not contain parsed output")
            _validate_labels(parsed.labels, expected_ids)
            usage = getattr(response, "usage", None)
            return parsed.labels, {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"OpenAI labeling failed after {max_retries + 1} attempts"
    ) from last_error


def _silver_cache_key(
    row: pd.Series, model: str, prompt_version: str
) -> str:
    return stable_hash(
        {
            "prompt_version": prompt_version,
            "model": model,
            "news_cluster_id": str(row["news_cluster_id"]),
            "title": str(row["canonical_title"]),
            "lead": str(row["cleaned_lead"]),
            "source": str(row["canonical_source"]),
            "timestamp": str(row["canonical_publication_time"]),
        }
    )


def _run_silver_pass(
    frame: pd.DataFrame,
    model: str,
    prompt_version: str,
    cache_path: Path,
    batch_size: int,
    client: Any,
    logger: logging.Logger,
    max_retries: int,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    cache = _load_label_cache(cache_path)
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, str]] = []
    for index, row in frame.iterrows():
        key = _silver_cache_key(row, model, prompt_version)
        if key in cache:
            results[str(row["news_cluster_id"])] = cache[key]
        else:
            pending.append((index, key))
    logger.info(
        "GPT SILVER CACHE | pass=%s hits=%d misses=%d model=%s",
        prompt_version,
        len(frame) - len(pending),
        len(pending),
        model,
    )
    usage: Counter[str] = Counter()
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        request_rows = [
            {
                "news_cluster_id": str(frame.at[index, "news_cluster_id"]),
                "publication_time": str(
                    frame.at[index, "canonical_publication_time"]
                ),
                "source": str(frame.at[index, "canonical_source"]),
                "title": str(frame.at[index, "canonical_title"]),
                "lead": str(frame.at[index, "cleaned_lead"]),
            }
            for index, _ in batch
        ]
        labels, batch_usage = _call_label_batch(
            client, request_rows, model, max_retries
        )
        usage.update(batch_usage)
        labels_by_id = {label.news_cluster_id: label for label in labels}
        cache_items: list[dict[str, Any]] = []
        for index, key in batch:
            cluster_id = str(frame.at[index, "news_cluster_id"])
            item = {
                "cache_key": key,
                "model": model,
                "prompt_version": prompt_version,
                "labeled_utc": utc_now(),
                "label": labels_by_id[cluster_id].model_dump(),
            }
            cache_items.append(item)
            results[cluster_id] = item
        _append_cache(cache_path, cache_items)
        completed = min(offset + len(batch), len(pending))
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed / elapsed
        eta = (len(pending) - completed) / max(rate, 1e-9)
        logger.info(
            "GPT SILVER | pass=%s %d/%d (%.1f%%) ETA=%.1f min",
            prompt_version,
            completed,
            len(pending),
            100.0 * completed / max(len(pending), 1),
            eta / 60.0,
        )
    return results, usage


def run_gpt_silver_holdout(
    expanded_review_path: Path,
    original_review_path: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    dpapi_path: Path | None,
    logger: logging.Logger,
    adjudicate: bool = True,
    max_retries: int = 3,
) -> dict[str, Any]:
    from openai import OpenAI
    from sklearn.metrics import cohen_kappa_score

    if batch_size < 1:
        raise ValueError("Silver-label batch_size must be positive")
    expanded = pd.read_csv(
        expanded_review_path, dtype=str, keep_default_na=False
    )
    original = pd.read_csv(
        original_review_path, dtype=str, keep_default_na=False
    )
    if len(original) != 366:
        raise ValueError(
            f"Locked silver holdout must contain exactly 366 rows, got "
            f"{len(original)}"
        )
    if original["news_cluster_id"].nunique() != len(original):
        raise ValueError("Original 366 holdout contains duplicate IDs")
    pass1_columns = {
        "gpt_relevance_label",
        "gpt_forecast_relevance",
        "gpt_event_types",
        "gpt_is_price_recap",
        "gpt_confidence",
        "gpt_reason",
    }
    if not pass1_columns.issubset(expanded.columns):
        raise RuntimeError("Expanded review is missing first-pass GPT labels")
    expanded_by_id = expanded.set_index("news_cluster_id", drop=False)
    missing = sorted(
        set(original["news_cluster_id"]) - set(expanded_by_id.index)
    )
    if missing:
        raise RuntimeError(
            f"Expanded review is missing {len(missing)} holdout IDs"
        )
    holdout = expanded_by_id.loc[
        original["news_cluster_id"].tolist()
    ].reset_index(drop=True)
    if (holdout["gpt_relevance_label"].str.len() == 0).any():
        raise RuntimeError("First-pass GPT labels are incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = _load_api_key(dpapi_path)
    try:
        client = OpenAI(api_key=api_key, max_retries=0)
    finally:
        api_key = ""
    pass2, usage2 = _run_silver_pass(
        holdout,
        model,
        SILVER_PASS2_PROMPT_VERSION,
        output_dir / "gpt_silver_pass2_cache.jsonl",
        batch_size,
        client,
        logger,
        max_retries,
    )
    pass1_labels = holdout["gpt_relevance_label"].tolist()
    pass2_labels = [
        pass2[str(row.news_cluster_id)]["label"]["relevance_label"]
        for row in holdout.itertuples(index=False)
    ]
    disagreements = [
        index
        for index, (first, second) in enumerate(
            zip(pass1_labels, pass2_labels, strict=True)
        )
        if first != second
    ]
    pass3: dict[str, dict[str, Any]] = {}
    usage3: Counter[str] = Counter()
    if adjudicate and disagreements:
        pass3, usage3 = _run_silver_pass(
            holdout.iloc[disagreements].reset_index(drop=True),
            model,
            SILVER_ADJUDICATION_PROMPT_VERSION,
            output_dir / "gpt_silver_adjudication_cache.jsonl",
            batch_size,
            client,
            logger,
            max_retries,
        )

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for row in holdout.to_dict(orient="records"):
        cluster_id = str(row["news_cluster_id"])
        first = str(row["gpt_relevance_label"])
        second_item = pass2[cluster_id]
        second = str(second_item["label"]["relevance_label"])
        votes = [first, second]
        third_item = pass3.get(cluster_id)
        if third_item is not None:
            votes.append(str(third_item["label"]["relevance_label"]))
        counts = Counter(votes)
        label, count = counts.most_common(1)[0]
        if count < 2:
            label = "uncertain"
            unresolved += 1
        selected_item = second_item
        if (
            selected_item["label"]["relevance_label"] != label
            and third_item is not None
            and third_item["label"]["relevance_label"] == label
        ):
            selected_item = third_item
        output = dict(row)
        output.update(
            {
                "silver_pass1_model": str(row["gpt_model"]),
                "silver_pass1_label": first,
                "silver_pass1_reason": str(row["gpt_reason"]),
                "silver_pass2_model": model,
                "silver_pass2_label": second,
                "silver_pass2_reason": second_item["label"]["reason"],
                "silver_pass3_label": (
                    third_item["label"]["relevance_label"]
                    if third_item is not None
                    else ""
                ),
                "silver_pass3_reason": (
                    third_item["label"]["reason"]
                    if third_item is not None
                    else ""
                ),
                "silver_relevance_label": label,
                "silver_forecast_relevance": (
                    selected_item["label"]["forecast_relevance"]
                    if label != "uncertain"
                    else "uncertain"
                ),
                "silver_event_types": "|".join(
                    selected_item["label"]["event_types"]
                ),
                "silver_is_price_recap": bool(
                    selected_item["label"]["is_price_recap"]
                ),
                "silver_confidence": float(
                    selected_item["label"]["confidence"]
                ),
                "silver_reason": selected_item["label"]["reason"],
            }
        )
        rows.append(output)
    silver = pd.DataFrame(rows)
    silver_path = output_dir / "gpt_silver_holdout_366.csv"
    silver.to_csv(silver_path, index=False)
    agreement = float(
        sum(a == b for a, b in zip(pass1_labels, pass2_labels, strict=True))
        / len(holdout)
    )
    kappa_value = float(cohen_kappa_score(pass1_labels, pass2_labels))
    kappa: float | None = kappa_value if math.isfinite(kappa_value) else None
    usage = usage2 + usage3
    report = {
        "status": "completed",
        "scope": (
            "Locked original 366-row holdout; prompts receive title, lead, "
            "source and publication time only."
        ),
        "holdout_n": int(len(holdout)),
        "pass2_model": model,
        "adjudication_enabled": adjudicate,
        "pass1_pass2_agreement": agreement,
        "pass1_pass2_cohen_kappa": kappa,
        "pass1_pass2_disagreement_n": len(disagreements),
        "unresolved_n": unresolved,
        "silver_label_counts": {
            str(key): int(value)
            for key, value in silver["silver_relevance_label"]
            .value_counts()
            .items()
        },
        "usage": dict(usage),
        "output_path": str(silver_path),
        "statistical_claim": (
            "None. These are GPT-silver labels, not expert ground truth."
        ),
        "completed_utc": utc_now(),
    }
    write_json(output_dir / "gpt_silver_holdout_report.json", report)
    logger.info(
        "GPT SILVER COMPLETED | n=%d agreement=%.4f kappa=%s "
        "disagreements=%d unresolved=%d",
        len(holdout),
        agreement,
        f"{kappa:.4f}" if kappa is not None else "undefined",
        len(disagreements),
        unresolved,
    )
    return report


def _apply_cached_label(
    frame: pd.DataFrame,
    index: int,
    item: dict[str, Any],
) -> None:
    label = item["label"]
    frame.at[index, "gpt_relevance_label"] = label["relevance_label"]
    frame.at[index, "gpt_forecast_relevance"] = label["forecast_relevance"]
    frame.at[index, "gpt_event_types"] = "|".join(label["event_types"])
    frame.at[index, "gpt_is_price_recap"] = bool(label["is_price_recap"])
    frame.at[index, "gpt_confidence"] = float(label["confidence"])
    frame.at[index, "gpt_reason"] = label["reason"]
    frame.at[index, "gpt_model"] = item["model"]
    frame.at[index, "gpt_prompt_version"] = item["prompt_version"]
    frame.at[index, "gpt_labeled_utc"] = item["labeled_utc"]


def _weighted_confusion(frame: pd.DataFrame) -> dict[str, Any]:
    labeled = frame[
        frame["gpt_relevance_label"].isin(["relevant", "irrelevant"])
    ].copy()
    labeled["sampling_weight"] = pd.to_numeric(
        labeled["sampling_weight"], errors="raise"
    )
    cells: dict[str, float] = {}
    for decision in ("retained", "removed"):
        for label in ("relevant", "irrelevant"):
            mask = (
                (labeled["decision"] == decision)
                & (labeled["gpt_relevance_label"] == label)
            )
            cells[f"{decision}_{label}"] = float(
                labeled.loc[mask, "sampling_weight"].sum()
            )
    tp = cells["retained_relevant"]
    fp = cells["retained_irrelevant"]
    fn = cells["removed_relevant"]
    tn = cells["removed_irrelevant"]
    return {
        "weighted_cells": cells,
        "precision_proxy": tp / (tp + fp) if tp + fp else None,
        "recall_proxy": tp / (tp + fn) if tp + fn else None,
        "specificity_proxy": tn / (tn + fp) if tn + fp else None,
        "warning": (
            "GPT labels are weak labels, not human ground truth. Weighted "
            "estimates use stratum population/sample weights."
        ),
    }


def _write_label_report(
    frame: pd.DataFrame,
    output_dir: Path,
    sampling_summary: dict[str, Any],
    model: str,
    usage: Counter[str],
) -> dict[str, Any]:
    labeled = frame[frame["gpt_relevance_label"].astype(str).str.len() > 0]
    disagreements = labeled[
        (
            (labeled["decision"] == "retained")
            & (labeled["gpt_relevance_label"] == "irrelevant")
        )
        | (
            (labeled["decision"] == "removed")
            & (labeled["gpt_relevance_label"] == "relevant")
        )
    ]
    uncertain = labeled[labeled["gpt_relevance_label"] == "uncertain"]
    disagreements.to_csv(
        output_dir / "gpt_filter_disagreements.csv", index=False
    )
    uncertain.to_csv(output_dir / "gpt_filter_uncertain.csv", index=False)
    cross = pd.crosstab(
        labeled["decision"],
        labeled["gpt_relevance_label"],
        dropna=False,
    )
    cross.to_csv(output_dir / "gpt_filter_confusion_unweighted.csv")
    by_year = pd.crosstab(
        labeled["year"],
        labeled["gpt_relevance_label"],
        dropna=False,
    )
    by_year.to_csv(output_dir / "gpt_filter_labels_by_year.csv")
    report = {
        "status": "completed" if len(labeled) == len(frame) else "partial",
        "scope": "development-only news-filter audit; no market outcomes used",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "sample_n": int(len(frame)),
        "labeled_n": int(len(labeled)),
        "label_counts": {
            str(key): int(value)
            for key, value in labeled["gpt_relevance_label"]
            .value_counts()
            .items()
        },
        "forecast_relevance_counts": {
            str(key): int(value)
            for key, value in labeled["gpt_forecast_relevance"]
            .value_counts()
            .items()
        },
        "disagreement_n": int(len(disagreements)),
        "uncertain_n": int(len(uncertain)),
        "weighted_filter_diagnostic": _weighted_confusion(frame),
        "sampling": sampling_summary,
        "usage": dict(usage),
        "statistical_claim": (
            "None. GPT annotations are weak labels for development audit and "
            "require human adjudication."
        ),
        "completed_utc": utc_now(),
    }
    write_json(output_dir / "gpt_news_filter_audit.json", report)
    return report


def run_gpt_news_filter_audit(
    news_path: Path,
    review_path: Path,
    start: str,
    end: str,
    target_size: int,
    model: str,
    batch_size: int,
    dpapi_path: Path | None,
    logger: logging.Logger,
    prepare_only: bool = False,
    max_retries: int = 3,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sampling_path = review_path.with_name("expanded_review_sampling.json")
    reusable = False
    if review_path.exists() and sampling_path.exists():
        candidate = pd.read_csv(
            review_path, dtype=str, keep_default_na=False
        )
        sampling_candidate = json.loads(
            sampling_path.read_text(encoding="utf-8")
        )
        required = {
            "news_cluster_id",
            "stratum_population_n",
            "stratum_sample_n",
            "sampling_weight",
            *LABEL_COLUMNS,
        }
        reusable = (
            len(candidate) == target_size
            and candidate["news_cluster_id"].nunique() == target_size
            and required.issubset(candidate.columns)
            and int(sampling_candidate.get("actual_size", -1)) == target_size
        )
        if reusable:
            frame = candidate
            sampling = sampling_candidate
            logger.info(
                "GPT AUDIT SAMPLE REUSE | rows=%d path=%s",
                len(frame),
                review_path,
            )
    if not reusable:
        frame, sampling = prepare_expanded_review(
            news_path,
            review_path,
            start,
            end,
            target_size,
            logger,
        )
    if prepare_only:
        return {
            "status": "prepared",
            "review_path": str(review_path),
            "sampling": sampling,
        }

    from openai import OpenAI

    api_key = _load_api_key(dpapi_path)
    try:
        client = OpenAI(api_key=api_key, max_retries=0)
    finally:
        api_key = ""

    cache_path = review_path.with_name("gpt_news_filter_label_cache.jsonl")
    cache = _load_label_cache(cache_path)
    pending: list[tuple[int, str]] = []
    for index, row in frame.iterrows():
        key = _cache_key(row, model)
        cached = cache.get(key)
        if cached is not None:
            _apply_cached_label(frame, index, cached)
        else:
            pending.append((index, key))
    logger.info(
        "GPT LABEL CACHE | hits=%d misses=%d model=%s",
        len(frame) - len(pending),
        len(pending),
        model,
    )

    usage: Counter[str] = Counter()
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        request_rows = [
            {
                "news_cluster_id": str(frame.at[index, "news_cluster_id"]),
                "publication_time": str(
                    frame.at[index, "canonical_publication_time"]
                ),
                "source": str(frame.at[index, "canonical_source"]),
                "title": str(frame.at[index, "canonical_title"]),
                "lead": str(frame.at[index, "cleaned_lead"]),
            }
            for index, _ in batch
        ]
        labels, batch_usage = _call_label_batch(
            client, request_rows, model, max_retries
        )
        usage.update(batch_usage)
        by_id = {label.news_cluster_id: label for label in labels}
        cache_items: list[dict[str, Any]] = []
        for index, key in batch:
            cluster_id = str(frame.at[index, "news_cluster_id"])
            label = by_id[cluster_id]
            item = {
                "cache_key": key,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "labeled_utc": utc_now(),
                "label": label.model_dump(),
            }
            cache_items.append(item)
            _apply_cached_label(frame, index, item)
        _append_cache(cache_path, cache_items)
        frame.to_csv(review_path, index=False)
        completed = min(offset + len(batch), len(pending))
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (len(pending) - completed) / rate if rate > 0 else math.inf
        logger.info(
            "GPT LABEL | %d/%d (%.1f%%) batch=%d ETA=%.1f min "
            "tokens_in=%d tokens_out=%d",
            completed,
            len(pending),
            100.0 * completed / max(len(pending), 1),
            len(batch),
            eta / 60.0,
            usage["input_tokens"],
            usage["output_tokens"],
        )

    report = _write_label_report(
        frame,
        review_path.parent,
        sampling,
        model,
        usage,
    )
    logger.info(
        "GPT NEWS FILTER AUDIT COMPLETED | labeled=%d disagreements=%d "
        "uncertain=%d report=%s",
        report["labeled_n"],
        report["disagreement_n"],
        report["uncertain_n"],
        review_path.parent / "gpt_news_filter_audit.json",
    )
    return report
