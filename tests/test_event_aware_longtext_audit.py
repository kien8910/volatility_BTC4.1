from pathlib import Path

import numpy as np
import pandas as pd

from btc_main_pilot.event_aware_longtext_audit import (
    CONTEXT_FAMILIES,
    EventAwarePolicy,
    REPRESENTATION_NAMES,
    VariantVectorCache,
    _screen,
    _token_budgeted_text,
    _weighted_binary_metrics,
    event_aware_decision,
    fit_event_aware_policy,
)
from btc_main_pilot.news import DeterministicSmokeEncoder, FilteredArticle


def _policy(*families: str) -> EventAwarePolicy:
    return EventAwarePolicy(
        enabled_context_families=tuple(families),
        relevant_rate_threshold=0.5,
        minimum_family_examples=1,
        training_rows=10,
        holdout_rows=2,
        family_statistics=tuple(),
        label_source="test",
    )


def test_event_aware_filter_keeps_context_and_rejects_bch_recap():
    keep, score, reason, _ = event_aware_decision(
        "Major cryptocurrency exchange hacked",
        "The exchange suspended crypto withdrawals after a security breach.",
        _policy("security_hack"),
    )
    assert keep
    assert score == 2
    assert reason == "context_event:security_hack"

    keep, score, reason, recap = event_aware_decision(
        "Bitcoin Cash Price Analysis",
        "BCH tests its support and resistance levels.",
        _policy(*CONTEXT_FAMILIES),
    )
    assert not keep
    assert score == 0
    assert reason == "bitcoin_cash_price_recap"
    assert recap


def test_event_aware_policy_is_fit_without_locked_holdout(tmp_path: Path):
    review = pd.DataFrame(
        {
            "news_cluster_id": ["h", "a", "b", "c", "d"],
            "event_family_hint": [
                "security_hack",
                "security_hack",
                "security_hack",
                "macro_liquidity",
                "macro_liquidity",
            ],
            "gpt_relevance_label": [
                "irrelevant",
                "relevant",
                "relevant",
                "irrelevant",
                "irrelevant",
            ],
            "sampling_weight": [1, 1, 1, 1, 1],
        }
    )
    holdout = pd.DataFrame({"news_cluster_id": ["h"]})
    review_path = tmp_path / "review.csv"
    holdout_path = tmp_path / "holdout.csv"
    review.to_csv(review_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    policy = fit_event_aware_policy(
        review_path,
        holdout_path,
        relevant_rate_threshold=0.5,
        minimum_family_examples=2,
    )
    assert policy.training_rows == 4
    assert "security_hack" in policy.enabled_context_families
    assert "macro_liquidity" not in policy.enabled_context_families


def test_token_budgeted_smoke_text_respects_word_budget():
    article = FilteredArticle(
        cluster_id="a",
        timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
        title="Bitcoin event",
        cleaned_text=" ".join(f"word{i}" for i in range(100)),
        source="source",
        relevance=2,
    )
    text = _token_budgeted_text(
        article, tokenizer=None, max_tokens=20, include_title=True
    )
    assert len(text.split()) <= 20
    assert text.startswith("Title: Bitcoin event")


def test_variant_cache_roundtrip(tmp_path: Path):
    cache = VariantVectorCache(tmp_path / "cache.sqlite")
    try:
        value = np.arange(8, dtype=np.float32)
        cache.put("hash", "model", "variant", value)
        cache.commit()
        loaded = cache.get("hash", "model", "variant", 8)
        np.testing.assert_array_equal(loaded, value)
    finally:
        cache.close()


def test_weighted_filter_metrics():
    frame = pd.DataFrame(
        {
            "silver_relevance_label": [
                "relevant",
                "relevant",
                "irrelevant",
                "irrelevant",
            ],
            "prediction": [True, False, True, False],
            "holdout_sampling_weight": [2.0, 3.0, 1.0, 4.0],
        }
    )
    metrics = _weighted_binary_metrics(frame, "prediction")
    assert metrics["precision_proxy"] == 2.0 / 3.0
    assert metrics["recall_proxy"] == 2.0 / 5.0
    assert metrics["specificity_proxy"] == 4.0 / 5.0


def test_deterministic_encoder_split_methods_match_combined():
    encoder = DeterministicSmokeEncoder(embedding_dim=16)
    texts = ["one", "two"]
    semantic, sentiment = encoder.encode(texts)
    np.testing.assert_array_equal(semantic, encoder.encode_semantic(texts))
    np.testing.assert_array_equal(sentiment, encoder.encode_sentiment(texts))


def test_representation_screen_handles_a_smoke_block_without_spike_days():
    rows = [
        {
            "fold": "smoke_fold",
            "model": "har_qlike",
            "mean_qlike": 0.2,
            "normal_qlike": 0.2,
            "spike_qlike": None,
        }
    ]
    rows.extend(
        {
            "fold": "smoke_fold",
            "model": name,
            "mean_qlike": 0.19,
            "normal_qlike": 0.19,
            "spike_qlike": None,
        }
        for name in REPRESENTATION_NAMES
    )
    fold_metrics = pd.DataFrame(rows)
    pooled_metrics = fold_metrics.drop(columns="fold")
    screen = _screen(fold_metrics, pooled_metrics, min_delta=1e-5)
    assert set(screen["candidates"]) == set(REPRESENTATION_NAMES)
    assert all(
        candidate["pooled_spike_delta"] is None
        for candidate in screen["candidates"].values()
    )
