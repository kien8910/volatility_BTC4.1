from dataclasses import replace

import numpy as np
import pandas as pd

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.news import (
    FilteredArticle,
    aggregate_daily_news,
    conservative_deduplicate_articles,
    relevance_evidence,
)
from btc_main_pilot.news_representation_audit import (
    NEWS_PROBE_NAMES,
    _feature_names,
    _fit_probe,
)


def _article(
    cluster: str,
    timestamp: str,
    title: str,
    text: str,
    source: str,
    relevance: int = 2,
) -> FilteredArticle:
    return FilteredArticle(
        cluster_id=cluster,
        timestamp=pd.Timestamp(timestamp),
        title=title,
        cleaned_text=text,
        source=source,
        relevance=relevance,
    )


def test_news_representation_profile_and_probe_dimensions_are_locked():
    config = replace(
        MainPilotConfig(),
        profile="development-news-representation-audit",
    )
    config.validate()
    expected = {
        "news_scalars_11": 11,
        "finbert_slow_fast_6": 6,
        "bge_pca8_slow_fast_16": 16,
        "bge_pca16_slow_fast_32": 32,
        "surprise_norms_2": 2,
        "combined_pca8_33": 33,
        "source_balanced_dedup_pca8_33": 33,
        "combined_pca16_surprise_51": 51,
    }
    assert tuple(expected) == NEWS_PROBE_NAMES
    for name, dimension in expected.items():
        assert len(_feature_names(name)) == dimension
        assert len(set(_feature_names(name))) == dimension


def test_relevance_evidence_distinguishes_title_from_body_only():
    assert (
        relevance_evidence("Bitcoin ETF approved", "Market update")
        == "title_primary"
    )
    assert (
        relevance_evidence(
            "Dollar market update",
            "Bitcoin moved sharply while bitcoin liquidity fell.",
        )
        == "content_repeated"
    )


def test_conservative_dedup_removes_encoding_twins_and_exact_content():
    articles = [
        _article(
            "a",
            "2022-01-01 01:00:00+00:00",
            "Bitcoin – rises",
            "first unique content",
            "A",
        ),
        _article(
            "b",
            "2022-01-01 01:00:00+00:00",
            "Bitcoin - rises",
            "second unique content",
            "A",
        ),
        _article(
            "c",
            "2022-01-01 02:00:00+00:00",
            "Different title",
            "first unique content",
            "B",
        ),
        _article(
            "d",
            "2022-01-01 03:00:00+00:00",
            "Independent event",
            "third unique content",
            "B",
        ),
    ]
    indices, audit = conservative_deduplicate_articles(articles)
    assert indices == [0, 3]
    assert audit["removed_same_timestamp_normalized_title"] == 1
    assert audit["removed_same_day_exact_cleaned_content"] == 1


def test_source_balanced_daily_centroid_prevents_article_count_dominance():
    articles = [
        _article("a", "2022-01-01 01:00:00+00:00", "A1", "x", "A"),
        _article("b", "2022-01-01 02:00:00+00:00", "A2", "y", "A"),
        _article("c", "2022-01-01 03:00:00+00:00", "B1", "z", "B"),
    ]
    semantics = np.zeros((3, 768), dtype=np.float32)
    semantics[:, 0] = [0.0, 2.0, 10.0]
    sentiments = np.asarray(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.8, 0.1, 0.1]],
        dtype=np.float32,
    )
    current = aggregate_daily_news(
        articles, semantics, sentiments, "2022-01-01", "2022-01-01"
    )
    balanced = aggregate_daily_news(
        articles,
        semantics,
        sentiments,
        "2022-01-01",
        "2022-01-01",
        source_balanced=True,
    )
    assert np.isclose(current.iloc[0]["semantic"][0], 4.0)
    assert np.isclose(balanced.iloc[0]["semantic"][0], 5.5)


def test_har_anchored_probe_keeps_zero_correction_without_validation_gain():
    rng = np.random.default_rng(42)
    x_core = rng.normal(size=(40, 2))
    x_validation = rng.normal(size=(12, 2))
    x_test = rng.normal(size=(10, 2))
    core_anchor = rng.normal(-7.0, 0.2, size=40)
    validation_anchor = rng.normal(-7.0, 0.2, size=12)
    test_anchor = rng.normal(-7.0, 0.2, size=10)
    result = _fit_probe(
        "test_probe",
        x_core,
        x_validation,
        x_test,
        np.exp(core_anchor),
        np.exp(validation_anchor),
        np.exp(test_anchor),
        core_anchor,
        validation_anchor,
        test_anchor,
        ["x1", "x2"],
        (1.0,),
        1e-5,
    )
    validation, test, metadata, grid = result
    assert not metadata["correction_selected"]
    np.testing.assert_allclose(validation["delta_log_rv"], 0.0)
    np.testing.assert_allclose(test["delta_log_rv"], 0.0)
    assert bool(grid.iloc[0]["selection_eligible"])
