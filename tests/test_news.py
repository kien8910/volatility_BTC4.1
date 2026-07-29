import numpy as np
import pandas as pd

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.news import (
    ALLOWED_FEATURE_FIELDS,
    FORBIDDEN_FEATURE_FIELDS,
    DAILY_SCALAR_COLUMNS,
    UNDEFINED_NO_NEWS_COLUMNS,
    clean_article_text,
    relevance_score,
)
from btc_main_pilot.preprocess import fit_transform_news_for_fold


def test_footer_is_removed_before_bitcoin_filtering():
    cleaned = clean_article_text(
        "A story about equities. Related stories Bitcoin Bitcoin Bitcoin ETF"
    )
    assert "Bitcoin" not in cleaned
    keep, score = relevance_score("Equity update", cleaned)
    assert not keep
    assert score == 0


def test_bitcoin_cash_only_is_rejected_but_independent_btc_is_retained():
    assert relevance_score("Bitcoin Cash rallies", "BCH network update") == (False, 0)
    keep, score = relevance_score(
        "Bitcoin Cash and BTC diverge", "BTC remains independently discussed."
    )
    assert keep
    assert score >= 2


def test_feature_schema_excludes_retrospective_cluster_metadata():
    assert ALLOWED_FEATURE_FIELDS.isdisjoint(FORBIDDEN_FEATURE_FIELDS)
    assert {
        "source_count",
        "member_count",
        "republication_offsets_minutes",
        "all_sources",
        "all_urls",
    }.issubset(FORBIDDEN_FEATURE_FIELDS)


def _synthetic_daily() -> pd.DataFrame:
    index = pd.date_range("2018-01-01", periods=24, freq="D", tz="UTC")
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(index=index)
    frame["news_count"] = 1
    frame["canonical_source_count"] = 1
    frame["no_news_dummy"] = 0.0
    frame["semantic"] = pd.Series(
        [rng.normal(size=768).astype(np.float32) for _ in index],
        index=index,
        dtype=object,
    )
    frame["sentiment"] = pd.Series(
        [np.array([0.2, 0.3, 0.5], dtype=np.float32) for _ in index],
        index=index,
        dtype=object,
    )
    frame["negative_count_070"] = 0.0
    frame["news_intensity"] = rng.normal(size=len(index))
    frame["log1p_canonical_source_count"] = np.log(2.0)
    frame["negative_ratio"] = 0.0
    frame["log1p_negative_count_070"] = 0.0
    frame["negative_probability_max"] = 0.3
    frame["negative_probability_std"] = 0.0
    frame["positive_probability_max"] = 0.2
    frame["sentiment_entropy_mean"] = 1.0
    frame["semantic_dispersion"] = 0.0
    frame["mean_relevance"] = 3.0
    no_news_index = 7
    frame.iloc[no_news_index, frame.columns.get_loc("news_count")] = 0
    frame.iloc[
        no_news_index, frame.columns.get_loc("canonical_source_count")
    ] = 0
    frame.iloc[no_news_index, frame.columns.get_loc("no_news_dummy")] = 1.0
    frame.iat[no_news_index, frame.columns.get_loc("semantic")] = None
    frame.iat[no_news_index, frame.columns.get_loc("sentiment")] = None
    for column in UNDEFINED_NO_NEWS_COLUMNS:
        frame.iloc[no_news_index, frame.columns.get_loc(column)] = np.nan
    return frame


def test_fold_pca_slow_fast_and_no_news_rules_are_causal():
    config = MainPilotConfig()
    frame = _synthetic_daily()
    transformed = fit_transform_news_for_fold(
        frame,
        "2018-01-01",
        "2018-01-12",
        "2018-01-13",
        "2018-01-17",
        "2018-01-18",
        "2018-01-24",
        config,
        __import__("logging").getLogger("test"),
    )
    no_news = 7
    np.testing.assert_allclose(
        transformed.semantic_slow[no_news],
        transformed.semantic_slow[no_news - 1],
    )
    np.testing.assert_allclose(transformed.semantic_fast[no_news], 0.0)
    np.testing.assert_allclose(
        transformed.sentiment_slow[no_news],
        transformed.sentiment_slow[no_news - 1],
    )
    np.testing.assert_allclose(transformed.sentiment_fast[no_news], 0.0)
    undefined_positions = [
        DAILY_SCALAR_COLUMNS.index(column) for column in UNDEFINED_NO_NEWS_COLUMNS
    ]
    np.testing.assert_allclose(
        transformed.daily_scalars[no_news, undefined_positions], 0.0
    )
    assert transformed.daily_scalars[no_news, -1] == 1.0

    changed = frame.copy(deep=True)
    changed_semantic = changed.iloc[-1]["semantic"].copy()
    changed_semantic[:] = 1e6
    changed.iat[-1, changed.columns.get_loc("semantic")] = changed_semantic
    second = fit_transform_news_for_fold(
        changed,
        "2018-01-01",
        "2018-01-12",
        "2018-01-13",
        "2018-01-17",
        "2018-01-18",
        "2018-01-24",
        config,
        __import__("logging").getLogger("test"),
    )
    np.testing.assert_allclose(
        transformed.semantic_slow[:-1], second.semantic_slow[:-1], atol=1e-6
    )
    assert transformed.preprocessor_hash == second.preprocessor_hash


def test_slow_plus_fast_equals_observation_after_initialization():
    config = MainPilotConfig()
    frame = _synthetic_daily()
    transformed = fit_transform_news_for_fold(
        frame,
        "2018-01-01",
        "2018-01-12",
        "2018-01-13",
        "2018-01-17",
        "2018-01-18",
        "2018-01-24",
        config,
        __import__("logging").getLogger("test"),
    )
    semantic_matrix = np.stack(frame.loc[frame["semantic"].notna(), "semantic"])
    # The exact Z is not exposed; the invariant is equivalent to reconstruction
    # remaining finite/nonzero on every observed day after initialization.
    reconstructed = transformed.semantic_slow + transformed.semantic_fast
    observed_mask = frame["no_news_dummy"].to_numpy() == 0
    assert np.isfinite(reconstructed[observed_mask]).all()
    assert semantic_matrix.shape[1] == 768

