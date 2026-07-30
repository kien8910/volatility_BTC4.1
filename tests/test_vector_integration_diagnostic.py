import numpy as np
import pandas as pd
import torch

from btc_main_pilot.cli import build_parser
from btc_main_pilot.config import Fold, MainPilotConfig
from btc_main_pilot.news import FilteredArticle
from btc_main_pilot.vector_integration_diagnostic import (
    CANDIDATES,
    EVENT_FAMILIES,
    PROFILE,
    HarVectorCrossAttention,
    VectorAttentionDataset,
    _eligible_dates,
    build_core_event_prototype_features,
)


def _article(day: str, title: str, text: str, cluster: str) -> FilteredArticle:
    return FilteredArticle(
        cluster_id=cluster,
        timestamp=pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=1),
        title=title,
        cleaned_text=text,
        source=f"source-{cluster}",
        relevance=2,
    )


def test_vector_profile_and_candidates_are_registered():
    config = MainPilotConfig(profile=PROFILE)
    config.validate()
    args = build_parser().parse_args(["--profile", PROFILE, "--smoke"])
    assert args.profile == PROFILE
    assert CANDIDATES == (
        "har_qlike",
        "finbert_normal",
        "finbert_bge_directional",
        "finbert_event_prototypes",
        "transformer_cross_attention_slow",
        "transformer_cross_attention_fast",
    )


def test_event_prototypes_do_not_fit_on_validation_vectors():
    fold = Fold(
        "test",
        "2020-01-01",
        "2020-01-03",
        "2020-01-04",
        "2020-01-04",
        "2020-01-05",
        "2020-01-05",
    )
    dates = pd.date_range("2020-01-01", "2020-01-05", tz="UTC")
    articles = [
        _article(
            "2020-01-02",
            "Bitcoin ETF approval",
            "Regulators approved a spot Bitcoin fund.",
            "core",
        ),
        _article(
            "2020-01-04",
            "Bitcoin ETF filing",
            "A regulator reviewed a Bitcoin fund filing.",
            "validation",
        ),
    ]
    base = np.zeros((2, 8), dtype=np.float64)
    base[0, 0] = 1.0
    base[1, 1] = 1.0
    changed = base.copy()
    changed[1] = np.asarray([0, 0, 1, 0, 0, 0, 0, 0])
    frame_a, metadata_a = build_core_event_prototype_features(
        articles, base, dates, fold
    )
    frame_b, metadata_b = build_core_event_prototype_features(
        articles, changed, dates, fold
    )
    pd.testing.assert_frame_equal(
        frame_a.loc[:"2020-01-03"], frame_b.loc[:"2020-01-03"]
    )
    assert metadata_a["prototype_counts"] == metadata_b["prototype_counts"]
    assert metadata_a["target_outcomes_used"] is False
    assert set(metadata_a["families"]) == set(EVENT_FAMILIES)


def test_vector_dataset_uses_exactly_t_minus_30_through_t_minus_1():
    dates = pd.date_range("2020-01-01", periods=50, tz="UTC")
    target = pd.Timestamp("2020-02-10", tz="UTC")
    eligible = _eligible_dates([target], dates, 30)
    assert eligible == [target]
    daily = np.arange(50, dtype=np.float64)[:, None]
    dataset = VectorAttentionDataset(
        [target],
        np.zeros((1, 7)),
        daily,
        dates,
        np.zeros(1),
        np.ones(1),
        30,
    )
    expected = np.arange(10, 40, dtype=np.float32)
    np.testing.assert_array_equal(
        dataset[0]["news_tokens"][:, 0].numpy(), expected
    )
    assert dataset[0]["target_date"] == "2020-02-10"


def test_transformer_cross_attention_is_exact_har_at_initialization():
    config = MainPilotConfig(profile=PROFILE)
    for state in ("slow", "fast"):
        model = HarVectorCrossAttention(config, token_dim=40, state=state)
        model.eval()
        anchor = torch.tensor([0.25, -0.75], dtype=torch.float64)
        output = model(
            {
                "market_query": torch.randn(2, 7),
                "news_tokens": torch.randn(2, 30, 40),
                "har_anchor_log_rv": anchor,
            }
        )
        assert torch.equal(output["predicted_log_rv"], anchor)
        assert torch.count_nonzero(output["delta_log_rv"]) == 0
        assert model.news_blocks
        assert model.cross_blocks
