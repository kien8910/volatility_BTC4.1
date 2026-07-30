import numpy as np
import pandas as pd
import torch

from btc_main_pilot.cli import build_parser
from btc_main_pilot.config import Fold, MainPilotConfig
from btc_main_pilot.news import FilteredArticle
from btc_main_pilot.slow_transformer_v2_diagnostic import (
    BLEND_ALPHA_GRID,
    CANDIDATES,
    PROFILE,
    SlowUpdateDataset,
    SlowUpdateTransformer,
    build_core_centered_directional_features,
    select_log_blend,
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


def test_slow_v2_profile_and_candidate_set_are_registered():
    config = MainPilotConfig(profile=PROFILE)
    config.validate()
    args = build_parser().parse_args(["--profile", PROFILE, "--smoke"])
    assert args.profile == PROFILE
    assert CANDIDATES == (
        "har_qlike",
        "finbert_normal",
        "core_centered_event_prototypes",
        "slow_calendar_control",
        "slow_update_tokens",
        "slow_update_multiquery",
        "slow_update_multiquery_gated",
        "finbert_normal_slow_blend",
    )


def test_centered_directional_prototypes_ignore_validation_vectors():
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
            "2020-01-01",
            "Bitcoin ETF approval",
            "Regulators approved a spot Bitcoin fund.",
            "core-a",
        ),
        _article(
            "2020-01-02",
            "Bitcoin exchange withdrawals suspended",
            "A cryptocurrency exchange halted Bitcoin withdrawals.",
            "core-b",
        ),
        _article(
            "2020-01-04",
            "Bitcoin ETF filing",
            "A regulator reviewed a Bitcoin ETF filing.",
            "validation",
        ),
    ]
    semantics = np.asarray(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.2, 0.2, 0.0, 1.0],
        ]
    )
    changed = semantics.copy()
    changed[2] = np.asarray([10.0, -5.0, 3.0, 7.0])
    auxiliary = pd.DataFrame(index=dates)
    for family in (
        "direct_bitcoin",
        "regulation_etf",
        "exchange_custody",
        "macro_liquidity",
        "mining_energy",
        "security_hack",
    ):
        auxiliary[f"{family}__prototype_cosine"] = 0.9
        auxiliary[f"{family}__count_surprise_365d"] = 0.0
        auxiliary[f"{family}__log1p_source_count"] = 0.0
    frame_a, metadata_a = build_core_centered_directional_features(
        articles, semantics, dates, fold, auxiliary
    )
    frame_b, metadata_b = build_core_centered_directional_features(
        articles, changed, dates, fold, auxiliary
    )
    pd.testing.assert_frame_equal(
        frame_a.loc[:"2020-01-03"], frame_b.loc[:"2020-01-03"]
    )
    assert metadata_a["core_mean_hash"] == metadata_b["core_mean_hash"]
    assert metadata_a["target_outcomes_used"] is False
    assert all(
        "prototype_cosine" not in column for column in frame_a.columns
    )


def test_update_dataset_masks_no_news_and_stops_at_t_minus_1():
    dates = pd.date_range("2020-01-01", periods=50, tz="UTC")
    target = pd.Timestamp("2020-02-10", tz="UTC")
    update_daily = np.arange(100, dtype=np.float64).reshape(50, 2)
    update_available = np.zeros(50, dtype=bool)
    update_available[[11, 20, 39, 40]] = True
    level_daily = np.arange(150, dtype=np.float64).reshape(50, 3)
    dataset = SlowUpdateDataset(
        [target],
        np.zeros((1, 7)),
        np.zeros((1, 21)),
        update_daily,
        update_available,
        level_daily,
        dates,
        np.zeros(1),
        np.ones(1),
        30,
    )
    row = dataset[0]
    np.testing.assert_array_equal(
        row["update_tokens"].numpy(), update_daily[10:40].astype(np.float32)
    )
    np.testing.assert_array_equal(
        row["update_padding_mask"].numpy(), ~update_available[10:40]
    )
    np.testing.assert_array_equal(
        row["slow_level"].numpy(), level_daily[39].astype(np.float32)
    )
    assert bool(update_available[40]) is True
    assert row["target_date"] == "2020-02-10"


def test_all_slow_update_variants_start_exactly_at_har():
    config = MainPilotConfig(profile=PROFILE)
    for variant in (
        "slow_update_tokens",
        "slow_update_multiquery",
        "slow_update_multiquery_gated",
    ):
        model = SlowUpdateTransformer(
            config,
            update_dim=34,
            level_dim=22,
            gate_dim=21,
            variant=variant,
        )
        model.eval()
        anchor = torch.tensor([0.25, -0.75], dtype=torch.float64)
        output = model(
            {
                "market_query": torch.randn(2, 7),
                "gate_features": torch.randn(2, 21),
                "update_tokens": torch.randn(2, 30, 34),
                "update_padding_mask": torch.ones(2, 30, dtype=torch.bool),
                "slow_level": torch.randn(2, 22),
                "har_anchor_log_rv": anchor,
            }
        )
        assert torch.equal(output["predicted_log_rv"], anchor)
        assert torch.count_nonzero(output["delta_log_rv"]) == 0
        if variant.endswith("_gated"):
            assert torch.all((output["correction_gate"] > 0))
            assert torch.all((output["correction_gate"] < 1))


def test_blend_selects_alpha_using_validation_only():
    def frame(true, predicted):
        true = np.asarray(true, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        return pd.DataFrame(
            {
                "target_date": ["2020-01-01", "2020-01-02"],
                "true_rv": true,
                "true_log_rv": np.log(true),
                "predicted_rv": predicted,
                "predicted_log_rv": np.log(predicted),
                "har_anchor_log_rv": np.zeros(2),
                "delta_log_rv": np.log(predicted),
            }
        )

    validation_finbert = frame([2.0, 3.0], [1.0, 1.0])
    validation_slow = frame([2.0, 3.0], [2.0, 3.0])
    test_finbert = frame([4.0, 5.0], [4.0, 5.0])
    test_slow = frame([4.0, 5.0], [1.0, 1.0])
    _, test, metadata, grid = select_log_blend(
        validation_finbert,
        validation_slow,
        test_finbert,
        test_slow,
    )
    assert metadata["alpha_slow"] == 1.0
    assert tuple(grid["alpha_slow"]) == BLEND_ALPHA_GRID
    np.testing.assert_allclose(test["predicted_rv"], [1.0, 1.0])
