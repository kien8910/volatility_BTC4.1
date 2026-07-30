from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from btc_main_pilot.cli import build_parser
from btc_main_pilot.config import Fold, MainPilotConfig
from btc_main_pilot.preprocess import FoldNewsFeatures
from btc_main_pilot.slow_transformer_fold5_evaluation import (
    PCA_DIMS,
    PRIMARY_MODEL,
    PROFILE,
    _comparison_rows,
)
from btc_main_pilot.slow_transformer_v2_diagnostic import (
    SlowUpdateTransformer,
    _assert_epoch_zero_har,
    _prepare_update_daily,
)


def test_fold5_profile_is_registered_and_dates_are_locked():
    config = MainPilotConfig(profile=PROFILE)
    config.validate()
    args = build_parser().parse_args(["--profile", PROFILE, "--smoke"])
    assert args.profile == PROFILE
    assert PCA_DIMS == (8, 16)
    assert PRIMARY_MODEL == "slow_calendar_control"
    assert config.fold_5 == Fold(
        "fold_5",
        "2018-01-01",
        "2023-07-21",
        "2023-07-22",
        "2023-10-19",
        "2023-10-20",
        "2024-04-16",
    )


def test_pca16_update_and_level_feature_dimensions_are_dynamic():
    dates = pd.date_range("2020-01-01", periods=40, tz="UTC")
    rng = np.random.default_rng(11)
    news = FoldNewsFeatures(
        dates=dates,
        semantic_slow=rng.normal(size=(40, 16)).astype(np.float32),
        semantic_fast=np.zeros((40, 16), dtype=np.float32),
        sentiment_slow=rng.normal(size=(40, 3)).astype(np.float32),
        sentiment_fast=np.zeros((40, 3), dtype=np.float32),
        daily_scalars=np.column_stack(
            [rng.normal(size=(40, 10)), np.zeros(40)]
        ).astype(np.float32),
        preprocessor_hash="pca16",
        metadata={"pca_n_components": 16},
    )
    causal_event = pd.DataFrame(index=dates)
    for family in (
        "direct_bitcoin",
        "regulation_etf",
        "exchange_custody",
        "macro_liquidity",
        "mining_energy",
        "security_hack",
    ):
        causal_event[f"{family}__count_surprise_365d"] = 0.0
        causal_event[f"{family}__log1p_source_count"] = 0.0
    fold = Fold(
        "test",
        "2020-01-01",
        "2020-02-09",
        "2020-02-10",
        "2020-02-10",
        "2020-02-11",
        "2020-02-11",
    )
    (
        updates,
        _,
        levels,
        _,
        _,
        update_names,
        level_names,
    ) = _prepare_update_daily(news, causal_event, fold)
    assert updates.shape == (40, 42)
    assert levels.shape == (40, 30)
    assert len(update_names) == updates.shape[1]
    assert len(level_names) == levels.shape[1]
    assert "delta_semantic_slow_pc_16" in update_names
    assert "semantic_slow_pc_16" in level_names


def test_epoch_zero_assertion_restores_mode_and_reseeds_rng():
    config = MainPilotConfig(
        profile=PROFILE,
        d_model=16,
        ffn_dim=32,
    )
    model = SlowUpdateTransformer(
        config,
        update_dim=42,
        level_dim=30,
        gate_dim=21,
        variant="slow_update_multiquery_gated",
    )
    model.train()
    anchor = torch.tensor([0.25], dtype=torch.float64)
    batch = {
        "market_query": torch.randn(1, 7),
        "gate_features": torch.randn(1, 21),
        "update_tokens": torch.randn(1, 30, 42),
        "update_padding_mask": torch.zeros(1, 30, dtype=torch.bool),
        "slow_level": torch.randn(1, 30),
        "har_anchor_log_rv": anchor,
    }
    _assert_epoch_zero_har(model, batch, model.variant, config.seed)
    assert model.training is True
    observed = torch.rand(4)
    torch.manual_seed(config.seed)
    expected = torch.rand(4)
    assert torch.equal(observed, expected)


def test_pca_comparison_uses_pca8_as_fixed_reference():
    def metrics(mean: float) -> dict[str, float | str | int]:
        return {
            "model": "slow_calendar_control",
            "n_predictions": 10,
            "mean_qlike": mean,
            "normal_qlike": mean + 0.1,
            "spike_qlike": mean + 1.0,
            "r2_logrv": 0.2 - mean,
            "rmse_logrv": 0.8 + mean,
        }

    rows = _comparison_rows(
        {
            "pca8": {"pca_dim": 8, "pooled_metrics": [metrics(0.30)]},
            "pca16": {"pca_dim": 16, "pooled_metrics": [metrics(0.25)]},
        }
    )
    pca16 = next(row for row in rows if row["pca_dim"] == 16)
    assert np.isclose(pca16["delta_mean_qlike_vs_pca8"], -0.05)
    assert np.isclose(pca16["delta_normal_qlike_vs_pca8"], -0.05)
    assert np.isclose(pca16["delta_spike_qlike_vs_pca8"], -0.05)
