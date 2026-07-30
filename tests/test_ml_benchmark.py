from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from btc_main_pilot.baselines import fit_arch_family_baselines
from btc_main_pilot.cli import build_parser
from btc_main_pilot.config import (
    DEVELOPMENT_FOLDS,
    FINAL_HOLDOUT_FOLD,
    Fold,
    MainPilotConfig,
)
from btc_main_pilot.data import MarketData
from btc_main_pilot.ml_benchmark import (
    PRIMARY_SEEDS,
    PROFILE,
    _common_keys,
    _freeze_protocol,
    _seed_ensemble,
    _verify_frozen_protocol,
    parse_benchmark_seeds,
)


def test_profile_dates_and_cli_are_locked():
    config = MainPilotConfig(profile=PROFILE)
    config.validate()
    args = build_parser().parse_args(
        [
            "--profile",
            PROFILE,
            "--benchmark-phase",
            "development",
            "--benchmark-seeds",
            "11",
        ]
    )
    assert args.profile == PROFILE
    assert args.benchmark_phase == "development"
    assert parse_benchmark_seeds("11,22,33,44,55") == PRIMARY_SEEDS
    assert len(DEVELOPMENT_FOLDS) == 5
    assert FINAL_HOLDOUT_FOLD == Fold(
        "final_holdout",
        "2018-01-01",
        "2024-01-17",
        "2024-01-18",
        "2024-04-16",
        "2024-04-17",
        "2025-06-30",
    )


def test_seed_parser_rejects_duplicates_and_nonintegers():
    with pytest.raises(ValueError):
        parse_benchmark_seeds("11,11")
    with pytest.raises(ValueError):
        parse_benchmark_seeds("11,bad")


def test_frozen_protocol_detects_any_change(tmp_path):
    payload = {"candidates": ["slow"], "seeds": [11]}
    _freeze_protocol(tmp_path, payload)
    report = tmp_path / "development" / "metrics" / "benchmark_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        '{"status":"completed","primary_model_main_table_eligible":true}',
        encoding="utf-8",
    )
    assert _verify_frozen_protocol(tmp_path, payload)["payload"] == payload
    with pytest.raises(RuntimeError, match="differs"):
        _verify_frozen_protocol(
            tmp_path, {"candidates": ["fast"], "seeds": [11]}
        )


def test_seed_ensemble_averages_variance_not_log_variance(tmp_path):
    for seed, predictions in ((11, [1.0, 4.0]), (22, [3.0, 8.0])):
        directory = tmp_path / f"seed_{seed}" / "predictions"
        directory.mkdir(parents=True)
        metrics_directory = tmp_path / f"seed_{seed}" / "metrics"
        metrics_directory.mkdir(parents=True)
        (metrics_directory / "model_completion.json").write_text(
            (
                '{"har_qlike":{"eligible_for_seed_ensemble":true,'
                '"expected_folds":["fold_1"],'
                '"completed_folds":["fold_1"],"missing_folds":[]}}'
            ),
            encoding="utf-8",
        )
        predicted = np.asarray(predictions, dtype=np.float64)
        true = np.asarray([2.0, 6.0], dtype=np.float64)
        pd.DataFrame(
            {
                "target_date": ["2020-01-01", "2020-01-02"],
                "true_rv": true,
                "true_log_rv": np.log(true),
                "predicted_rv": predicted,
                "predicted_log_rv": np.log(predicted),
                "fold": ["fold_1", "fold_1"],
                "model": ["har_qlike", "har_qlike"],
                "spike_threshold": [5.0, 5.0],
                "is_spike": [False, True],
                "predicted_spike": [False, seed == 22],
                "qlike": [0.0, 0.0],
            }
        ).to_csv(directory / "pooled_har_qlike.csv", index=False)
    frames, _ = _seed_ensemble(tmp_path, (11, 22))
    np.testing.assert_allclose(
        frames["har_qlike"]["predicted_rv"], [2.0, 6.0]
    )
    np.testing.assert_allclose(
        frames["har_qlike"]["predicted_log_rv"], np.log([2.0, 6.0])
    )


def test_seed_ensemble_excludes_incomplete_model_even_if_stale_file_exists(
    tmp_path,
):
    directory = tmp_path / "seed_11" / "predictions"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "target_date": ["2020-01-01"],
            "fold": ["fold_1"],
            "true_rv": [1.0],
            "true_log_rv": [0.0],
            "predicted_rv": [1.0],
            "predicted_log_rv": [0.0],
            "spike_threshold": [2.0],
            "is_spike": [False],
            "predicted_spike": [False],
            "qlike": [0.0],
            "model": ["slow_calendar_control"],
        }
    ).to_csv(
        directory / "pooled_slow_calendar_control.csv", index=False
    )
    metrics_directory = tmp_path / "seed_11" / "metrics"
    metrics_directory.mkdir(parents=True)
    (metrics_directory / "model_completion.json").write_text(
        (
            '{"slow_calendar_control":{'
            '"eligible_for_seed_ensemble":false,'
            '"expected_folds":["fold_1","fold_2"],'
            '"completed_folds":["fold_1"],"missing_folds":["fold_2"]}}'
        ),
        encoding="utf-8",
    )
    frames, diagnostics = _seed_ensemble(tmp_path, (11,))
    assert "slow_calendar_control" not in frames
    status = next(
        row["status"]
        for row in diagnostics
        if row["model"] == "slow_calendar_control"
    )
    assert status == "ineligible_incomplete_folds"


def test_common_support_is_fold_and_date_intersection():
    first = pd.DataFrame(
        {
            "fold": ["fold_1", "fold_1"],
            "target_date": ["2020-01-01", "2020-01-02"],
        }
    )
    second = pd.DataFrame(
        {
            "fold": ["fold_1", "fold_2"],
            "target_date": ["2020-01-02", "2020-01-02"],
        }
    )
    assert _common_keys({"a": first, "b": second}) == {
        ("fold_1", "2020-01-02")
    }


def _synthetic_market() -> MarketData:
    dates = pd.date_range("2019-01-01", periods=380, tz="UTC")
    rng = np.random.default_rng(11)
    daily = rng.normal(0.0002, 0.025, len(dates))
    raw_returns = np.zeros((len(dates), 288), dtype=np.float64)
    raw_returns[:, 0] = daily
    rv = daily**2 + 1e-8
    return MarketData(
        dates=dates,
        features=np.zeros((len(dates), 288, 7), dtype=np.float32),
        raw_returns=raw_returns,
        rv=rv,
        log_rv=np.log(rv),
        valid=np.ones(len(dates), dtype=bool),
        zero_volume=np.zeros((len(dates), 288), dtype=bool),
        maintenance_synthetic=np.zeros((len(dates), 288), dtype=bool),
        date_to_index={date: index for index, date in enumerate(dates)},
        audit={},
    )


def test_arch_family_uses_core_and_emits_positive_test_variance():
    market = _synthetic_market()
    fold = Fold(
        "synthetic",
        "2019-01-01",
        "2019-10-27",
        "2019-10-28",
        "2019-11-06",
        "2019-11-07",
        "2019-12-16",
    )
    test_dates = list(
        market.dates[
            (market.dates >= pd.Timestamp(fold.test_start, tz="UTC"))
            & (market.dates <= pd.Timestamp(fold.test_end, tz="UTC"))
        ]
    )
    results, failures = fit_arch_family_baselines(
        market, fold, test_dates, logging.getLogger("arch-test")
    )
    assert "garch_1_1_normal" in {result.name for result in results}
    assert all(
        np.isfinite(result.predictions["predicted_rv"]).all()
        and (result.predictions["predicted_rv"] > 0).all()
        for result in results
    )
    assert all(item["fold"] == "synthetic" for item in failures)
