import numpy as np
import pandas as pd

from btc_main_pilot.cli import build_parser
from btc_main_pilot.config import FOLD_1, MainPilotConfig
from btc_main_pilot.point_in_time_refit_diagnostic import (
    PROFILE,
    _comparison_frame,
    _fit_locked_refit_probe,
    _refit_preprocess_fold,
)


def test_refit_profile_is_registered_in_config_and_cli():
    config = MainPilotConfig(profile=PROFILE)
    config.validate()
    args = build_parser().parse_args(["--profile", PROFILE, "--smoke"])
    assert args.profile == PROFILE
    assert args.smoke


def test_refit_preprocessor_ends_at_validation_and_never_uses_test():
    refit = _refit_preprocess_fold(FOLD_1)
    assert refit.core_start == FOLD_1.core_start
    assert refit.core_end == FOLD_1.validation_end
    assert refit.core_end < FOLD_1.test_start
    assert refit.validation_start == FOLD_1.test_start
    assert refit.validation_end == FOLD_1.test_end


def test_locked_rejected_probe_refits_to_har_fallback():
    train_rv = np.asarray([1.0, 1.5, 2.0, 2.5])
    test_rv = np.asarray([1.2, 2.2])
    train_anchor = np.log(np.asarray([1.1, 1.4, 1.9, 2.4]))
    test_anchor = np.log(np.asarray([1.3, 2.0]))
    dates = list(
        pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    )
    frame, metadata = _fit_locked_refit_probe(
        name="finbert_slow_fast_6",
        x_train=np.arange(24, dtype=float).reshape(4, 6),
        x_test=np.arange(12, dtype=float).reshape(2, 6),
        train_rv=train_rv,
        test_rv=test_rv,
        train_anchor_log=train_anchor,
        test_anchor_log=test_anchor,
        test_dates=dates,
        feature_names=[f"x{i}" for i in range(6)],
        locked_selection={
            "correction_selected": False,
            "selected_lambda_sum": 10.0,
        },
    )
    np.testing.assert_allclose(frame["predicted_log_rv"], test_anchor)
    np.testing.assert_allclose(frame["delta_log_rv"], 0.0)
    assert metadata["refit_status"] == "locked_har_fallback"
    assert metadata["selection_locked_before_refit"]


def test_selected_probe_uses_locked_lambda_on_core_plus_validation():
    x_train = np.asarray(
        [
            [-2.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    train_anchor = np.zeros(len(x_train))
    train_rv = np.exp(0.15 * x_train[:, 0])
    x_test = np.asarray([[0.5], [1.5]])
    dates = list(
        pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    )
    frame, metadata = _fit_locked_refit_probe(
        name="synthetic",
        x_train=x_train,
        x_test=x_test,
        train_rv=train_rv,
        test_rv=np.asarray([1.0, 1.0]),
        train_anchor_log=train_anchor,
        test_anchor_log=np.zeros(2),
        test_dates=dates,
        feature_names=["signal"],
        locked_selection={
            "correction_selected": True,
            "selected_lambda_sum": 0.01,
        },
    )
    assert metadata["refit_status"] == "fitted"
    assert metadata["refit_n"] == 6
    assert np.isclose(
        metadata["selected_alpha_mean_loss"], 2.0 * 0.01 / 6
    )
    assert np.all(np.isfinite(frame["predicted_rv"]))
    assert np.all(frame["predicted_rv"] > 0)


def test_before_after_comparison_uses_after_minus_before():
    before = [
        {
            "fold": "fold_1",
            "model": "har_qlike",
            "mean_qlike": 0.4,
            "normal_qlike": 0.3,
            "spike_qlike": 1.0,
            "r2_logrv": 0.1,
            "rmse_logrv": 1.0,
            "mae_logrv": 0.8,
        }
    ]
    after = [
        {
            "fold": "fold_1",
            "model": "har_qlike",
            "mean_qlike": 0.3,
            "normal_qlike": 0.25,
            "spike_qlike": 0.8,
            "r2_logrv": 0.2,
            "rmse_logrv": 0.9,
            "mae_logrv": 0.7,
        }
    ]
    comparison = _comparison_frame(before, after, ("fold", "model"))
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert np.isclose(row["delta_mean_qlike"], -0.1)
    assert np.isclose(row["relative_change_pct_mean_qlike"], -25.0)
