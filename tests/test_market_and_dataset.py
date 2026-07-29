from pathlib import Path

import numpy as np
import pandas as pd

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.data import load_market_data
from btc_main_pilot.dataset import patch_logrv


def _write_market_csv(path: Path, include_boundary: bool) -> tuple[float, float]:
    start = pd.Timestamp("2017-12-31 23:55:00")
    timestamps = pd.date_range(start, periods=289, freq="5min")
    if not include_boundary:
        timestamps = timestamps[1:]
    close = 10_000.0 * np.exp(np.arange(len(timestamps)) * 1e-4)
    frame = pd.DataFrame(
        {
            "Open Time": timestamps,
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": 10.0,
            "Quote Asset Volume": close * 10.0,
            "Number of Trades": 5,
            "Taker Buy Base Asset Volume": 5.0,
            "Taker Buy Quote Asset Volume": close * 5.0,
        }
    )
    frame.to_csv(path, index=False)
    return float(close[0]), float(close[1])


def test_market_day_has_288_returns_and_uses_previous_2355(tmp_path):
    path = tmp_path / "market.csv"
    previous, midnight = _write_market_csv(path, include_boundary=True)
    config = MainPilotConfig(research_start="2018-01-01", research_end="2018-01-01")
    market = load_market_data(
        path,
        config,
        __import__("logging").getLogger("test"),
        "2018-01-01",
        "2018-01-01",
    )
    assert market.valid.tolist() == [True]
    assert market.raw_returns.shape == (1, 288)
    np.testing.assert_allclose(
        market.raw_returns[0, 0], np.log(midnight) - np.log(previous), rtol=1e-12
    )


def test_287_intraday_differences_without_boundary_are_invalid(tmp_path):
    path = tmp_path / "market.csv"
    _write_market_csv(path, include_boundary=False)
    config = MainPilotConfig(research_start="2018-01-01", research_end="2018-01-01")
    market = load_market_data(
        path,
        config,
        __import__("logging").getLogger("test"),
        "2018-01-01",
        "2018-01-01",
    )
    assert market.valid.tolist() == [False]
    assert market.audit["invalid_reasons"]["missing_boundary_return"] == 1


def test_patch_logrv_is_log_sum_raw_squared_returns_not_mean_logs():
    returns = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    actual = patch_logrv(returns, patch_length=2, epsilon=1e-12)
    expected = np.array(
        [
            np.log(0.1**2 + 0.2**2 + 1e-12),
            np.log(0.3**2 + 0.4**2 + 1e-12),
        ]
    )
    np.testing.assert_allclose(actual, expected)
    mean_log_squared = np.log(returns.reshape(2, 2) ** 2 + 1e-12).mean(axis=1)
    assert not np.allclose(actual, mean_log_squared)

