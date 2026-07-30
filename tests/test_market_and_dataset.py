from pathlib import Path

import numpy as np
import pandas as pd

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.data import (
    MarketData,
    load_market_data,
    sample_dates_for_block,
    sample_dates_for_har_text_block,
)
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


def _write_verified_maintenance_gap_csv(path: Path) -> tuple[float, float]:
    timestamps = pd.date_range(
        "2021-08-12 23:55:00", "2021-08-13 23:55:00", freq="5min"
    )
    positions = np.arange(len(timestamps))
    close = 10_000.0 * np.exp(positions * 1e-4)
    missing = (timestamps >= pd.Timestamp("2021-08-13 02:00:00")) & (
        timestamps <= pd.Timestamp("2021-08-13 06:25:00")
    )
    frame = pd.DataFrame(
        {
            "Open Time": timestamps[~missing],
            "Open": close[~missing],
            "High": close[~missing] * 1.001,
            "Low": close[~missing] * 0.999,
            "Close": close[~missing],
            "Volume": 10.0,
            "Quote Asset Volume": close[~missing] * 10.0,
            "Number of Trades": 5,
            "Taker Buy Base Asset Volume": 5.0,
            "Taker Buy Quote Asset Volume": close[~missing] * 5.0,
        }
    )
    frame.to_csv(path, index=False)
    preceding_close = float(close[np.where(timestamps == pd.Timestamp("2021-08-13 01:55:00"))[0][0]])
    reopening_close = float(close[np.where(timestamps == pd.Timestamp("2021-08-13 06:30:00"))[0][0]])
    return preceding_close, reopening_close


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


def test_verified_maintenance_is_filled_causally_as_no_trade_bars(tmp_path):
    path = tmp_path / "market.csv"
    preceding_close, reopening_close = _write_verified_maintenance_gap_csv(path)
    config = MainPilotConfig(
        research_start="2021-08-13", research_end="2021-08-13"
    )
    market = load_market_data(
        path,
        config,
        __import__("logging").getLogger("test"),
        "2021-08-13",
        "2021-08-13",
    )
    assert market.valid.tolist() == [True]
    assert int(market.maintenance_synthetic.sum()) == 54
    assert int(market.zero_volume.sum()) == 54
    # Positions 24..77 are the closure. They use only the 01:55 close and have zero return.
    np.testing.assert_allclose(market.raw_returns[0, 24:78], 0.0, atol=0.0)
    np.testing.assert_allclose(
        market.raw_returns[0, 78],
        np.log(reopening_close) - np.log(preceding_close),
        rtol=1e-12,
    )
    maintenance = market.audit["verified_maintenance_fill"]
    assert maintenance["total_synthetic_bars_inserted"] == 54
    assert maintenance["feature_channel_added"] is False
    assert market.audit["development_only_organic_zero_volume_bars"] == 0
    assert market.audit["development_only_maintenance_synthetic_bars"] == 54


def test_gap_outside_verified_maintenance_remains_invalid(tmp_path):
    path = tmp_path / "market.csv"
    _write_market_csv(path, include_boundary=True)
    frame = pd.read_csv(path)
    frame = frame[frame["Open Time"] != "2018-01-01 12:00:00"]
    frame.to_csv(path, index=False)
    config = MainPilotConfig(
        research_start="2018-01-01", research_end="2018-01-01"
    )
    market = load_market_data(
        path,
        config,
        __import__("logging").getLogger("test"),
        "2018-01-01",
        "2018-01-01",
    )
    assert market.valid.tolist() == [False]
    assert int(market.maintenance_synthetic.sum()) == 0
    assert (
        market.audit["verified_maintenance_fill"][
            "total_synthetic_bars_inserted"
        ]
        == 0
    )


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


def _daily_market(days: int, invalid_index: int | None = None) -> MarketData:
    dates = pd.date_range("2018-01-01", periods=days, freq="D", tz="UTC")
    valid = np.ones(days, dtype=bool)
    if invalid_index is not None:
        valid[invalid_index] = False
    rv = np.full(days, 1e-4, dtype=np.float64)
    log_rv = np.log(rv)
    rv[~valid] = np.nan
    log_rv[~valid] = np.nan
    return MarketData(
        dates=dates,
        features=np.zeros((days, 288, 7), dtype=np.float64),
        raw_returns=np.zeros((days, 288), dtype=np.float64),
        rv=rv,
        log_rv=log_rv,
        valid=valid,
        zero_volume=np.zeros((days, 288), dtype=bool),
        maintenance_synthetic=np.zeros((days, 288), dtype=bool),
        date_to_index={date: index for index, date in enumerate(dates)},
        audit={},
    )


def test_har_text_sampler_uses_only_the_22_daily_rv_lags():
    market = _daily_market(90, invalid_index=10)
    dates, audit = sample_dates_for_har_text_block(
        market,
        "2018-01-01",
        "2018-03-31",
    )
    # The invalid day leaves the 22-day lag window at index 33. A legacy
    # 60-day intraday sampler would still reject this target.
    assert market.dates[33] in dates
    assert market.dates[32] not in dates
    assert audit["final_sample_count"] == len(dates)


def test_patchtst_sampler_keeps_30_day_news_burn_in_but_22_market_days():
    market = _daily_market(40)
    text_dates, _ = sample_dates_for_har_text_block(
        market,
        "2018-01-01",
        "2018-02-09",
    )
    patch_dates, audit = sample_dates_for_block(
        market,
        "2018-01-01",
        "2018-02-09",
        coarse_lookback=22,
        fine_lookback=7,
        minimum_calendar_lookback=30,
    )
    assert len(text_dates) == 18
    assert len(patch_dates) == 10
    assert patch_dates[0] == pd.Timestamp("2018-01-31", tz="UTC")
    assert audit["removed_for_insufficient_calendar_history"] == 30
