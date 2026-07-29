from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MainPilotConfig
from .utils import ensure_finite, file_fingerprint, write_json


MODEL_CHANNELS = [
    "log_return",
    "log_squared_return",
    "log_high_low_range",
    "log1p_volume",
    "log1p_number_of_trades",
    "taker_buy_ratio",
    "taker_buy_imbalance",
]
CSV_COLUMNS = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume",
]


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    features: np.ndarray  # [day, 288, 7], NaN for invalid day
    raw_returns: np.ndarray  # [day, 288]
    rv: np.ndarray  # [day]
    log_rv: np.ndarray  # [day]
    valid: np.ndarray  # [day]
    zero_volume: np.ndarray  # [day, 288]
    maintenance_synthetic: np.ndarray  # [day, 288], audit-only
    date_to_index: dict[pd.Timestamp, int]
    audit: dict[str, Any]


def _insert_verified_maintenance_bars(
    raw: pd.DataFrame,
    config: MainPilotConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Causally insert no-trade bars only inside pre-locked maintenance intervals."""
    raw = raw.copy()
    raw["_maintenance_synthetic"] = False
    interval_audits: list[dict[str, Any]] = []
    inserted_frames: list[pd.DataFrame] = []
    available_times = set(raw["Open Time"])
    for name, start_text, end_text in config.verified_maintenance_intervals:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        expected = pd.date_range(start, end, freq="5min")
        in_loaded_range = expected[
            (expected >= raw["Open Time"].min()) & (expected <= raw["Open Time"].max())
        ]
        missing = in_loaded_range[~in_loaded_range.isin(available_times)]
        reference_time = start - pd.Timedelta(minutes=5)
        reference = raw.loc[raw["Open Time"] == reference_time]
        if len(missing) and len(reference) != 1:
            raise RuntimeError(
                f"Cannot causally fill {name}: expected exactly one preceding bar at "
                f"{reference_time}, found {len(reference)}"
            )
        reference_close = (
            float(reference.iloc[0]["Close"]) if len(reference) == 1 else None
        )
        if len(missing):
            if not np.isfinite(reference_close) or reference_close <= 0:
                raise RuntimeError(f"Invalid preceding close for maintenance interval {name}")
            synthetic = pd.DataFrame(
                {
                    "Open Time": missing,
                    "Open": reference_close,
                    "High": reference_close,
                    "Low": reference_close,
                    "Close": reference_close,
                    "Volume": 0.0,
                    "Quote Asset Volume": 0.0,
                    "Number of Trades": 0.0,
                    "Taker Buy Base Asset Volume": 0.0,
                    "Taker Buy Quote Asset Volume": 0.0,
                    "_maintenance_synthetic": True,
                }
            )
            inserted_frames.append(synthetic)
            available_times.update(missing)
        interval_audits.append(
            {
                "name": name,
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "expected_grid_bars": int(len(in_loaded_range)),
                "bars_already_present": int(len(in_loaded_range) - len(missing)),
                "synthetic_bars_inserted": int(len(missing)),
                "reference_close_time_utc": reference_time.isoformat(),
                "reference_close": reference_close,
                "policy": (
                    "OHLC=last observed close before verified closure; volume, quote "
                    "volume, trades and taker-buy volumes=0; no future price used"
                ),
            }
        )
    if inserted_frames:
        raw = pd.concat([raw, *inserted_frames], ignore_index=True)
        raw.sort_values("Open Time", inplace=True)
        raw.reset_index(drop=True, inplace=True)
    return raw, {
        "policy": "verified_exchange_maintenance_no_trade_bars",
        "feature_channel_added": False,
        "total_synthetic_bars_inserted": int(
            sum(item["synthetic_bars_inserted"] for item in interval_audits)
        ),
        "intervals": interval_audits,
    }


def load_market_data(
    path: Path,
    config: MainPilotConfig,
    logger: logging.Logger,
    start: str | None = None,
    end: str | None = None,
) -> MarketData:
    research_start = pd.Timestamp(start or config.research_start, tz="UTC")
    research_end = pd.Timestamp(end or config.research_end, tz="UTC")
    load_start = research_start - pd.Timedelta(minutes=5)
    load_end = research_end + pd.Timedelta(days=1)
    logger.info("MARKET LOAD | %s to %s", research_start.date(), research_end.date())
    raw = pd.read_csv(path, usecols=CSV_COLUMNS, low_memory=False)
    raw["Open Time"] = pd.to_datetime(raw["Open Time"], utc=True, errors="coerce")
    raw = raw[
        (raw["Open Time"] >= load_start) & (raw["Open Time"] < load_end)
    ].copy()
    raw.sort_values("Open Time", inplace=True)
    duplicate_count = int(raw["Open Time"].duplicated().sum())
    if duplicate_count:
        raw = raw.drop_duplicates("Open Time", keep=False)
    numeric_columns = [column for column in CSV_COLUMNS if column != "Open Time"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw, maintenance_audit = _insert_verified_maintenance_bars(raw, config)
    log_close = np.log(raw["Close"].to_numpy(dtype=np.float64))
    timestamps = raw["Open Time"].to_numpy()
    returns = np.empty(len(raw), dtype=np.float64)
    returns.fill(np.nan)
    if len(raw) > 1:
        gaps = np.diff(timestamps).astype("timedelta64[m]").astype(np.int64)
        continuous = gaps == 5
        diff = np.diff(log_close)
        returns[1:] = np.where(continuous, diff, np.nan)
    raw["_return"] = returns
    raw["_date"] = raw["Open Time"].dt.normalize()

    dates = pd.date_range(research_start, research_end, freq="D", tz="UTC")
    n_days = len(dates)
    features = np.full(
        (n_days, config.bars_per_day, len(MODEL_CHANNELS)), np.nan, dtype=np.float32
    )
    raw_returns = np.full((n_days, config.bars_per_day), np.nan, dtype=np.float64)
    zero_volume = np.zeros((n_days, config.bars_per_day), dtype=bool)
    maintenance_synthetic = np.zeros((n_days, config.bars_per_day), dtype=bool)
    rv = np.full(n_days, np.nan, dtype=np.float64)
    valid = np.zeros(n_days, dtype=bool)
    date_to_index = {day: index for index, day in enumerate(dates)}

    invalid_counts = {
        "wrong_bar_count": 0,
        "grid_gap": 0,
        "missing_boundary_return": 0,
        "nonfinite_or_nonpositive": 0,
    }
    expected_offsets = pd.timedelta_range(
        start="0min", periods=config.bars_per_day, freq="5min"
    )
    for day, group in raw.groupby("_date", sort=True):
        if day not in date_to_index:
            continue
        day_index = date_to_index[day]
        if len(group) != config.bars_per_day:
            invalid_counts["wrong_bar_count"] += 1
            continue
        expected = day + expected_offsets
        actual = pd.DatetimeIndex(group["Open Time"])
        if not actual.equals(expected):
            invalid_counts["grid_gap"] += 1
            continue
        day_returns = group["_return"].to_numpy(dtype=np.float64)
        if not np.isfinite(day_returns[0]):
            invalid_counts["missing_boundary_return"] += 1
            continue
        volume = group["Volume"].to_numpy(dtype=np.float64)
        high = group["High"].to_numpy(dtype=np.float64)
        low = group["Low"].to_numpy(dtype=np.float64)
        trades = group["Number of Trades"].to_numpy(dtype=np.float64)
        taker_base = group["Taker Buy Base Asset Volume"].to_numpy(dtype=np.float64)
        zero = volume <= 0
        synthetic = group["_maintenance_synthetic"].to_numpy(dtype=bool)
        ratio = np.divide(
            taker_base,
            volume,
            out=np.full_like(volume, 0.5),
            where=~zero,
        )
        channel_values = np.column_stack(
            [
                day_returns,
                np.log(day_returns**2 + config.epsilon_r),
                np.log(high / low),
                np.log1p(volume),
                np.log1p(trades),
                ratio,
                2.0 * ratio - 1.0,
            ]
        )
        day_rv = float(np.sum(day_returns**2))
        if (
            not np.isfinite(channel_values).all()
            or not np.isfinite(day_rv)
            or day_rv <= 0
        ):
            invalid_counts["nonfinite_or_nonpositive"] += 1
            continue
        features[day_index] = channel_values.astype(np.float32)
        raw_returns[day_index] = day_returns
        zero_volume[day_index] = zero
        maintenance_synthetic[day_index] = synthetic
        rv[day_index] = day_rv
        valid[day_index] = True
    log_rv = np.log(rv)

    development_mask = dates <= pd.Timestamp(config.development_end, tz="UTC")
    development_bars = int(np.sum(valid & development_mask) * config.bars_per_day)
    development_zero = int(np.sum(zero_volume[valid & development_mask]))
    development_synthetic = int(
        np.sum(maintenance_synthetic[valid & development_mask])
    )
    development_organic_zero = int(
        np.sum(
            zero_volume[valid & development_mask]
            & ~maintenance_synthetic[valid & development_mask]
        )
    )
    quote = raw["Taker Buy Quote Asset Volume"].to_numpy(dtype=np.float64)
    base = raw["Taker Buy Base Asset Volume"].to_numpy(dtype=np.float64)
    close = raw["Close"].to_numpy(dtype=np.float64)
    quote_consistency = np.abs(quote - base * close) / np.maximum(np.abs(quote), 1e-12)
    audit: dict[str, Any] = {
        "input": file_fingerprint(path),
        "range_start": str(research_start),
        "range_end": str(research_end),
        "calendar_days": n_days,
        "valid_days": int(valid.sum()),
        "invalid_days": int((~valid).sum()),
        "duplicate_timestamps_removed": duplicate_count,
        "invalid_reasons": invalid_counts,
        "development_only_zero_volume_bars": development_zero,
        "development_only_organic_zero_volume_bars": development_organic_zero,
        "development_only_maintenance_synthetic_bars": development_synthetic,
        "development_only_bars": development_bars,
        "development_only_zero_volume_rate": development_zero
        / max(development_bars, 1),
        "zero_volume_mask_used_as_model_channel": False,
        "maintenance_synthetic_mask_used_as_model_channel": False,
        "verified_maintenance_fill": maintenance_audit,
        "taker_buy_quote_used_for_model": False,
        "taker_buy_quote_qc_relative_error": {
            "median": float(np.nanmedian(quote_consistency)),
            "p99": float(np.nanquantile(quote_consistency, 0.99)),
            "max": float(np.nanmax(quote_consistency)),
        },
        "model_channels": MODEL_CHANNELS,
    }
    logger.info(
        "MARKET QC DONE | valid_days=%d invalid_days=%d zero_volume_dev=%d/%d "
        "(maintenance=%d organic=%d)",
        int(valid.sum()),
        int((~valid).sum()),
        development_zero,
        development_bars,
        development_synthetic,
        development_organic_zero,
    )
    return MarketData(
        dates=dates,
        features=features,
        raw_returns=raw_returns,
        rv=rv,
        log_rv=log_rv,
        valid=valid,
        zero_volume=zero_volume,
        maintenance_synthetic=maintenance_synthetic,
        date_to_index=date_to_index,
        audit=audit,
    )


def sample_dates_for_block(
    market: MarketData,
    start: str,
    end: str,
    coarse_lookback: int = 60,
    fine_lookback: int = 7,
) -> tuple[list[pd.Timestamp], dict[str, int]]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    candidates = list(market.dates[(market.dates >= start_ts) & (market.dates <= end_ts)])
    kept: list[pd.Timestamp] = []
    audit = {
        "candidate_target_days": len(candidates),
        "removed_for_invalid_target_day": 0,
        "removed_for_incomplete_fine_window": 0,
        "removed_for_incomplete_coarse_window": 0,
        "final_sample_count": 0,
    }
    for target in candidates:
        index = market.date_to_index[target]
        if not market.valid[index]:
            audit["removed_for_invalid_target_day"] += 1
            continue
        fine_start = index - fine_lookback
        if fine_start < 0 or not bool(np.all(market.valid[fine_start:index])):
            audit["removed_for_incomplete_fine_window"] += 1
            continue
        coarse_start = index - coarse_lookback
        if coarse_start < 0 or not bool(np.all(market.valid[coarse_start:index])):
            audit["removed_for_incomplete_coarse_window"] += 1
            continue
        expected = pd.date_range(
            target - pd.Timedelta(days=coarse_lookback),
            target - pd.Timedelta(days=1),
            freq="D",
            tz="UTC",
        )
        if not market.dates[coarse_start:index].equals(expected):
            audit["removed_for_incomplete_coarse_window"] += 1
            continue
        kept.append(target)
    audit["final_sample_count"] = len(kept)
    return kept, audit


def write_market_audit(market: MarketData, output_dir: Path) -> None:
    write_json(output_dir / "audit" / "market_data_qc.json", market.audit)
