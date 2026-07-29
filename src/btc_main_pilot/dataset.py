from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import MainPilotConfig
from .data import MarketData
from .preprocess import FoldMarketScaler, FoldNewsFeatures


def patch_logrv(raw_returns: np.ndarray, patch_length: int, epsilon: float) -> np.ndarray:
    raw = np.asarray(raw_returns, dtype=np.float64).reshape(-1, patch_length)
    return np.log(np.sum(raw**2, axis=1) + epsilon)


def _time_features(
    target: pd.Timestamp, lookback_days: int, patches_per_day: int
) -> np.ndarray:
    starts = pd.date_range(
        target - pd.Timedelta(days=lookback_days),
        target,
        periods=lookback_days * patches_per_day + 1,
        inclusive="left",
    )
    hour = starts.hour.to_numpy(dtype=np.float64)
    dow = starts.dayofweek.to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    ).astype(np.float32)


class RVWindowDataset(Dataset):
    def __init__(
        self,
        market: MarketData,
        news: FoldNewsFeatures,
        target_dates: list[pd.Timestamp],
        market_scaler: FoldMarketScaler,
        target_mean: float,
        target_scale: float,
        config: MainPilotConfig,
    ):
        self.market = market
        self.news = news
        self.target_dates = target_dates
        self.target_indices = [market.date_to_index[date] for date in target_dates]
        self.market_scaler = market_scaler
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.config = config
        self.news_date_to_index = {day: i for i, day in enumerate(news.dates)}

    def __len__(self) -> int:
        return len(self.target_dates)

    def _market_scale(
        self, target_index: int, lookback: int, patch_length: int, fine: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_features = self.market.features[target_index - lookback : target_index]
        raw_returns = self.market.raw_returns[target_index - lookback : target_index]
        scaled = self.market_scaler.transform(raw_features).reshape(-1, 7)
        patches = scaled.reshape(-1, patch_length, 7).transpose(2, 0, 1)
        logrv = patch_logrv(raw_returns, patch_length, self.config.epsilon_rv)
        if fine:
            logrv = (
                logrv - self.market_scaler.fine_patch_logrv_mean
            ) / self.market_scaler.fine_patch_logrv_scale
        else:
            logrv = (
                logrv - self.market_scaler.coarse_patch_logrv_mean
            ) / self.market_scaler.coarse_patch_logrv_scale
        return patches.astype(np.float32), logrv.astype(np.float32)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        target = self.target_dates[item]
        target_index = self.target_indices[item]
        fine, fine_logrv = self._market_scale(
            target_index,
            self.config.fine_lookback_days,
            self.config.fine_patch_length,
            True,
        )
        coarse, coarse_logrv = self._market_scale(
            target_index,
            self.config.coarse_lookback_days,
            self.config.coarse_patch_length,
            False,
        )
        news_end = self.news_date_to_index[target - pd.Timedelta(days=1)] + 1
        news_start = news_end - self.config.news_lookback_days
        if news_start < 0:
            raise IndexError("Insufficient news lookback")
        latest_scalars = self.news.daily_scalars[news_end - 1]
        true_log_rv = self.market.log_rv[target_index]
        if not np.isfinite(true_log_rv):
            raise FloatingPointError(f"Invalid target RV on {target}")
        har_lags = self.market.log_rv[target_index - 22 : target_index]
        if len(har_lags) != 22 or not np.isfinite(har_lags).all():
            raise FloatingPointError(f"Invalid HAR lag window on {target}")
        har_scalars = np.asarray(
            [
                har_lags[-1],
                np.mean(har_lags[-5:]),
                np.mean(har_lags),
            ],
            dtype=np.float64,
        )
        har_scalars = (har_scalars - self.target_mean) / self.target_scale
        return {
            "fine": torch.from_numpy(fine),
            "fine_patch_logrv": torch.from_numpy(fine_logrv),
            "fine_time": torch.from_numpy(_time_features(target, 7, 24)),
            "coarse": torch.from_numpy(coarse),
            "coarse_patch_logrv": torch.from_numpy(coarse_logrv),
            "coarse_time": torch.from_numpy(_time_features(target, 60, 4)),
            "semantic_slow": torch.from_numpy(
                self.news.semantic_slow[news_start:news_end]
            ),
            "semantic_fast": torch.from_numpy(
                self.news.semantic_fast[news_start:news_end]
            ),
            "sentiment_slow": torch.from_numpy(
                self.news.sentiment_slow[news_start:news_end]
            ),
            "sentiment_fast": torch.from_numpy(
                self.news.sentiment_fast[news_start:news_end]
            ),
            "daily_scalars": torch.from_numpy(
                self.news.daily_scalars[news_start:news_end]
            ),
            "head_scalars": torch.tensor(
                [latest_scalars[0], latest_scalars[-1]], dtype=torch.float32
            ),
            "har_scalars": torch.from_numpy(har_scalars.astype(np.float32)),
            "true_log_rv": torch.tensor(true_log_rv, dtype=torch.float64),
            "true_rv": torch.tensor(self.market.rv[target_index], dtype=torch.float64),
            "target_date": target.strftime("%Y-%m-%d"),
        }
