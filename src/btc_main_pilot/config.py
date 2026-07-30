from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Fold:
    name: str
    core_start: str
    core_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


FOLD_1 = Fold(
    "fold_1",
    "2018-01-01",
    "2021-07-31",
    "2021-08-01",
    "2021-10-29",
    "2021-10-30",
    "2022-04-27",
)
FOLD_2 = Fold(
    "fold_2",
    "2018-01-01",
    "2022-01-27",
    "2022-01-28",
    "2022-04-27",
    "2022-04-28",
    "2022-10-24",
)
FOLD_3 = Fold(
    "fold_3",
    "2018-01-01",
    "2022-07-26",
    "2022-07-27",
    "2022-10-24",
    "2022-10-25",
    "2023-04-22",
)
FOLD_4 = Fold(
    "fold_4",
    "2018-01-01",
    "2023-01-22",
    "2023-01-23",
    "2023-04-22",
    "2023-04-23",
    "2023-10-19",
)
FOLD_5 = Fold(
    "fold_5",
    "2018-01-01",
    "2023-07-21",
    "2023-07-22",
    "2023-10-19",
    "2023-10-20",
    "2024-04-16",
)

SPIKE_DIAGNOSTIC_FOLDS = (FOLD_1, FOLD_2, FOLD_3, FOLD_4)
SPIKE_DIAGNOSTIC_VARIANTS = (
    "main",
    "market_only",
    "hybrid_har",
)
REGIME_ANCHOR_VARIANTS = (
    "har_anchor_market",
    "har_anchor_market_text",
)


@dataclass
class MainPilotConfig:
    profile: str = "main-pilot"
    market_path: str = "data/BTCUSDT_5min_2018_2025_present.csv"
    news_path: str = "data/news_clusters.json"
    output_dir: str = "outputs/main_pilot"
    embedding_cache_path: str | None = None
    research_start: str = "2018-01-01"
    research_end: str = "2025-06-30"
    development_end: str = "2024-04-16"
    seed: int = 11

    semantic_model: str = "BAAI/bge-base-en-v1.5"
    sentiment_model: str = "ProsusAI/finbert"
    embedding_dim: int = 768
    max_tokens: int = 512
    embedding_batch_size: int = 32
    pca_dim: int = 8
    slow_alpha: float = 2.0 / 31.0

    fine_lookback_days: int = 7
    fine_patch_length: int = 12
    coarse_lookback_days: int = 22
    coarse_patch_length: int = 72
    news_lookback_days: int = 30
    bars_per_day: int = 288
    channels: int = 7
    daily_scalars: int = 11
    epsilon_r: float = 1e-12
    epsilon_rv: float = 1e-12
    scaler_epsilon: float = 1e-8
    verified_maintenance_intervals: tuple[tuple[str, str, str], ...] = (
        (
            "binance_spot_upgrade_2021_08_13",
            "2021-08-13 02:00:00+00:00",
            "2021-08-13 06:25:00+00:00",
        ),
        (
            "binance_spot_upgrade_2021_09_29",
            "2021-09-29 07:00:00+00:00",
            "2021-09-29 08:55:00+00:00",
        ),
    )

    d_model: int = 32
    attention_heads: int = 4
    patch_layers: int = 2
    news_layers: int = 1
    cross_layers: int = 1
    ffn_dim: int = 64
    forecast_queries: int = 4
    dropout: float = 0.1
    parameter_budget: int = 60_000

    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-6
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    provisional_horizon_epochs: int = 200
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    gradient_clip_norm: float = 1.0
    amp_grad_scaler_initial_scale: float = 1024.0
    amp_grad_scaler_growth_interval: int = 2000
    effective_batch_size: int = 32
    physical_batch_size: int = 32
    num_workers: int = 0
    training_loss: str = "exact_qlike"

    fold_1: Fold = field(default_factory=lambda: FOLD_1)
    fold_5: Fold = field(default_factory=lambda: FOLD_5)

    smoke: bool = False
    smoke_start: str = "2018-01-01"
    smoke_end: str = "2018-05-31"
    smoke_max_train_batches: int = 2
    smoke_max_eval_batches: int = 2
    smoke_epochs: int = 2

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def validate(self) -> None:
        assert self.profile in {
            "main-pilot",
            "development-spike-diagnostic",
            "development-regime-anchor-diagnostic",
            "development-news-representation-audit",
            "development-event-aware-longtext-audit",
            "development-point-in-time-gate-diagnostic",
            "development-point-in-time-refit-diagnostic",
            "development-tail-regime-diagnostic",
            "development-vector-integration-diagnostic",
            "development-slow-transformer-v2-diagnostic",
            "slow-transformer-fold5-evaluation",
        }
        assert self.seed == 11
        assert self.pca_dim == 8
        assert self.optimizer == "AdamW"
        assert self.learning_rate == 3e-4
        assert self.weight_decay == 1e-4
        assert self.warmup_steps == 100
        assert self.max_epochs == 200
        assert self.patience == 20
        assert self.gradient_clip_norm == 1.0
        assert self.amp_grad_scaler_initial_scale == 1024.0
        assert self.amp_grad_scaler_growth_interval == 2000
        assert self.training_loss == "exact_qlike"
        assert self.fine_patch_length == 12 and self.coarse_patch_length == 72
        assert self.fine_lookback_days == 7 and self.coarse_lookback_days == 22
        assert self.news_lookback_days == 30
        assert self.embedding_dim == 768
        assert self.channels == 7
        assert self.daily_scalars == 11
        assert self.semantic_model == "BAAI/bge-base-en-v1.5"
        assert self.sentiment_model == "ProsusAI/finbert"
        assert self.max_tokens == 512
        assert self.verified_maintenance_intervals == (
            (
                "binance_spot_upgrade_2021_08_13",
                "2021-08-13 02:00:00+00:00",
                "2021-08-13 06:25:00+00:00",
            ),
            (
                "binance_spot_upgrade_2021_09_29",
                "2021-09-29 07:00:00+00:00",
                "2021-09-29 08:55:00+00:00",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
