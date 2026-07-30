from types import SimpleNamespace

import numpy as np
import pandas as pd

from btc_main_pilot.cli import build_parser
from btc_main_pilot.config import FOLD_1, MainPilotConfig
from btc_main_pilot.news import DAILY_SCALAR_COLUMNS, FilteredArticle
from btc_main_pilot.tail_regime_diagnostic import (
    CONFIGURATION_NAMES,
    PROFILE,
    _article_day_summary,
    _fit_configuration,
    _selection_screen,
    _spike_information_rows,
    _temporal_training_weights,
)


def test_tail_regime_profile_is_registered():
    config = MainPilotConfig(profile=PROFILE)
    config.validate()
    args = build_parser().parse_args(["--profile", PROFILE, "--smoke"])
    assert args.profile == PROFILE
    assert len(CONFIGURATION_NAMES) == 6


def test_temporal_weights_are_causal_and_calendar_based():
    dates = list(
        pd.date_range("2018-01-01", periods=1000, freq="D", tz="UTC")
    )
    expanding_mask, expanding_weights, _ = _temporal_training_weights(
        dates, "expanding"
    )
    assert expanding_mask.all()
    np.testing.assert_allclose(expanding_weights, 1.0)

    rolling_mask, rolling_weights, rolling_meta = (
        _temporal_training_weights(dates, "rolling730")
    )
    assert rolling_mask.sum() == 730
    assert rolling_meta["fit_end"] == dates[-1].strftime("%Y-%m-%d")
    np.testing.assert_allclose(rolling_weights, 1.0)

    decay_mask, decay_weights, _ = _temporal_training_weights(
        dates, "exp_decay365"
    )
    assert decay_mask.all()
    assert np.isclose(decay_weights.mean(), 1.0)
    assert decay_weights[-1] > decay_weights[0]
    ratio = decay_weights[-1] / decay_weights[-366]
    assert np.isclose(ratio, 2.0, rtol=0.01)


def test_soft_mixture_fit_uses_only_core_for_tail_threshold():
    rng = np.random.default_rng(11)
    core_dates = list(
        pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    )
    validation_dates = list(
        pd.date_range("2020-05-01", periods=30, freq="D", tz="UTC")
    )
    test_dates = list(
        pd.date_range("2020-06-01", periods=30, freq="D", tz="UTC")
    )
    core_normal = rng.normal(size=(120, 6))
    validation_normal = rng.normal(size=(30, 6))
    test_normal = rng.normal(size=(30, 6))
    core_tail = np.column_stack(
        [core_normal, rng.normal(size=(120, 5))]
    )
    validation_tail = np.column_stack(
        [validation_normal, rng.normal(size=(30, 5))]
    )
    test_tail = np.column_stack(
        [test_normal, rng.normal(size=(30, 5))]
    )
    core_signal = 0.20 * core_normal[:, 0] + 0.10 * core_tail[:, 6]
    core_rv = np.exp(core_signal)
    validation_rv = np.exp(
        0.20 * validation_normal[:, 0]
        + 0.10 * validation_tail[:, 6]
    )
    test_rv = np.exp(
        0.20 * test_normal[:, 0] + 0.10 * test_tail[:, 6]
    )
    validation, test, metadata, grid = _fit_configuration(
        name="expanding__finbert_tail_soft_mixture",
        correction_kind="finbert_tail_soft_mixture",
        temporal_mode="expanding",
        core_dates=core_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
        core_rv=core_rv,
        validation_rv=validation_rv,
        test_rv=test_rv,
        core_anchor_log=np.zeros(120),
        validation_anchor_log=np.zeros(30),
        test_anchor_log=np.zeros(30),
        normal_matrices=(
            core_normal,
            validation_normal,
            test_normal,
        ),
        tail_matrices=(core_tail, validation_tail, test_tail),
        lambda_grid=(0.1,),
        min_delta=1e-5,
    )
    assert metadata["core_tail_quantile"] == 0.80
    assert np.isclose(
        metadata["core_tail_threshold"],
        np.quantile(core_rv, 0.80),
    )
    assert metadata["gate_train_tail_n"] == int(
        (core_rv > np.quantile(core_rv, 0.80)).sum()
    )
    assert len(grid) == 1
    assert np.all(np.isfinite(validation["predicted_rv"]))
    assert np.all(np.isfinite(test["predicted_rv"]))
    assert np.all((test["tail_probability"] >= 0.0))
    assert np.all((test["tail_probability"] <= 1.0))
    assert set(test["information_cutoff"]) == {
        (
            pd.Timestamp(value) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        for value in test["target_date"]
    }


def test_spike_audit_separates_target_day_posthoc_fields():
    target = pd.Timestamp("2023-01-02", tz="UTC")
    article = FilteredArticle(
        cluster_id="a",
        timestamp=target + pd.Timedelta(hours=3),
        title="Bitcoin exchange outage",
        cleaned_text="A cryptocurrency exchange suspended withdrawals.",
        source="source",
        relevance=2,
    )
    article_days = _article_day_summary([article])
    dates = pd.date_range(
        "2023-01-01", "2023-01-02", freq="D", tz="UTC"
    )
    scalars = np.zeros((2, len(DAILY_SCALAR_COLUMNS)))
    scalars[:, DAILY_SCALAR_COLUMNS.index("no_news_dummy")] = 1.0
    features = SimpleNamespace(
        dates=dates,
        daily_scalars=scalars,
        semantic_fast=np.zeros((2, 8)),
        sentiment_fast=np.zeros((2, 3)),
    )
    annotated = pd.DataFrame(
        {
            "target_date": ["2023-01-02"],
            "true_rv": [2.0],
            "predicted_rv": [1.0],
            "spike_threshold": [1.5],
            "is_spike": [True],
            "qlike": [2.0 - np.log(2.0) - 1.0],
        }
    )
    rows = _spike_information_rows(
        FOLD_1,
        annotated,
        features,
        article_days,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["prior_news_count"] == 0
    assert row["target_day_posthoc_news_count"] == 1
    assert row["availability_class"] == (
        "contemporaneous_only_no_t_minus_1_news"
    )
    assert row["target_day_fields_used_by_model"] is False


def test_selection_requires_three_spike_fold_wins_and_normal_guard():
    fold_rows = []
    for index in range(4):
        fold = f"fold_{index + 1}"
        fold_rows.append(
            {
                "fold": fold,
                "model": "har_qlike",
                "mean_qlike": 0.40,
                "normal_qlike": 0.30,
                "spike_qlike": 1.00,
            }
        )
        for name in CONFIGURATION_NAMES:
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "mean_qlike": 0.39,
                    "normal_qlike": 0.302,
                    "spike_qlike": 0.90 if index < 3 else 1.10,
                }
            )
    pooled = [
        {
            "model": "har_qlike",
            "mean_qlike": 0.40,
            "normal_qlike": 0.30,
            "spike_qlike": 1.00,
        }
    ]
    pooled.extend(
        {
            "model": name,
            "mean_qlike": 0.39,
            "normal_qlike": 0.302,
            "spike_qlike": 0.95,
        }
        for name in CONFIGURATION_NAMES
    )
    screen = _selection_screen(
        pd.DataFrame(fold_rows),
        pd.DataFrame(pooled),
        min_delta=1e-5,
    )
    assert all(
        item["passes_predeclared_screen"]
        for item in screen["candidates"].values()
    )
