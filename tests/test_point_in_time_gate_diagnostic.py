from types import SimpleNamespace

import numpy as np
import pandas as pd

from btc_main_pilot.config import SPIKE_DIAGNOSTIC_FOLDS
from btc_main_pilot.event_aware_longtext_audit import EventAwarePolicy
from btc_main_pilot.point_in_time_gate_diagnostic import (
    DEFERRED_CONTEXT_FAMILIES,
    GATE_NAMES,
    GATE_SPECS,
    REFINED_CONTEXT_FAMILIES,
    GateSpec,
    _gate_feature_row,
    _market_gate_features,
    _route_frame,
    refined_event_decision,
    select_and_apply_gate,
)


def _policy() -> EventAwarePolicy:
    return EventAwarePolicy(
        enabled_context_families=REFINED_CONTEXT_FAMILIES,
        relevant_rate_threshold=0.5,
        minimum_family_examples=10,
        training_rows=829,
        holdout_rows=366,
        family_statistics=tuple(),
        label_source="test",
    )


def _frame(true_rv: list[float], predicted_rv: list[float]) -> pd.DataFrame:
    true = np.asarray(true_rv, dtype=np.float64)
    predicted = np.asarray(predicted_rv, dtype=np.float64)
    anchor = np.zeros(len(true), dtype=np.float64)
    return pd.DataFrame(
        {
            "target_date": [
                f"2020-01-{index + 1:02d}" for index in range(len(true))
            ],
            "true_rv": true,
            "true_log_rv": np.log(true),
            "har_anchor_log_rv": anchor,
            "delta_log_rv": np.log(predicted) - anchor,
            "predicted_rv": predicted,
            "predicted_log_rv": np.log(predicted),
        }
    )


def test_refined_filter_keeps_strong_families_and_defers_noisy_ones():
    policy = _policy()
    keep, _, reason, _ = refined_event_decision(
        "SEC approves a crypto ETF",
        "The regulator issued a final approval.",
        policy,
    )
    assert keep
    assert reason == "context_event:regulation_etf"

    keep, _, reason, _ = refined_event_decision(
        "Ethereum protocol hack",
        "A crypto protocol suffered a security exploit.",
        policy,
    )
    assert not keep
    assert reason == "no_enabled_event:security_hack"
    assert "security_hack" in DEFERRED_CONTEXT_FAMILIES


def test_refined_filter_repairs_only_concrete_direct_bitcoin_events():
    policy = _policy()
    keep, _, reason, _ = refined_event_decision(
        "Protocol update",
        "A Bitcoin protocol upgrade was launched by its maintainers.",
        policy,
    )
    assert keep
    assert reason == "direct_bitcoin_concrete_repair"

    keep, _, _, _ = refined_event_decision(
        "Weekly technology links",
        "The newsletter briefly mentions Bitcoin among many topics.",
        policy,
    )
    assert not keep


def test_market_gate_features_do_not_read_target_or_future_logrv():
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    original = np.linspace(-10.0, -5.0, len(dates))
    market = SimpleNamespace(
        log_rv=original.copy(),
        date_to_index={date: index for index, date in enumerate(dates)},
    )
    target = dates[25]
    expected = _market_gate_features(market, target)
    market.log_rv[25:] = 999.0
    actual = _market_gate_features(market, target)
    np.testing.assert_allclose(actual, expected)


def test_gate_news_features_use_only_target_minus_one_day():
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    market = SimpleNamespace(
        log_rv=np.linspace(-10.0, -5.0, len(dates)),
        date_to_index={date: index for index, date in enumerate(dates)},
    )
    title = SimpleNamespace(
        dates=dates,
        semantic_fast=np.arange(30 * 8, dtype=float).reshape(30, 8),
    )
    lead = SimpleNamespace(
        dates=dates,
        daily_scalars=np.arange(30 * 11, dtype=float).reshape(30, 11),
        sentiment_fast=np.arange(30 * 3, dtype=float).reshape(30, 3),
    )
    features = {"title": title, "lead": lead}
    target = dates[25]
    expected = _gate_feature_row(market, features, target)
    title.semantic_fast[25:] = 1e9
    lead.daily_scalars[25:] = 1e9
    lead.sentiment_fast[25:] = 1e9
    actual = _gate_feature_row(market, features, target)
    np.testing.assert_allclose(actual, expected)


def test_route_frame_hard_switch_and_rejected_fallback():
    normal = _frame([1.0, 4.0], [1.0, 1.0])
    spike = _frame([1.0, 4.0], [2.0, 2.0])
    probability = np.asarray([0.1, 0.9])
    selected = _route_frame(
        normal, spike, probability, "hard", 0.5, selected=True
    )
    np.testing.assert_allclose(selected["predicted_rv"], [1.0, 2.0])
    rejected = _route_frame(
        normal, spike, probability, "hard", 0.5, selected=False
    )
    np.testing.assert_allclose(rejected["predicted_rv"], [1.0, 1.0])
    np.testing.assert_allclose(rejected["gate_weight"], [0.0, 0.0])


def test_gate_selection_prefers_conservative_threshold_inside_tie():
    validation = {
        "har_qlike": _frame([1.0, 4.0], [1.0, 1.0]),
        "title_only_pca8_16": _frame([1.0, 4.0], [2.0, 2.0]),
    }
    test = {
        "har_qlike": _frame([1.0, 4.0], [1.0, 1.0]),
        "title_only_pca8_16": _frame([1.0, 4.0], [2.0, 2.0]),
    }
    _, routed, metadata, grid = select_and_apply_gate(
        GateSpec(
            "test_gate",
            "hard",
            "har_qlike",
            "title_only_pca8_16",
        ),
        validation,
        test,
        np.asarray([0.1, 0.9]),
        np.asarray([0.1, 0.9]),
        min_delta=1e-5,
    )
    assert metadata["correction_selected"]
    assert metadata["selected_gate_threshold"] == 0.5
    assert len(grid) == 4
    np.testing.assert_allclose(routed["predicted_rv"], [1.0, 2.0])


def test_gate_scope_is_fold_1_to_4_only():
    assert [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    ]
    assert all(fold.test_end <= "2023-10-19" for fold in SPIKE_DIAGNOSTIC_FOLDS)
    assert len(GATE_NAMES) == len(GATE_SPECS) == 5
