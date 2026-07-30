from dataclasses import replace

import pandas as pd

from btc_main_pilot.config import (
    MainPilotConfig,
    REGIME_ANCHOR_VARIANTS,
    SPIKE_DIAGNOSTIC_FOLDS,
)
from btc_main_pilot.regime_anchor_diagnostic import _anchor_selection


def test_regime_anchor_scope_is_exactly_four_by_two():
    assert [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    ]
    assert REGIME_ANCHOR_VARIANTS == (
        "har_anchor_market",
        "har_anchor_market_text",
    )
    assert len(SPIKE_DIAGNOSTIC_FOLDS) * len(REGIME_ANCHOR_VARIANTS) == 8
    assert all(
        fold.test_end <= "2023-10-19" for fold in SPIKE_DIAGNOSTIC_FOLDS
    )
    config = replace(
        MainPilotConfig(),
        profile="development-regime-anchor-diagnostic",
        output_dir="outputs/regime_anchor_diagnostic",
    )
    config.validate()


def test_anchor_screen_separates_har_improvement_from_text_contribution():
    rows = []
    for fold in ("fold_1", "fold_2", "fold_3", "fold_4"):
        rows.extend(
            [
                {
                    "fold": fold,
                    "model": "har_qlike",
                    "mean_qlike": 0.40,
                    "spike_qlike": 1.00,
                    "correction_selected": None,
                },
                {
                    "fold": fold,
                    "model": "har_anchor_market",
                    "mean_qlike": 0.35 if fold != "fold_4" else 0.45,
                    "spike_qlike": 0.90 if fold != "fold_4" else 1.10,
                    "correction_selected": fold != "fold_4",
                },
                {
                    "fold": fold,
                    "model": "har_anchor_market_text",
                    "mean_qlike": 0.34,
                    "spike_qlike": 0.80,
                    "correction_selected": True,
                },
            ]
        )
    pooled = pd.DataFrame(
        [
            {"model": "har_qlike", "mean_qlike": 0.40, "spike_qlike": 1.00},
            {
                "model": "har_anchor_market",
                "mean_qlike": 0.37,
                "spike_qlike": 0.92,
            },
            {
                "model": "har_anchor_market_text",
                "mean_qlike": 0.34,
                "spike_qlike": 0.80,
            },
        ]
    )
    result = _anchor_selection(pd.DataFrame(rows), pooled)
    assert result["har_anchor_candidates"]["har_anchor_market"][
        "passes_predeclared_screen"
    ]
    assert result["har_anchor_candidates"]["har_anchor_market_text"][
        "passes_predeclared_screen"
    ]
    assert result["incremental_text_contribution"][
        "passes_predeclared_screen"
    ]
