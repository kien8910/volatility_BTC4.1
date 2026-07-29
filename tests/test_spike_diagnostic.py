from dataclasses import replace

import pandas as pd

from btc_main_pilot.config import (
    MainPilotConfig,
    SPIKE_DIAGNOSTIC_FOLDS,
    SPIKE_DIAGNOSTIC_VARIANTS,
)
from btc_main_pilot.spike_diagnostic import (
    _annotate_predictions,
    _selection_diagnostic,
)


def test_development_spike_diagnostic_scope_is_exactly_four_by_three():
    assert [fold.name for fold in SPIKE_DIAGNOSTIC_FOLDS] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    ]
    assert SPIKE_DIAGNOSTIC_VARIANTS == (
        "main",
        "market_only",
        "hybrid_har",
    )
    assert len(SPIKE_DIAGNOSTIC_FOLDS) * len(SPIKE_DIAGNOSTIC_VARIANTS) == 12
    assert all(fold.test_end <= "2023-10-19" for fold in SPIKE_DIAGNOSTIC_FOLDS)
    config = replace(
        MainPilotConfig(),
        profile="development-spike-diagnostic",
        output_dir="outputs/spike_diagnostic",
    )
    config.validate()


def test_prediction_annotation_uses_fold_core_p90_threshold():
    frame = pd.DataFrame(
        {
            "target_date": ["2022-01-01", "2022-01-02"],
            "true_rv": [1.0, 3.0],
            "true_log_rv": [0.0, 1.0986122886681098],
            "predicted_rv": [2.5, 1.5],
            "predicted_log_rv": [0.9162907318741551, 0.4054651081081644],
        }
    )
    annotated = _annotate_predictions(frame, "fold_1", "main", 2.0)
    assert annotated["is_spike"].tolist() == [False, True]
    assert annotated["predicted_spike"].tolist() == [True, False]
    assert (annotated["fold"] == "fold_1").all()
    assert (annotated["model"] == "main").all()
    assert annotated["qlike"].notna().all()


def test_predeclared_screen_requires_three_fold_spike_wins_and_no_overall_cost():
    rows = []
    for fold in ("fold_1", "fold_2", "fold_3", "fold_4"):
        rows.append({"fold": fold, "model": "main", "spike_qlike": 1.0})
        rows.append(
            {
                "fold": fold,
                "model": "market_only",
                "spike_qlike": 0.9 if fold != "fold_4" else 1.1,
            }
        )
        rows.append(
            {"fold": fold, "model": "hybrid_har", "spike_qlike": 0.8}
        )
    pooled = pd.DataFrame(
        [
            {"model": "main", "mean_qlike": 0.40, "spike_qlike": 1.0},
            {
                "model": "market_only",
                "mean_qlike": 0.39,
                "spike_qlike": 0.9,
            },
            {
                "model": "hybrid_har",
                "mean_qlike": 0.41,
                "spike_qlike": 0.8,
            },
        ]
    )
    result = _selection_diagnostic(pd.DataFrame(rows), pooled)
    assert result["candidates"]["market_only"]["passes_predeclared_screen"]
    assert not result["candidates"]["hybrid_har"]["passes_predeclared_screen"]
