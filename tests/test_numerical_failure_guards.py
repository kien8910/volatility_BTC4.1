from types import SimpleNamespace

import pytest

from btc_main_pilot.pipeline import _require_numerically_successful
from btc_main_pilot.training import scheduler_payload


def test_numerical_failure_blocks_downstream_artifacts():
    result = SimpleNamespace(
        numerical_failure=True,
        failure_reason="Gradient contains NaN/Inf",
    )
    with pytest.raises(RuntimeError, match="no scheduler lock"):
        _require_numerically_successful(result, "Fold 1 scheduler pilot")


def test_successful_training_passes_failure_guard():
    result = SimpleNamespace(
        numerical_failure=False,
        failure_reason=None,
    )
    _require_numerically_successful(result, "Fold 5 training")


def test_scheduler_payload_certifies_successful_pilot():
    payload = scheduler_payload(
        e_pilot=31,
        h_cos=35,
        config_hash="abc",
        pilot_early_stopped=True,
    )
    assert payload["pilot_completed_without_numerical_failure"] is True
    assert payload["pilot_early_stopped"] is True
