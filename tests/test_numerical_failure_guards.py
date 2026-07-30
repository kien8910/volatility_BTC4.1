import json
import logging
from types import SimpleNamespace

import pytest
import torch

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.pipeline import _require_numerically_successful
from btc_main_pilot.training import (
    _recover_amp_overflow,
    _reconstruct_loader_generator_state,
    scheduler_payload,
    train_model,
)
from torch.utils.data import DataLoader


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


def test_amp_overflow_skips_step_reduces_scale_and_clears_gradients():
    class FakeScaler:
        def __init__(self):
            self.scale = 4096.0
            self.step_calls = 0

        def get_scale(self):
            return self.scale

        def step(self, optimizer):
            self.step_calls += 1

        def update(self):
            self.scale /= 2.0

    class FakeOptimizer:
        def __init__(self):
            self.zero_calls = 0

        def zero_grad(self, set_to_none):
            assert set_to_none is True
            self.zero_calls += 1

    scaler = FakeScaler()
    optimizer = FakeOptimizer()
    before, after = _recover_amp_overflow(scaler, optimizer)
    assert (before, after) == (4096.0, 2048.0)
    assert scaler.step_calls == 1
    assert optimizer.zero_calls == 1


def test_unrecoverable_amp_overflow_still_fails():
    class StuckScaler:
        def get_scale(self):
            return 1024.0

        def step(self, optimizer):
            pass

        def update(self):
            pass

    class FakeOptimizer:
        def zero_grad(self, set_to_none):
            pass

    with pytest.raises(FloatingPointError, match="did not reduce"):
        _recover_amp_overflow(StuckScaler(), FakeOptimizer())


def test_resume_reuses_completed_training_without_starting_new_epoch(tmp_path):
    class TinyModel(torch.nn.Module):
        variant = "tiny"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    model = TinyModel()
    saved = TinyModel()
    saved.weight.data.fill_(3.0)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save({"model": saved.state_dict()}, checkpoint / "best.pt")
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "model_variant": "tiny",
                "seed": 11,
                "preprocessor_hash": "prep",
                "scheduler_hash": "scheduler",
                "horizon_epochs": 200,
                "objective": "exact_qlike",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "training_result.json").write_text(
        json.dumps(
            {
                "best_epoch": 17,
                "epochs_run": 37,
                "early_stopped": True,
                "best_validation_qlike": 0.25,
                "history": [],
                "numerical_failure": False,
                "failure_reason": None,
                "training_seconds": 12.0,
                "peak_gpu_memory_bytes": 0,
            }
        ),
        encoding="utf-8",
    )
    config = MainPilotConfig()
    result = train_model(
        model,
        list(range(101)),
        list(range(10)),
        config,
        200,
        checkpoint,
        "prep",
        "scheduler",
        "tiny-run",
        logging.getLogger("resume-test"),
        torch.device("cpu"),
        resume=True,
    )
    assert result.best_epoch == 17
    assert result.early_stopped is True
    assert torch.equal(model.weight.detach(), torch.tensor([3.0]))


def test_legacy_resume_reconstructs_exact_dataloader_rng_state():
    actual_generator = torch.Generator().manual_seed(44)
    actual_loader = DataLoader(
        list(range(17)),
        batch_size=4,
        shuffle=True,
        generator=actual_generator,
    )
    for _ in range(3):
        list(actual_loader)

    replay_generator = torch.Generator().manual_seed(44)
    replay_loader = DataLoader(
        list(range(17)),
        batch_size=4,
        shuffle=True,
        generator=replay_generator,
    )
    _reconstruct_loader_generator_state(
        replay_loader, replay_generator, completed_epochs=3
    )
    assert torch.equal(
        actual_generator.get_state(), replay_generator.get_state()
    )
