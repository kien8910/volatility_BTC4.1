import numpy as np
import torch

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.model import CrossAttentionBlock, build_model
from btc_main_pilot.pipeline import _initialization_assertion


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "fine": torch.randn(batch_size, 7, 168, 12),
        "fine_patch_logrv": torch.randn(batch_size, 168),
        "fine_time": torch.randn(batch_size, 168, 4),
        "coarse": torch.randn(batch_size, 7, 240, 72),
        "coarse_patch_logrv": torch.randn(batch_size, 240),
        "coarse_time": torch.randn(batch_size, 240, 4),
        "semantic_slow": torch.randn(batch_size, 30, 8),
        "semantic_fast": torch.randn(batch_size, 30, 8),
        "sentiment_slow": torch.rand(batch_size, 30, 3),
        "sentiment_fast": torch.randn(batch_size, 30, 3),
        "daily_scalars": torch.randn(batch_size, 30, 11),
        "head_scalars": torch.randn(batch_size, 2),
        "har_scalars": torch.randn(batch_size, 3),
        "true_log_rv": torch.full((batch_size,), -10.0, dtype=torch.float64),
    }


def test_model_budget_shapes_and_unconditional_initialization():
    config = MainPilotConfig()
    core_mean_rv = 2.5e-4
    built = build_model(config, target_mean=-9.0, target_scale=1.5, unconditional_mean_rv=core_mean_rv)
    assert 20_000 <= built.parameter_count <= 60_000
    assert built.forecast_queries == 4
    built.model.eval()
    with torch.no_grad():
        output = built.model(_batch(1))
    assert output["market_attention_output"].shape == (1, 408, 32)
    assert torch.isfinite(output["market_attention_output"]).all()
    assert torch.isfinite(output["predicted_rv"]).all()
    np.testing.assert_allclose(
        output["predicted_rv"].numpy(),
        np.array([core_mean_rv]),
        rtol=1e-6,
    )


def test_forecast_head_stays_float32_inside_autocast():
    config = MainPilotConfig()
    built = build_model(
        config,
        target_mean=-9.0,
        target_scale=1.5,
        unconditional_mean_rv=2.5e-4,
    )
    built.model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        output = built.model(_batch(1))
    assert output["forecast_z"].dtype == torch.float32
    assert output["predicted_log_rv"].dtype == torch.float64
    assert output["predicted_rv"].dtype == torch.float64


def test_locked_spike_diagnostic_variants_forward_and_initialize():
    config = MainPilotConfig()
    counts = {}
    for variant in ("main", "market_only", "hybrid_har"):
        built = build_model(
            config,
            target_mean=-9.0,
            target_scale=1.5,
            unconditional_mean_rv=2.5e-4,
            variant=variant,
        )
        built.model.eval()
        with torch.no_grad():
            output = built.model(_batch(2))
        assert built.variant == variant
        assert 20_000 <= built.parameter_count <= 60_000
        np.testing.assert_allclose(
            output["predicted_rv"].numpy(),
            np.full(2, 2.5e-4),
            rtol=1e-6,
        )
        counts[variant] = built.parameter_count
    assert counts["market_only"] < counts["main"] < counts["hybrid_har"]


def test_all_masked_news_keys_are_rejected_and_null_must_be_unmasked():
    block = CrossAttentionBlock(32, 4, 64, 0.1)
    query = torch.randn(2, 5, 32)
    context = torch.randn(2, 61, 32)
    all_masked = torch.ones(2, 61, dtype=torch.bool)
    try:
        block(query, context, all_masked)
        raise AssertionError("all-masked rows should fail")
    except AssertionError:
        pass
    bad_null = torch.zeros(2, 61, dtype=torch.bool)
    bad_null[:, -1] = True
    try:
        block(query, context, bad_null)
        raise AssertionError("masked null token should fail")
    except AssertionError:
        pass


def test_initialization_assertion_moves_model_before_forward():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("offset", torch.tensor(0.0))
            self.requested_device = None

        def to(self, device):
            self.requested_device = torch.device(device)
            return super().to(device)

        def forward(self, batch):
            assert batch["input"].device == self.weight.device
            return {"predicted_rv": self.weight.reshape(1) + self.offset}

    class OneItemDataset:
        def __getitem__(self, index):
            assert index == 0
            return {"input": torch.tensor([1.0]), "target_date": "2021-01-01"}

    model = RecordingModel()
    _initialization_assertion(
        model,
        OneItemDataset(),
        expected_rv=1.0,
        device=torch.device("cpu"),
    )
    assert model.requested_device == torch.device("cpu")
