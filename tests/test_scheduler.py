import torch

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.training import _scheduler


def test_warmup_starts_at_zero_and_cosine_has_positive_phase():
    config = MainPilotConfig()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = _scheduler(optimizer, config, floor_step=200)
    assert optimizer.param_groups[0]["lr"] == 0.0
    lrs = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[-1] == config.learning_rate
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - config.min_learning_rate) < 1e-15

