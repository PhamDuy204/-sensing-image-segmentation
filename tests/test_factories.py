import torch

from oemseg.optimizers.factory import build_optimizer
from oemseg.schedulers.factory import build_scheduler, should_evaluate


def test_optimizer_and_poly_schedule():
    p1 = torch.nn.Parameter(torch.tensor(1.0))
    p2 = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = build_optimizer(
        "adamw",
        [{"params": [p1], "lr": 1e-4}, {"params": [p2], "lr": 1e-3}],
        weight_decay=0.01,
    )
    scheduler = build_scheduler(optimizer, epochs=10, warmup_epochs=2, power=0.9)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert should_evaluate(30, 45, 2 / 3, 3)
    assert should_evaluate(45, 45, 2 / 3, 3)
