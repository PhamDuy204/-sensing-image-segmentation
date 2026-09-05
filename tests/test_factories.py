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


def test_optimizer_honors_explicit_no_weight_decay_parameters():
    decay = torch.nn.Parameter(torch.tensor(1.0))
    no_decay = torch.nn.Parameter(torch.tensor(2.0))
    no_decay._no_weight_decay = True
    optimizer = build_optimizer(
        "adamw",
        [{"params": [decay, no_decay], "lr": 1e-3, "group_name": "main"}],
        weight_decay=0.01,
    )
    groups = {
        tuple(id(parameter) for parameter in group["params"]): group
        for group in optimizer.param_groups
    }
    assert groups[(id(decay),)]["weight_decay"] == 0.01
    assert groups[(id(no_decay),)]["weight_decay"] == 0.0
