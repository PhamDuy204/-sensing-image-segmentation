import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from oemseg.engine.checkpoint import save_checkpoint
from oemseg.engine.trainer import configure_torch_performance, train_one_epoch, update_validation_state


def test_checkpoint_keeps_legacy_keys(tmp_path: Path):
    model = torch.nn.Conv2d(3, 9, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scheduler, 3, argparse.Namespace(model="unetpp"))
    checkpoint = torch.load(path, weights_only=False)
    assert {"epoch", "model", "optimizer", "scheduler", "args"} <= checkpoint.keys()


def test_checkpoint_accepts_unwrapped_model_state(tmp_path: Path):
    model = torch.nn.Conv2d(3, 9, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "last.pt"
    expected = {"sentinel": torch.tensor([7])}
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        3,
        argparse.Namespace(model="unetpp"),
        model_state_dict=expected,
    )
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["model"]["sentinel"].item() == 7


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def test_accelerator_gradient_accumulation_steps_once_for_two_batches():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Conv2d(3, 9, 1)
    images = torch.randn(2, 3, 8, 8)
    targets = torch.randint(0, 9, (2, 8, 8))
    loader = DataLoader(TensorDataset(images, targets), batch_size=1)
    raw_optimizer = CountingSGD(model.parameters())
    model, raw_optimizer, loader = accelerator.prepare(model, raw_optimizer, loader)
    criterion = torch.nn.CrossEntropyLoss()
    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=raw_optimizer,
        accelerator=accelerator,
        max_batches=None,
    )
    assert raw_optimizer.optimizer.steps == 1


def test_validation_improvement_resets_stale_and_marks_new_best():
    best, stale, improved = update_validation_state(0.61, 0.60, 4)
    assert (best, stale, improved) == (0.61, 0, True)


def test_validation_non_improvement_increments_stale():
    best, stale, improved = update_validation_state(0.59, 0.60, 4)
    assert (best, stale, improved) == (0.60, 5, False)


def test_performance_helper_uses_high_matmul_precision():
    previous = torch.get_float32_matmul_precision()
    try:
        configure_torch_performance(torch.device("cpu"))
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
