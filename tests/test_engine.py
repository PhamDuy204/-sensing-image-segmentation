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


def test_train_loss_improvement_resets_stale_and_marks_new_best():
    from oemseg.engine.trainer import update_loss_state

    best, stale, improved = update_loss_state(0.49, 0.50, 3)
    assert (best, stale, improved) == (0.49, 0, True)


def test_train_loss_non_improvement_increments_stale():
    from oemseg.engine.trainer import update_loss_state

    best, stale, improved = update_loss_state(0.51, 0.50, 3)
    assert (best, stale, improved) == (0.50, 4, False)


def test_checkpoint_selection_uses_val_when_present_and_train_loss_without_val():
    from oemseg.engine.trainer import selected_checkpoint

    assert selected_checkpoint(True, best_train_epoch=4, best_val_epoch=7) == (
        "best_val_miou.pt", 7, "validation", "val_miou"
    )
    assert selected_checkpoint(False, best_train_epoch=4, best_val_epoch=None) == (
        "best_train_loss.pt", 4, "train_loss", "train_loss"
    )


def test_best_model_artifact_files_include_checkpoint_and_below_mean_logs(tmp_path: Path):
    from oemseg.engine.trainer import best_artifact_files

    for name in ("best_train_loss.pt", "best_checkpoint_summary.json", "below_mean_test.tsv"):
        (tmp_path / name).write_text("x")
    files = {path.name for path in best_artifact_files(tmp_path, tmp_path / "best_train_loss.pt")}
    assert files == {"best_train_loss.pt", "best_checkpoint_summary.json", "below_mean_test.tsv"}

    (tmp_path / "below_mean_val.tsv").write_text("x")
    files = {path.name for path in best_artifact_files(tmp_path, tmp_path / "best_train_loss.pt")}
    assert "below_mean_val.tsv" in files


def test_flatten_best_metrics_keeps_all_selected_test_scores():
    from oemseg.engine.evaluator import EvaluationResult
    from oemseg.engine.trainer import flatten_best_metrics
    from oemseg.metrics.segmentation import ConfusionMatrix

    target = torch.tensor([[[0, 1], [1, 1]]])
    prediction = torch.tensor([[[0, 1], [0, 1]]])
    matrix = ConfusionMatrix(classes=2)
    matrix.update(prediction, target)
    result = EvaluationResult(loss=0.4, metrics=matrix.compute(), samples=[])
    summary = flatten_best_metrics("test", result)

    for key in (
        "best/test_loss",
        "best/test_oa",
        "best/test_miou",
        "best/test_f1",
        "best/test_precision",
        "best/test_recall",
        "best/test_iou_background",
        "best/test_f1_background",
        "best/test_precision_background",
        "best/test_recall_background",
    ):
        assert key in summary
