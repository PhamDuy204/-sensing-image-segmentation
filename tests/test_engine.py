import argparse
from pathlib import Path

import pytest
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
    previous_precision = torch.get_float32_matmul_precision()
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        configure_torch_performance(torch.device("cuda"))
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.benchmark = previous_benchmark


def test_distributed_batchnorm_syncs_only_for_multiple_processes():
    from oemseg.engine.trainer import configure_distributed_batchnorm

    single = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    multi = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))

    assert isinstance(configure_distributed_batchnorm(single, 1)[1], torch.nn.BatchNorm2d)
    assert isinstance(configure_distributed_batchnorm(multi, 2)[1], torch.nn.SyncBatchNorm)


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


def test_train_one_epoch_can_use_model_native_loss_without_calling_shared_criterion():
    class NativeLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 9, 1)

        def forward(self, images, targets=None):
            logits = self.conv(images)
            if targets is None:
                return logits
            return torch.nn.functional.cross_entropy(logits, targets)

    class ForbiddenCriterion(torch.nn.Module):
        def forward(self, logits, targets):
            raise AssertionError("shared criterion must not be called for native-loss models")

    accelerator = Accelerator(cpu=True)
    model = NativeLossModel()
    loader = DataLoader(
        TensorDataset(torch.randn(1, 3, 8, 8), torch.randint(0, 9, (1, 8, 8))),
        batch_size=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    loss = train_one_epoch(
        model=model,
        loader=loader,
        criterion=ForbiddenCriterion(),
        optimizer=optimizer,
        accelerator=accelerator,
        native_loss=True,
    )
    assert loss > 0


class PrecisionProbeAccelerator:
    device = torch.device("cpu")
    is_local_main_process = True
    sync_gradients = True

    def __init__(self):
        self.clipped = False
        self.triggered = False

    def accumulate(self, model):
        from contextlib import nullcontext
        return nullcontext()

    def autocast(self):
        return torch.autocast("cpu", dtype=torch.bfloat16)

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        self.clipped = True
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, tensor, reduction="sum"):
        return tensor

    def set_trigger(self):
        self.triggered = True

    def check_trigger(self):
        triggered = self.triggered
        self.triggered = False
        return triggered


def test_train_one_epoch_computes_shared_loss_in_float32_under_autocast():
    class ProbeLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = None

        def forward(self, logits, targets):
            self.dtype = logits.dtype
            return torch.nn.functional.cross_entropy(logits, targets)

    accelerator = PrecisionProbeAccelerator()
    model = torch.nn.Conv2d(3, 9, 1)
    loader = DataLoader(
        TensorDataset(torch.randn(1, 3, 8, 8), torch.randint(0, 9, (1, 8, 8))),
        batch_size=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = ProbeLoss()
    train_one_epoch(model, loader, criterion, optimizer, accelerator)
    assert criterion.dtype == torch.float32


def test_train_one_epoch_does_not_clip_gradients_by_default():
    accelerator = PrecisionProbeAccelerator()
    model = torch.nn.Conv2d(3, 9, 1)
    loader = DataLoader(
        TensorDataset(torch.randn(1, 3, 8, 8), torch.randint(0, 9, (1, 8, 8))),
        batch_size=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, loader, torch.nn.CrossEntropyLoss(), optimizer, accelerator)
    assert not accelerator.clipped


def test_train_one_epoch_fails_before_step_on_nonfinite_loss():
    class NaNLoss(torch.nn.Module):
        def forward(self, logits, targets):
            return logits.sum() * torch.tensor(float("nan"))

    accelerator = PrecisionProbeAccelerator()
    model = torch.nn.Conv2d(3, 9, 1)
    loader = DataLoader(
        TensorDataset(torch.randn(1, 3, 8, 8), torch.randint(0, 9, (1, 8, 8))),
        batch_size=1,
    )
    optimizer = CountingSGD(model.parameters())
    try:
        train_one_epoch(model, loader, NaNLoss(), optimizer, accelerator)
    except FloatingPointError as error:
        assert "Non-finite loss" in str(error)
    else:
        raise AssertionError("non-finite loss must stop training")
    assert optimizer.steps == 0


def test_train_one_epoch_clips_gradients_only_when_requested():
    accelerator = PrecisionProbeAccelerator()
    model = torch.nn.Conv2d(3, 9, 1)
    loader = DataLoader(
        TensorDataset(torch.randn(1, 3, 8, 8), torch.randint(0, 9, (1, 8, 8))),
        batch_size=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        accelerator,
        max_grad_norm=0.01,
    )
    assert accelerator.clipped


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path):
    from oemseg.engine.checkpoint import restore_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    path = tmp_path / "last.pt"
    training_state = {
        "best_train_loss": 0.42,
        "best_train_epoch": 7,
        "train_stale": 2,
    }
    args = argparse.Namespace(
        model="unet",
        model_variant="resnet18",
        epochs=45,
        size=1024,
        batch_size=2,
        grad_accumulation=1,
        optimizer="adamw",
        lr=6e-4,
        encoder_lr=6e-5,
        weight_decay=0.01,
        warmup_epochs=5,
        poly_power=0.9,
        loss="ce_dice",
        seed=42,
    )
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        7,
        args,
        metadata={"model_name": "unet", "model_variant": "resnet18", "world_size": 1},
        training_state=training_state,
    )

    with torch.no_grad():
        model.weight.zero_()
    restored = restore_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        args,
        world_size=1,
        map_location="cpu",
    )

    assert torch.all(model.weight == 3.0)
    assert restored["epoch"] == 7
    assert restored["training_state"] == training_state


def test_checkpoint_resume_rejects_changed_training_recipe(tmp_path: Path):
    from oemseg.engine.checkpoint import restore_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    path = tmp_path / "last.pt"
    original = argparse.Namespace(
        model="unet",
        model_variant="resnet18",
        epochs=45,
        size=1024,
        batch_size=2,
        grad_accumulation=1,
        optimizer="adamw",
        lr=6e-4,
        encoder_lr=6e-5,
        weight_decay=0.01,
        warmup_epochs=5,
        poly_power=0.9,
        loss="ce_dice",
        seed=42,
    )
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        3,
        original,
        metadata={"model_name": "unet", "model_variant": "resnet18", "world_size": 1},
    )
    changed = argparse.Namespace(**{**vars(original), "lr": 1e-3})

    with pytest.raises(ValueError, match="lr"):
        restore_checkpoint(path, model, optimizer, scheduler, changed, world_size=1, map_location="cpu")


def test_resume_file_prep_carries_metrics_and_best_checkpoint_forward(tmp_path: Path):
    from oemseg.engine.trainer import prepare_resume_files

    previous = tmp_path / "previous"
    current = tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    checkpoint = previous / "last.pt"
    checkpoint.write_bytes(b"last")
    (previous / "metrics.jsonl").write_text('{"epoch": 1}\n')
    (previous / "best_train_loss.pt").write_bytes(b"best")

    prepare_resume_files(checkpoint, current)

    assert (current / "metrics.jsonl").read_text() == '{"epoch": 1}\n'
    assert (current / "best_train_loss.pt").read_bytes() == b"best"


def test_training_epoch_window_uses_absolute_resume_epoch_and_chunk_end():
    from oemseg.engine.trainer import training_epoch_window

    epochs, is_chunk_boundary = training_epoch_window(
        total_epochs=45,
        stop_after_epoch=30,
        resume_epoch=15,
    )
    assert list(epochs) == list(range(16, 31))
    assert is_chunk_boundary is True

    epochs, is_chunk_boundary = training_epoch_window(
        total_epochs=45,
        stop_after_epoch=45,
        resume_epoch=30,
    )
    assert list(epochs) == list(range(31, 46))
    assert is_chunk_boundary is False

    with pytest.raises(ValueError, match="already reached"):
        training_epoch_window(total_epochs=45, stop_after_epoch=15, resume_epoch=15)


def test_training_accelerator_keeps_pytorch_random_sampler(monkeypatch):
    import oemseg.engine.trainer as trainer

    captured = {}

    class FakeAccelerator:
        device = torch.device("cpu")

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(trainer, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(trainer, "validate_email_settings", lambda args: None)
    args = argparse.Namespace(grad_accumulation=1, mixed_precision="no", seed=42)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        trainer.run_training(args)

    config = captured["dataloader_config"]
    assert config.non_blocking is True
    assert config.use_seedable_sampler is False
    assert config.data_seed is None


def test_checkpoint_round_trip_restores_rng_state(tmp_path: Path):
    import random
    import numpy as np
    from oemseg.engine.checkpoint import restore_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    args = argparse.Namespace(model="unet", model_variant="resnet18", seed=42)
    path = tmp_path / "last.pt"

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        1,
        args,
        metadata={"world_size": 1},
    )
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restore_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        args,
        world_size=1,
        map_location="cpu",
    )
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert actual == pytest.approx(expected)


def test_checkpoint_round_trip_restores_train_generator_state(tmp_path: Path):
    from oemseg.engine.checkpoint import restore_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    args = argparse.Namespace(model="unet", model_variant="resnet18", seed=42)
    path = tmp_path / "last.pt"
    generator = torch.Generator().manual_seed(123)

    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        1,
        args,
        metadata={"world_size": 1},
        train_generator=generator,
    )
    expected = torch.randperm(20, generator=generator)
    generator.manual_seed(999)

    restore_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        args,
        world_size=1,
        map_location="cpu",
        train_generator=generator,
    )
    actual = torch.randperm(20, generator=generator)

    assert torch.equal(actual, expected)
