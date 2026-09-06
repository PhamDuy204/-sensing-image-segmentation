"""Shared training engine for all registered OEM segmentation models."""

from __future__ import annotations

import json
import logging
import math
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from oemseg.data.loaders import build_loaders, write_split_manifests
from oemseg.engine.checkpoint import restore_checkpoint, save_checkpoint
from oemseg.engine.error_analysis import write_best_checkpoint_analysis, write_error_analysis
from oemseg.engine.evaluator import evaluate
from oemseg.losses.registry import build_loss
from oemseg.metrics.segmentation import flatten_metrics
from oemseg.models.registry import build_model
from oemseg.optimizers.factory import build_optimizer
from oemseg.schedulers.factory import build_scheduler, evaluation_schedule, should_evaluate
from oemseg.utils.logging import config_dict, format_log_metrics, logger_for
from oemseg.utils.notifications import send_training_email, validate_email_settings
from oemseg.utils.reproducibility import seed_everything
from oemseg.utils.visualization import render_best_checkpoint_visualizations, render_label_legend


def configure_torch_performance(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def configure_distributed_batchnorm(model: nn.Module, world_size: int) -> nn.Module:
    return nn.SyncBatchNorm.convert_sync_batchnorm(model) if world_size > 1 else model


def update_validation_state(
    val_miou: float, best_val_miou: float, stale: int
) -> tuple[float, int, bool]:
    if val_miou > best_val_miou:
        return val_miou, 0, True
    return best_val_miou, stale + 1, False


def update_loss_state(train_loss: float, best_train_loss: float, stale: int) -> tuple[float, int, bool]:
    if train_loss < best_train_loss:
        return train_loss, 0, True
    return best_train_loss, stale + 1, False


def selected_checkpoint(
    has_validation: bool, best_train_epoch: int | None, best_val_epoch: int | None
) -> tuple[str, int, str, str]:
    if has_validation:
        if best_val_epoch is None:
            raise RuntimeError("validation mode completed without a best validation checkpoint")
        return "best_val_miou.pt", best_val_epoch, "validation", "val_miou"
    if best_train_epoch is None:
        raise RuntimeError("training completed without a best training-loss checkpoint")
    return "best_train_loss.pt", best_train_epoch, "train_loss", "train_loss"


def best_artifact_files(run_dir: Path, checkpoint_path: Path) -> list[Path]:
    candidates = [
        checkpoint_path,
        run_dir / "best_checkpoint_summary.json",
        run_dir / "best_checkpoint_val_scores.tsv",
        run_dir / "best_checkpoint_test_scores.tsv",
        run_dir / "below_mean_val.tsv",
        run_dir / "below_mean_test.tsv",
    ]
    return [path for path in candidates if path.exists()]


def flatten_best_metrics(split: str, result) -> dict[str, float]:
    return {
        f"best/{key}": value
        for key, value in flatten_metrics(split, result.loss, result.metrics).items()
    }


def prepare_resume_files(resume_checkpoint: Path, run_dir: Path) -> None:
    previous = resume_checkpoint.parent
    for name in ("metrics.jsonl", "best_train_loss.pt", "best_val_miou.pt"):
        source = previous / name
        if source.exists():
            shutil.copy2(source, run_dir / name)


def training_epoch_window(
    *, total_epochs: int, stop_after_epoch: int | None, resume_epoch: int
) -> tuple[range, bool]:
    end_epoch = min(total_epochs, stop_after_epoch or total_epochs)
    if resume_epoch >= end_epoch:
        raise ValueError(
            f"resume checkpoint already reached epoch {resume_epoch}, chunk ends at {end_epoch}"
        )
    return range(resume_epoch + 1, end_epoch + 1), end_epoch < total_epochs


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    max_batches: int | None = None,
    description: str = "train",
    channels_last: bool = False,
    native_loss: bool = False,
    max_grad_norm: float = 0.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    planned_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    sample_count = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    progress = tqdm(loader, desc=description, leave=False, disable=not accelerator.is_local_main_process)

    for batch_index, (images, targets) in enumerate(progress):
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with accelerator.accumulate(model):
            with accelerator.autocast():
                if native_loss:
                    loss = model(images, targets=targets)
                else:
                    logits = model(images)
            if not native_loss:
                with torch.autocast(device_type=accelerator.device.type, enabled=False):
                    loss = criterion(logits.float(), targets)

            if not torch.isfinite(loss.detach()):
                accelerator.set_trigger()
            if accelerator.check_trigger():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite loss detected at batch {batch_index + 1}: {loss.detach().float().item()}"
                )

            accelerator.backward(loss)
            if max_grad_norm > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.detach().float() * targets.shape[0]
        sample_count += targets.shape[0]
        if batch_index + 1 >= planned_batches:
            break

    if sample_count.item() == 0:
        raise RuntimeError("Training loader produced no batches")
    stats = accelerator.reduce(torch.stack((loss_sum, sample_count)), reduction="sum")
    return float((stats[0] / stats[1].clamp_min(1)).item())


def _create_run_dir(args, accelerator: Accelerator) -> tuple[str, Path]:
    shared: list[object] = [None]
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = args.run_name or f"{args.model}-{args.model_variant}-{timestamp}"
        run_dir = args.output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        shared[0] = (run_name, str(run_dir))
    broadcast_object_list(shared)
    accelerator.wait_for_everyone()
    run_name, run_dir = shared[0]
    return str(run_name), Path(run_dir)


def _log_repro_artifact(wandb, wandb_run, run_name: str, run_dir: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    artifact = wandb.Artifact(f"{run_name}-repro", type="code")
    artifact.add_file(str(root / "train.py"), name="train.py")
    artifact.add_file(str(root / "requirements.txt"), name="requirements.txt")
    artifact.add_file(str(run_dir / "config.json"), name="config.json")
    artifact.add_dir(str(root / "oemseg"), name="oemseg")
    artifact.add_dir(str(root / "scripts"), name="scripts")
    notebook = root / "notebooks/kaggle_multi_gpu.ipynb"
    if notebook.exists():
        artifact.add_file(str(notebook), name="notebooks/kaggle_multi_gpu.ipynb")
    wandb_run.log_artifact(artifact)


def _log_best_model_artifact(
    wandb, wandb_run, run_name: str, run_dir: Path, checkpoint_path: Path, metadata: dict[str, object]
) -> None:
    artifact = wandb.Artifact(f"{run_name}-best-model", type="model", metadata=metadata)
    for path in best_artifact_files(run_dir, checkpoint_path):
        artifact.add_file(str(path), name=path.name)
    visualization_dir = run_dir / "visualizations"
    if visualization_dir.exists():
        artifact.add_dir(str(visualization_dir), name="visualizations")
    wandb_run.log_artifact(artifact)


def run_training(args) -> Path:
    validate_email_settings(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accumulation,
        mixed_precision=args.mixed_precision,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for this experiment")
    configure_torch_performance(accelerator.device)
    seed_everything(args.seed)

    run_name, run_dir = _create_run_dir(args, accelerator)
    logger = logger_for(run_dir) if accelerator.is_main_process else logging.getLogger("oemseg.worker")

    loaders = build_loaders(args)
    model = configure_distributed_batchnorm(build_model(args), accelerator.num_processes)
    native_loss = bool(getattr(model, "uses_native_loss", False))
    configuration = config_dict(args, str(accelerator.device)) | {
        "world_size": accelerator.num_processes,
        "train_count": loaders.train_count,
        "val_count": loaders.internal_val_count,
        "test_count": loaders.test_count,
        "effective_loss": "model_native" if native_loss else args.loss,
    }
    if accelerator.is_main_process:
        (run_dir / "config.json").write_text(json.dumps(configuration, indent=2))
        write_split_manifests(run_dir, loaders)
    accelerator.wait_for_everyone()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = build_loss(args.loss)
    optimizer = build_optimizer(
        args.optimizer,
        model.parameter_groups(base_lr=args.lr, backbone_lr=args.encoder_lr),
        weight_decay=args.weight_decay,
    )
    epochs = 1 if args.smoke else args.epochs
    scheduler = build_scheduler(optimizer, epochs, min(args.warmup_epochs, epochs), args.poly_power)

    if loaders.internal_val is not None:
        model, optimizer, loaders.train, loaders.internal_val, loaders.test, scheduler = accelerator.prepare(
            model, optimizer, loaders.train, loaders.internal_val, loaders.test, scheduler
        )
    else:
        model, optimizer, loaders.train, loaders.test, scheduler = accelerator.prepare(
            model, optimizer, loaders.train, loaders.test, scheduler
        )

    resume_checkpoint = None
    if args.resume_from is not None:
        if accelerator.is_main_process:
            prepare_resume_files(args.resume_from, run_dir)
        accelerator.wait_for_everyone()
        resume_checkpoint = restore_checkpoint(
            args.resume_from,
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            args,
            world_size=accelerator.num_processes,
            map_location=accelerator.device,
            train_generator=loaders.train_generator,
        )
        accelerator_state_dir = args.resume_from.parent / "accelerator_state"
        if accelerator_state_dir.is_dir():
            accelerator.load_state(str(accelerator_state_dir))

    wandb_run = None
    if args.wandb and accelerator.is_main_process:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config=configuration,
            dir=str(run_dir),
        )
        wandb.watch(accelerator.unwrap_model(model), log=None)
        _log_repro_artifact(wandb, wandb_run, run_name, run_dir)

    logger.info(
        "device=%s world_size=%d model=%s variant=%s train=%d val=%d test=%d params=%d",
        accelerator.device,
        accelerator.num_processes,
        args.model,
        args.model_variant,
        loaders.train_count,
        loaders.internal_val_count,
        loaders.test_count,
        parameter_count,
    )

    has_validation = loaders.internal_val is not None
    restored_state = (
        resume_checkpoint.get("training_state", {})
        if isinstance(resume_checkpoint, dict)
        else {}
    )
    best_train_loss = float(restored_state.get("best_train_loss", math.inf))
    best_train_epoch = restored_state.get("best_train_epoch")
    best_val_miou = float(restored_state.get("best_val_miou", -math.inf))
    best_val_epoch = restored_state.get("best_val_epoch")
    best_test_miou = float(restored_state.get("best_test_miou", -math.inf))
    best_test_epoch = restored_state.get("best_test_epoch")
    final_test_miou = float(restored_state.get("final_test_miou", float("nan")))
    train_stale = int(restored_state.get("train_stale", 0))
    val_stale = int(restored_state.get("val_stale", 0))
    resume_epoch = int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint else 0
    epoch_range, is_chunk_boundary = training_epoch_window(
        total_epochs=epochs,
        stop_after_epoch=args.stop_after_epoch,
        resume_epoch=resume_epoch,
    )
    max_batches = 1 if args.smoke else None
    metadata = {
        "format_version": 5,
        "model_name": args.model,
        "model_variant": args.model_variant,
        "world_size": accelerator.num_processes,
        "selection_mode": "validation" if has_validation else "train_loss",
    }

    def training_state() -> dict[str, object]:
        return {
            "best_train_loss": best_train_loss,
            "best_train_epoch": best_train_epoch,
            "best_val_miou": best_val_miou,
            "best_val_epoch": best_val_epoch,
            "best_test_miou": best_test_miou,
            "best_test_epoch": best_test_epoch,
            "final_test_miou": final_test_miou,
            "train_stale": train_stale,
            "val_stale": val_stale,
        }

    def save(name: str, epoch: int) -> None:
        if not accelerator.is_main_process:
            return
        save_checkpoint(
            run_dir / name,
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            epoch,
            args,
            metadata,
            model_state_dict=accelerator.get_state_dict(model),
            training_state=training_state(),
            train_generator=loaders.train_generator,
        )

    if resume_checkpoint is not None:
        logger.info("resumed_from=%s epoch=%d", args.resume_from, resume_epoch)

    metrics_context = (
        (run_dir / "metrics.jsonl").open("a", buffering=1)
        if accelerator.is_main_process
        else nullcontext(None)
    )
    early_stopped = False
    with metrics_context as metrics_file:
        for epoch in epoch_range:
            train_loss = train_one_epoch(
                model=model,
                loader=loaders.train,
                criterion=criterion,
                optimizer=optimizer,
                accelerator=accelerator,
                max_batches=max_batches,
                description=f"epoch {epoch}/{epochs}",
                channels_last=args.channels_last,
                native_loss=native_loss,
                max_grad_norm=args.max_grad_norm,
            )
            scheduler.step()
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "lr": optimizer.param_groups[1]["lr"],
            }
            logger.info(
                "epoch: %d train_loss: %.6f encoder_lr: %.8g lr: %.8g",
                epoch,
                train_loss,
                record["encoder_lr"],
                record["lr"],
            )
            best_train_loss, train_stale, new_best_train = update_loss_state(
                train_loss, best_train_loss, train_stale
            )
            if new_best_train:
                best_train_epoch = epoch
                save("best_train_loss.pt", epoch)

            if args.smoke:
                run_validation = run_test = True
            elif args.eval_start_fraction is not None:
                run_validation = run_test = should_evaluate(
                    epoch, epochs, args.eval_start_fraction, args.eval_every
                )
            else:
                run_validation, run_test = evaluation_schedule(
                    epoch,
                    epochs,
                    args.eval_start_epoch,
                    args.eval_every,
                    args.test_every_validations,
                )

            new_best_val = False
            if run_validation and has_validation:
                val_result = evaluate(
                    model,
                    loaders.internal_val,
                    criterion,
                    accelerator,
                    args.tta_scales,
                    not args.no_tta_flips,
                    max_batches,
                    channels_last=args.channels_last,
                )
                val_record = flatten_metrics("val", val_result.loss, val_result.metrics)
                record.update(val_record)
                logger.info("epoch: %d %s", epoch, format_log_metrics(val_record))
                best_val_miou, val_stale, new_best_val = update_validation_state(
                    val_result.metrics.miou, best_val_miou, val_stale
                )
                if new_best_val:
                    best_val_epoch = epoch
                    save("best_val_miou.pt", epoch)
                if accelerator.is_main_process:
                    write_error_analysis(
                        run_dir,
                        epoch,
                        "val",
                        val_result.samples,
                        args.bad_predict_top_n,
                        new_best_val,
                    )

            if run_test:
                test_result = evaluate(
                    model,
                    loaders.test,
                    criterion,
                    accelerator,
                    args.tta_scales,
                    not args.no_tta_flips,
                    max_batches,
                    channels_last=args.channels_last,
                )
                test_record = flatten_metrics("test", test_result.loss, test_result.metrics)
                record.update(test_record)
                logger.info("epoch: %d %s", epoch, format_log_metrics(test_record))
                final_test_miou = test_result.metrics.miou
                if final_test_miou > best_test_miou:
                    best_test_miou = final_test_miou
                    best_test_epoch = epoch
                if accelerator.is_main_process:
                    write_error_analysis(
                        run_dir,
                        epoch,
                        "test",
                        test_result.samples,
                        args.bad_predict_top_n,
                        new_best_val,
                    )

            save("last.pt", epoch)
            if accelerator.is_main_process:
                metrics_file.write(json.dumps(record) + "\n")
                if wandb_run:
                    wandb_run.log(record, step=epoch)
            stale = val_stale if has_validation else train_stale
            if args.patience > 0 and stale >= args.patience:
                logger.info(
                    "early_stop epoch: %d patience: %d selection_mode: %s",
                    epoch,
                    args.patience,
                    "validation" if has_validation else "train_loss",
                )
                early_stopped = True
                break

    completed_epoch = epoch
    if is_chunk_boundary and not early_stopped:
        accelerator.wait_for_everyone()
        accelerator.save_state(str(run_dir / "accelerator_state"))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            (run_dir / "chunk_state.json").write_text(
                json.dumps(
                    {
                        "epoch": completed_epoch,
                        "total_epochs": epochs,
                        "complete": False,
                        "reason": "chunk_boundary",
                    },
                    indent=2,
                )
            )
        if wandb_run:
            wandb_run.summary.update(
                {"chunk/epoch": completed_epoch, "chunk/complete": False}
            )
            wandb_run.finish()
        accelerator.wait_for_everyone()
        logger.info("chunk_complete epoch=%d output=%s", completed_epoch, run_dir)
        return run_dir

    checkpoint_name, selected_epoch, selection_mode, selection_metric = selected_checkpoint(
        has_validation, best_train_epoch, best_val_epoch
    )
    checkpoint_path = run_dir / checkpoint_name
    accelerator.wait_for_everyone()
    best_checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(best_checkpoint["model"])

    selected_val_result = None
    if has_validation:
        selected_val_result = evaluate(
            model, loaders.internal_val, criterion, accelerator, args.tta_scales,
            not args.no_tta_flips, max_batches, channels_last=args.channels_last,
        )
    selected_test_result = evaluate(
        model, loaders.test, criterion, accelerator, args.tta_scales,
        not args.no_tta_flips, max_batches, channels_last=args.channels_last,
    )

    best_checkpoint_stats: dict[str, object] = {
        "epoch": selected_epoch,
        "selection_mode": selection_mode,
        "selection_metric": selection_metric,
        "best_train_loss": best_train_loss,
        "test": {"loss": selected_test_result.loss, **asdict(selected_test_result.metrics)},
    }
    if selected_val_result is not None:
        best_checkpoint_stats["val"] = {
            "loss": selected_val_result.loss,
            **asdict(selected_val_result.metrics),
        }

    if accelerator.is_main_process:
        if selected_val_result is not None:
            best_checkpoint_stats["val"].update(
                write_best_checkpoint_analysis(run_dir, "val", selected_val_result.samples)
            )
        best_checkpoint_stats["test"].update(
            write_best_checkpoint_analysis(run_dir, "test", selected_test_result.samples)
        )
        (run_dir / "best_checkpoint_summary.json").write_text(
            json.dumps(best_checkpoint_stats, indent=2)
        )
        logger.info(
            "best_checkpoint epoch: %d selection_mode: %s test_miou: %.6f",
            selected_epoch,
            selection_mode,
            selected_test_result.metrics.miou,
        )

    if wandb_run:
        import wandb

        try:
            visualizations = render_best_checkpoint_visualizations(
                accelerator.unwrap_model(model), args, run_dir, accelerator
            )
            legend_path = render_label_legend(run_dir / "visualizations" / "legend.png")
            media = {
                "best_checkpoint/legend": wandb.Image(
                    str(legend_path), caption="OpenEarthMap class-color legend"
                )
            }
            for split, info in visualizations.items():
                names = list(info["names"])
                bad_names = list(info["bad_names"])
                media[f"best_checkpoint/{split}_examples"] = wandb.Image(
                    str(info["path"]),
                    caption=(
                        f"selected epoch={selected_epoch} by {selection_metric}; rows: original / ground truth / prediction; "
                        f"samples={len(names)}; sampled_from_below_mean={len(bad_names)}"
                    ),
                )
            wandb_run.log(media)
        except Exception:
            logger.exception("W&B best-checkpoint visualization failed")
        try:
            _log_best_model_artifact(
                wandb,
                wandb_run,
                run_name,
                run_dir,
                checkpoint_path,
                {
                    "epoch": selected_epoch,
                    "selection_mode": selection_mode,
                    "selection_metric": selection_metric,
                },
            )
        except Exception:
            logger.exception("W&B best-model artifact logging failed")

    if accelerator.is_main_process and args.notify_email:
        try:
            send_training_email(
                args,
                {
                    "run_name": run_name,
                    "best_val_epoch": best_val_epoch,
                    "best_val_miou": best_val_miou,
                    "best_test_epoch": best_test_epoch,
                    "best_test_miou": best_test_miou,
                    "final_test_miou": final_test_miou,
                    "output": str(run_dir),
                },
            )
        except Exception:
            logger.exception("email notification failed")

    if wandb_run:
        summary = {
            "best/epoch": selected_epoch,
            "best/selection_mode": selection_mode,
            "best/selection_metric": selection_metric,
            "best/train_loss": best_train_loss,
            "best_train_loss": best_train_loss,
            "best_train_epoch": best_train_epoch,
            "best_test_miou_observed": best_test_miou,
            "best_test_epoch_observed": best_test_epoch,
            "final_test_miou": final_test_miou,
        }
        if best_val_epoch is not None:
            summary.update({"best_val_miou": best_val_miou, "best_val_epoch": best_val_epoch})
        if selected_val_result is not None:
            summary.update(flatten_best_metrics("val", selected_val_result))
            summary["best/val_sample_mean_miou"] = best_checkpoint_stats["val"]["sample_mean_miou"]
            summary["best/val_below_mean_count"] = best_checkpoint_stats["val"]["below_mean_count"]
        summary.update(flatten_best_metrics("test", selected_test_result))
        summary["best/test_sample_mean_miou"] = best_checkpoint_stats["test"]["sample_mean_miou"]
        summary["best/test_below_mean_count"] = best_checkpoint_stats["test"]["below_mean_count"]
        wandb_run.summary.update(summary)
        wandb_run.finish()
    if accelerator.is_main_process:
        (run_dir / "chunk_state.json").write_text(
            json.dumps(
                {
                    "epoch": completed_epoch,
                    "total_epochs": epochs,
                    "complete": True,
                    "reason": "early_stop" if early_stopped else "epochs_complete",
                },
                indent=2,
            )
        )
    accelerator.wait_for_everyone()
    logger.info("run_complete output: %s", run_dir)
    return run_dir
