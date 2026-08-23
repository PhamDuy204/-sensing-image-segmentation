"""Shared training engine for all registered OEM segmentation models."""

from __future__ import annotations

import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from oemseg.data.loaders import build_loaders, write_split_manifests
from oemseg.engine.checkpoint import save_checkpoint
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
from oemseg.utils.visualization import render_best_checkpoint_visualizations


def configure_torch_performance(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def update_validation_state(
    val_miou: float, best_val_miou: float, stale: int
) -> tuple[float, int, bool]:
    if val_miou > best_val_miou:
        return val_miou, 0, True
    return best_val_miou, stale + 1, False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    max_batches: int | None = None,
    description: str = "train",
    channels_last: bool = False,
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
                loss = criterion(model(images), targets)
            accelerator.backward(loss)
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


def _log_analysis_artifact(wandb, wandb_run, run_name: str, run_dir: Path) -> None:
    artifact = wandb.Artifact(f"{run_name}-analysis", type="analysis")
    for filename in (
        "metrics.jsonl",
        "sample_scores.jsonl",
        "bad_predictions_val.tsv",
        "bad_predictions_test.tsv",
        "bad_predictions_val_best.tsv",
        "bad_predictions_test_at_best_val.tsv",
        "best_checkpoint_summary.json",
        "best_checkpoint_val_scores.tsv",
        "best_checkpoint_test_scores.tsv",
        "below_mean_val.tsv",
        "below_mean_test.tsv",
    ):
        path = run_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=filename)
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
        dataloader_config=DataLoaderConfiguration(
            non_blocking=True,
            use_seedable_sampler=True,
            data_seed=args.seed,
        ),
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for this experiment")
    configure_torch_performance(accelerator.device)
    seed_everything(args.seed)

    run_name, run_dir = _create_run_dir(args, accelerator)
    logger = logger_for(run_dir) if accelerator.is_main_process else logging.getLogger("oemseg.worker")

    loaders = build_loaders(args)
    configuration = config_dict(args, str(accelerator.device)) | {
        "world_size": accelerator.num_processes,
        "train_count": loaders.train_count,
        "val_count": loaders.internal_val_count,
        "test_count": loaders.test_count,
    }
    if accelerator.is_main_process:
        (run_dir / "config.json").write_text(json.dumps(configuration, indent=2))
        write_split_manifests(run_dir, loaders)
    accelerator.wait_for_everyone()

    model = build_model(args)
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
        )

    best_train_loss = math.inf
    best_val_miou = -math.inf
    best_val_epoch = None
    best_test_miou = -math.inf
    best_test_epoch = None
    final_test_miou = float("nan")
    stale = 0
    max_batches = 1 if args.smoke else None
    metadata = {
        "format_version": 3,
        "model_name": args.model,
        "model_variant": args.model_variant,
        "world_size": accelerator.num_processes,
    }
    metrics_context = (
        (run_dir / "metrics.jsonl").open("a", buffering=1)
        if accelerator.is_main_process
        else nullcontext(None)
    )
    with metrics_context as metrics_file:
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=loaders.train,
                criterion=criterion,
                optimizer=optimizer,
                accelerator=accelerator,
                max_batches=max_batches,
                description=f"epoch {epoch}/{epochs}",
                channels_last=args.channels_last,
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
            if train_loss < best_train_loss:
                best_train_loss = train_loss
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
            if run_validation and loaders.internal_val is not None:
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
                best_val_miou, stale, new_best_val = update_validation_state(
                    val_result.metrics.miou, best_val_miou, stale
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
            if loaders.internal_val is not None and args.patience > 0 and stale >= args.patience:
                logger.info("early_stop epoch: %d patience: %d", epoch, args.patience)
                break

    best_checkpoint_stats: dict[str, object] = {}
    if best_val_epoch is not None:
        accelerator.wait_for_everyone()
        best_checkpoint = torch.load(
            run_dir / "best_val_miou.pt",
            map_location=accelerator.device,
            weights_only=False,
        )
        best_model = accelerator.unwrap_model(model)
        best_model.load_state_dict(best_checkpoint["model"])
        best_val_result = evaluate(
            model, loaders.internal_val, criterion, accelerator, args.tta_scales,
            not args.no_tta_flips, max_batches, channels_last=args.channels_last,
        )
        best_test_result = evaluate(
            model, loaders.test, criterion, accelerator, args.tta_scales,
            not args.no_tta_flips, max_batches, channels_last=args.channels_last,
        )
        if accelerator.is_main_process:
            val_analysis = write_best_checkpoint_analysis(run_dir, "val", best_val_result.samples)
            test_analysis = write_best_checkpoint_analysis(run_dir, "test", best_test_result.samples)
            best_checkpoint_stats = {
                "epoch": best_val_epoch,
                "val": {"loss": best_val_result.loss, "miou": best_val_result.metrics.miou, **val_analysis},
                "test": {"loss": best_test_result.loss, "miou": best_test_result.metrics.miou, **test_analysis},
            }
            (run_dir / "best_checkpoint_summary.json").write_text(
                json.dumps(best_checkpoint_stats, indent=2)
            )
            logger.info(
                "best_checkpoint epoch: %d val_miou: %.6f test_miou: %.6f",
                best_val_epoch, best_val_result.metrics.miou, best_test_result.metrics.miou,
            )

    if wandb_run and best_val_epoch is not None:
        import wandb

        try:
            visualizations = render_best_checkpoint_visualizations(
                accelerator.unwrap_model(model), args, run_dir, accelerator
            )
            media = {}
            for split, info in visualizations.items():
                names = list(info["names"])
                bad_names = list(info["bad_names"])
                media[f"best_checkpoint/{split}_examples"] = wandb.Image(
                    str(info["path"]),
                    caption=(
                        f"best validation epoch={best_val_epoch}; rows: original / ground truth / prediction; "
                        f"samples={len(names)}; sampled_from_below_mean={len(bad_names)}"
                    ),
                )
            if media:
                wandb_run.log(media)
        except Exception:
            logger.exception("W&B best-checkpoint visualization failed")
        try:
            _log_analysis_artifact(wandb, wandb_run, run_name, run_dir)
        except Exception:
            logger.exception("W&B analysis artifact logging failed")

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
            "best_val_miou": best_val_miou,
            "best_val_epoch": best_val_epoch,
            "best_test_miou_observed": best_test_miou,
            "best_test_epoch_observed": best_test_epoch,
            "final_test_miou": final_test_miou,
        }
        if best_val_epoch is not None:
            summary.update({"best/epoch": best_val_epoch, "best/val_miou": best_val_miou})
        if best_checkpoint_stats:
            val_stats = best_checkpoint_stats["val"]
            test_stats = best_checkpoint_stats["test"]
            summary.update(
                {
                    "best/epoch": best_val_epoch,
                    "best/val_miou": val_stats["miou"],
                    "best/test_miou": test_stats["miou"],
                    "best/val_sample_mean_miou": val_stats["sample_mean_miou"],
                    "best/test_sample_mean_miou": test_stats["sample_mean_miou"],
                    "best/val_below_mean_count": val_stats["below_mean_count"],
                    "best/test_below_mean_count": test_stats["below_mean_count"],
                }
            )
        wandb_run.summary.update(summary)
        wandb_run.finish()
    accelerator.wait_for_everyone()
    logger.info("run_complete output: %s", run_dir)
    return run_dir
