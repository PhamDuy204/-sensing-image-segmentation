"""Model-independent segmentation evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from oemseg.data.dataset import region_for
from oemseg.metrics.segmentation import SegmentationMetrics, batch_confusion_matrices, metrics_from_matrix
from oemseg.utils.tta import model_logits


@dataclass
class EvaluationResult:
    loss: float
    metrics: SegmentationMetrics
    samples: list[dict[str, object]]

    def __iter__(self):
        # Backward compatible with the previous ``loss, metrics = evaluate(...)`` API.
        yield self.loss
        yield self.metrics


def _worst_class(metrics: SegmentationMetrics) -> tuple[str, float]:
    valid = [(name, value) for name, value in metrics.per_class_iou.items() if not math.isnan(value)]
    return min(valid, key=lambda item: item[1]) if valid else ("n/a", math.nan)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
    scales: list[float],
    flips: bool,
    max_batches: int | None = None,
    channels_last: bool = False,
) -> EvaluationResult:
    model.eval()
    gathered_matrices: list[torch.Tensor] = []
    gathered_losses: list[torch.Tensor] = []
    gathered_names: list[str] = []

    progress = tqdm(loader, desc="evaluate", leave=False, disable=not accelerator.is_local_main_process)
    with torch.inference_mode():
        for batch_index, batch in enumerate(progress):
            if len(batch) == 3:
                images, targets, names = batch
            else:
                images, targets = batch
                names = None
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with accelerator.autocast():
                logits = model_logits(model, images, scales, flips)
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                float_logits = logits.float()
                batch_loss = criterion(float_logits, targets)
                if names is None:
                    sample_losses = batch_loss.detach().repeat(targets.shape[0])
                else:
                    sample_losses = torch.stack(
                        [
                            criterion(float_logits[index : index + 1], targets[index : index + 1])
                            for index in range(targets.shape[0])
                        ]
                    ).detach()

            sample_matrices = batch_confusion_matrices(logits.argmax(1), targets)
            gathered_matrices.append(accelerator.gather_for_metrics(sample_matrices).cpu())
            gathered_losses.append(accelerator.gather_for_metrics(sample_losses).float().cpu())
            if names is not None:
                gathered_names.extend(
                    str(name)
                    for name in accelerator.gather_for_metrics(list(names), use_gather_object=True)
                )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    if not gathered_losses:
        raise RuntimeError("Evaluation loader produced no batches")

    matrices = torch.cat(gathered_matrices)
    losses = torch.cat(gathered_losses)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("Non-finite evaluation loss detected")
    metrics = metrics_from_matrix(matrices.sum(0))
    samples: list[dict[str, object]] = []
    if accelerator.is_main_process and gathered_names:
        # Accelerate trims padded duplicates in the final distributed evaluation batch.
        for name, sample_matrix, sample_loss in zip(gathered_names, matrices, losses):
            sample_metrics = metrics_from_matrix(sample_matrix)
            worst_class, worst_class_iou = _worst_class(sample_metrics)
            samples.append(
                {
                    "filename": name,
                    "region": region_for(name),
                    "loss": float(sample_loss.item()),
                    "oa": sample_metrics.oa,
                    "miou": sample_metrics.miou,
                    "worst_class": worst_class,
                    "worst_class_iou": worst_class_iou,
                }
            )
    return EvaluationResult(float(losses.mean().item()), metrics, samples)
