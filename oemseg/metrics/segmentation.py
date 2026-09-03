"""Confusion-matrix metrics for multiclass semantic segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from oemseg.constants import CLASS_NAMES, NUM_CLASSES


@dataclass
class SegmentationMetrics:
    oa: float
    miou: float
    f1: float
    precision: float
    recall: float
    per_class_iou: dict[str, float]
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]


def confusion_matrix_tensor(prediction: Tensor, target: Tensor, classes: int = NUM_CLASSES) -> Tensor:
    prediction = prediction.detach().long().flatten()
    target = target.detach().long().flatten()
    valid = (target >= 0) & (target < classes)
    bins = torch.bincount(
        classes * target[valid] + prediction[valid],
        minlength=classes**2,
    )
    return bins.reshape(classes, classes)


def batch_confusion_matrices(prediction: Tensor, target: Tensor, classes: int = NUM_CLASSES) -> Tensor:
    """Return one compact confusion matrix per sample without leaving the active device."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have matching batched shapes")
    batch_size = prediction.shape[0]
    prediction = prediction.detach().long().reshape(batch_size, -1)
    target = target.detach().long().reshape(batch_size, -1)
    valid = (target >= 0) & (target < classes)
    offsets = torch.arange(batch_size, device=target.device).unsqueeze(1) * (classes**2)
    indices = offsets + classes * target + prediction
    bins = torch.bincount(indices[valid], minlength=batch_size * classes**2)
    return bins.reshape(batch_size, classes, classes)


def metrics_from_matrix(matrix: Tensor) -> SegmentationMetrics:
    cm = matrix.to(dtype=torch.float64)
    tp = cm.diag()
    actual, predicted = cm.sum(1), cm.sum(0)
    union = actual + predicted - tp
    precision = torch.where(predicted > 0, tp / predicted, torch.nan)
    recall = torch.where(actual > 0, tp / actual, torch.nan)
    f1 = 2 * precision * recall / (precision + recall)
    iou = torch.where(union > 0, tp / union, torch.nan)

    def mean(values: Tensor) -> float:
        return float(torch.nanmean(values).item())

    def by_class(values: Tensor) -> dict[str, float]:
        return {name: float(value) for name, value in zip(CLASS_NAMES[: matrix.shape[0]], values.tolist())}

    return SegmentationMetrics(
        oa=float((tp.sum() / cm.sum().clamp_min(1)).item()),
        miou=mean(iou),
        f1=mean(f1),
        precision=mean(precision),
        recall=mean(recall),
        per_class_iou=by_class(iou),
        per_class_f1=by_class(f1),
        per_class_precision=by_class(precision),
        per_class_recall=by_class(recall),
    )


class ConfusionMatrix:
    def __init__(self, classes: int = NUM_CLASSES, device: torch.device | None = None):
        self.classes = classes
        self.matrix = torch.zeros((classes, classes), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        if self.matrix.device != prediction.device:
            self.matrix = self.matrix.to(prediction.device)
        self.matrix += confusion_matrix_tensor(prediction, target, self.classes)

    def compute(self) -> SegmentationMetrics:
        return metrics_from_matrix(self.matrix)


def flatten_metrics(prefix: str, loss: float, metrics: SegmentationMetrics) -> dict[str, float]:
    values = asdict(metrics)
    result = {f"{prefix}_loss": loss}
    for key in ("oa", "miou", "f1", "precision", "recall"):
        result[f"{prefix}_{key}"] = values[key]
    for metric in ("iou", "f1", "precision", "recall"):
        result.update(
            {
                f"{prefix}_{metric}_{class_name}": value
                for class_name, value in values[f"per_class_{metric}"].items()
            }
        )
    return result
