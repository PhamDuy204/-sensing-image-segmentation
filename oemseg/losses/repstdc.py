"""Official RepSTDC decode/auxiliary loss aggregation."""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn


def _data_samples(targets: Tensor):
    try:
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "RepSTDC loss requires the OpenMMLab environment; run bash scripts/setup_openmmlab_baselines.sh"
        ) from error

    samples = []
    for target in targets:
        sample = SegDataSample()
        sample.gt_sem_seg = PixelData(data=target.long().unsqueeze(0))
        samples.append(sample)
    return samples


def _sum_loss_terms(losses: dict[str, Tensor]) -> Tensor:
    terms = [value for key, value in losses.items() if "loss" in key]
    if not terms:
        raise RuntimeError("RepSTDC decode head returned no loss terms")
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def repstdc_native_loss(
    decode_head: nn.Module,
    auxiliary_heads: Iterable[nn.Module],
    features,
    targets: Tensor,
) -> Tensor:
    """Use MMSeg heads directly so OHEM, CE and boundary Dice match the official config."""
    samples = _data_samples(targets)
    total = _sum_loss_terms(decode_head.loss(features, samples, train_cfg={}))
    for head in auxiliary_heads:
        total = total + _sum_loss_terms(head.loss(features, samples, train_cfg={}))
    return total


def build_repstdc_reporting_loss() -> nn.Module:
    """The published main semantic head uses cross entropy."""
    return nn.CrossEntropyLoss()
