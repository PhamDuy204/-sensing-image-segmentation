"""Shared model adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class SegmentationModelAdapter(nn.Module, ABC):
    """A segmentation model that returns full-resolution logits."""

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError

    def parameter_groups(self, base_lr: float, backbone_lr: float) -> list[dict[str, object]]:
        backbone = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        backbone_ids = {id(parameter) for parameter in backbone}
        main = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        main_ids = {id(parameter) for parameter in main}
        if not backbone or not main or backbone_ids & main_ids:
            raise RuntimeError("Model parameter groups must be nonempty and nonoverlapping")
        return [
            {"params": backbone, "lr": backbone_lr, "group_name": "backbone"},
            {"params": main, "lr": base_lr, "group_name": "main"},
        ]
