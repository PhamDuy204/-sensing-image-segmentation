"""Compact UPerNet decoder for four-stage hierarchical backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SafeBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm that safely handles a single value per channel in PPM pools."""

    def forward(self, inputs: Tensor) -> Tensor:
        values_per_channel = inputs.numel() // inputs.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(inputs)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            SafeBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvNormAct(in_channels, channels, 1),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = ConvNormAct(
            in_channels + len(pool_scales) * channels,
            channels,
            3,
            padding=1,
        )

    def forward(self, feature: Tensor) -> Tensor:
        size = feature.shape[-2:]
        pooled = [
            F.interpolate(branch(feature), size=size, mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        return self.bottleneck(torch.cat([feature, *pooled], dim=1))


class UPerNetHead(nn.Module):
    """PSP + top-down FPN decoder used by UPerNet."""

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 512,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        num_classes: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("UPerNetHead requires exactly four feature stages")
        self.in_channels = tuple(in_channels)
        self.lateral_convs = nn.ModuleList(
            [ConvNormAct(value, channels, 1) for value in self.in_channels[:-1]]
        )
        self.psp = PyramidPoolingModule(self.in_channels[-1], channels, pool_scales)
        self.fpn_convs = nn.ModuleList(
            [ConvNormAct(channels, channels, 3, padding=1) for _ in self.in_channels[:-1]]
        )
        self.fpn_bottleneck = ConvNormAct(len(self.in_channels) * channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected four feature maps, received {len(features)}")
        actual_channels = [feature.shape[1] for feature in features]
        if actual_channels != list(self.in_channels):
            raise ValueError(
                f"UPerNet feature channels must be {list(self.in_channels)}, got {actual_channels}"
            )

        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features[:-1])]
        laterals.append(self.psp(features[-1]))
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outputs = [
            conv(feature) for conv, feature in zip(self.fpn_convs, laterals[:-1])
        ]
        outputs.append(laterals[-1])
        target_size = outputs[0].shape[-2:]
        outputs = [
            output
            if output.shape[-2:] == target_size
            else F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
            for output in outputs
        ]
        fused = self.fpn_bottleneck(torch.cat(outputs, dim=1))
        return self.classifier(self.dropout(fused))
