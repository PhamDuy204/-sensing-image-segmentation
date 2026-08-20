"""OpenEarthMap dataset reading and transforms."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from oemseg.constants import NUM_CLASSES


def region_for(filename: str) -> str:
    return filename.rsplit("_", 1)[0]


def read_split(root: Path, split: str) -> list[str]:
    split_file = root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    missing = [
        name
        for name in names
        if not (root / region_for(name) / "images" / name).exists()
        or not (root / region_for(name) / "labels" / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{split}: {len(missing)} unpaired samples; examples={missing[:5]}")
    return names


class OEMDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root: Path, names: list[str], size: int = 1024, augment: bool = False, return_name: bool = False):
        self.root = root
        self.names = names
        self.size = size
        self.augment = augment
        self.return_name = return_name

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        name = self.names[index]
        region = region_for(name)
        image = Image.open(self.root / region / "images" / name).convert("RGB")
        mask = Image.open(self.root / region / "labels" / name)
        image = TF.resize(image, [self.size, self.size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
        image_tensor = TF.normalize(
            TF.to_tensor(image),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        minimum, maximum = int(mask_tensor.min()), int(mask_tensor.max())
        if minimum < 0 or maximum >= NUM_CLASSES:
            raise ValueError(f"Invalid label in {name}: [{minimum}, {maximum}]")
        if self.return_name:
            return image_tensor, mask_tensor, name
        return image_tensor, mask_tensor
