"""DataLoader construction for OEM experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oemseg.data.dataset import OEMDataset, read_split, region_for


@dataclass
class LoaderBundle:
    train: DataLoader
    internal_val: DataLoader | None
    test: DataLoader
    train_count: int
    internal_val_count: int
    test_count: int
    train_generator: torch.Generator | None = None


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_train_val(names: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministically split official train names while preserving region coverage."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if not names or val_fraction == 0:
        return list(names), []

    target = max(1, round(len(names) * val_fraction))
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[region_for(name)].append(name)

    regions = sorted(groups)
    cover_all = target >= sum(len(groups[region]) > 1 for region in regions)
    quotas: dict[str, int] = {}
    minimums: dict[str, int] = {}
    desired: dict[str, float] = {}
    for region in regions:
        size = len(groups[region])
        desired[region] = size * val_fraction
        minimums[region] = 1 if cover_all and size > 1 else 0
        max_quota = max(0, size - 1)
        quotas[region] = min(max_quota, max(minimums[region], math.floor(desired[region])))

    while sum(quotas.values()) < target:
        candidates = [region for region in regions if quotas[region] < max(0, len(groups[region]) - 1)]
        if not candidates:
            raise ValueError("validation fraction leaves no training samples")
        region = max(candidates, key=lambda key: (desired[key] - quotas[key], len(groups[key]), key))
        quotas[region] += 1

    while sum(quotas.values()) > target:
        candidates = [region for region in regions if quotas[region] > minimums[region]]
        if not candidates:
            raise ValueError("validation target is smaller than required region coverage")
        region = max(candidates, key=lambda key: (quotas[key] - desired[key], quotas[key], key))
        quotas[region] -= 1

    rng = random.Random(seed)
    val_set: set[str] = set()
    for region in regions:
        candidates = list(groups[region])
        rng.shuffle(candidates)
        val_set.update(candidates[: quotas[region]])

    return [name for name in names if name not in val_set], [name for name in names if name in val_set]


def write_split_manifests(run_dir: Path, loaders: LoaderBundle) -> None:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    entries = {
        "train": loaders.train.dataset.names,
        "val": loaders.internal_val.dataset.names if loaders.internal_val is not None else [],
        "test": loaders.test.dataset.names,
    }
    for split, names in entries.items():
        (split_dir / f"{split}.txt").write_text("\n".join(names) + ("\n" if names else ""))


def build_loaders(args) -> LoaderBundle:
    official_train = read_split(args.data_root, "train")
    test_names = read_split(args.data_root, "val")
    train_names, val_names = split_train_val(official_train, args.internal_val_fraction, args.seed)

    resumable = bool(getattr(args, "stop_after_epoch", None) or getattr(args, "resume_from", None))
    train_generator = torch.Generator().manual_seed(args.seed) if resumable else None
    eval_loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        OEMDataset(args.data_root, train_names, args.size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0 and not resumable,
        generator=train_generator,
        worker_init_fn=seed_worker if resumable else None,
    )
    test_loader = DataLoader(
        OEMDataset(args.data_root, test_names, args.size, return_name=True),
        batch_size=args.eval_batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    internal_val = (
        DataLoader(
            OEMDataset(args.data_root, val_names, args.size, return_name=True),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **eval_loader_kwargs,
        )
        if val_names
        else None
    )
    return LoaderBundle(
        train=train_loader,
        internal_val=internal_val,
        test=test_loader,
        train_count=len(train_names),
        internal_val_count=len(val_names),
        test_count=len(test_names),
        train_generator=train_generator,
    )
