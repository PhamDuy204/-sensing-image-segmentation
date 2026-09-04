"""Pinned official repositories and commands for paper-specific model stacks."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / ".vendor"


@dataclass(frozen=True)
class UpstreamSpec:
    name: str
    repo: str
    revision: str
    dest: Path
    sparse_paths: tuple[str, ...]
    train_args: tuple[str, ...]
    eval_args: tuple[str, ...]
    checkpoint_flag: str | None = None
    eval_suffix: tuple[str, ...] = ()


UPSTREAMS = {
    "geosa_basa": UpstreamSpec(
        name="geosa_basa",
        repo="https://github.com/mmmll23/GeoSA-BaSA.git",
        revision="67cec152349db6b29bf14e991e80f01fd365f73c",
        dest=VENDOR_DIR / "GeoSA-BaSA",
        sparse_paths=("configs", "geosa_basa", "tools"),
        train_args=(
            "tools/train.py",
            "configs/dinov2/geosa_basa_dinov2_mask2former_512x512_bs1x4_whumix.py",
        ),
        eval_args=(
            "tools/test.py",
            "configs/dinov2/geosa_basa_dinov2_mask2former_512x512_bs1x4_whumix.py",
        ),
        eval_suffix=("--backbone", "./pretrained_weights/dinov2_converted.pth"),
    ),
    "hg_rsovsseg": UpstreamSpec(
        name="hg_rsovsseg",
        repo="https://github.com/HuangWBill/HG-RSOVSSeg.git",
        revision="58e1df1f68f15920e02320dc54fe7441acf37770",
        dest=VENDOR_DIR / "HG-RSOVSSeg",
        sparse_paths=("configs", "mmseg", "tools"),
        train_args=(
            "tools/train/train.py",
            "--config",
            "configs/OpenEarthMap_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_openearthmap-512x512.py",
            "--work-dir",
            "result/HG-RSOVSSeg/OpenEarthMap/",
        ),
        eval_args=(
            "tools/test.py",
            "--config",
            "configs/OpenEarthMap_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_openearthmap-512x512_test.py",
        ),
        checkpoint_flag="--checkpoint",
        eval_suffix=("--work-dir", "result/HG-RSOVSSeg/OpenEarthMap/test/"),
    ),
    "repstdc": UpstreamSpec(
        name="repstdc",
        repo="https://github.com/jayeliu/RepSTDC.git",
        revision="549c2e0cf81963aeae8e5c6b9c83a3aa496a8407",
        dest=VENDOR_DIR / "RepSTDC",
        sparse_paths=("config", "data", "mmseg_geo", "tools"),
        train_args=("tools/train.py", "config/repstdc/repstdc-ca_512x512_80k_oem.py"),
        eval_args=("tools/test.py", "config/repstdc/repstdc-ca_512x512_80k_oem.py"),
    ),
}

_ALIASES = {
    "geosa": "geosa_basa",
    "geosa_basa": "geosa_basa",
    "hg_rsovsseg": "hg_rsovsseg",
    "hgrsovsseg": "hg_rsovsseg",
    "repstdc": "repstdc",
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def get_upstream(name: str) -> UpstreamSpec:
    key = _ALIASES.get(_normalize(name))
    if key is None:
        raise ValueError(f"unknown paper model '{name}'; valid models: {', '.join(UPSTREAMS)}")
    return UPSTREAMS[key]


def build_train_command(name: str, passthrough: Sequence[str] = ()) -> list[str]:
    spec = get_upstream(name)
    return [sys.executable, *spec.train_args, *passthrough]


def build_eval_command(
    name: str,
    checkpoint: str | Path | None,
    passthrough: Sequence[str] = (),
) -> list[str]:
    if checkpoint is None:
        raise ValueError("evaluation requires a checkpoint path")
    spec = get_upstream(name)
    command = [sys.executable, *spec.eval_args]
    if spec.checkpoint_flag:
        command.extend((spec.checkpoint_flag, str(checkpoint)))
    else:
        command.append(str(checkpoint))
    command.extend(spec.eval_suffix)
    command.extend(passthrough)
    return command


def _git(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def ensure_checkout(name: str) -> Path:
    spec = get_upstream(name)
    dest = spec.dest
    if dest.exists() and not (dest / ".git").exists():
        raise RuntimeError(f"refusing to replace non-git path: {dest}")

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", "--no-checkout", spec.repo, str(dest))
        if spec.sparse_paths:
            _git("sparse-checkout", "init", "--cone", cwd=dest)
            _git("sparse-checkout", "set", *spec.sparse_paths, cwd=dest)
    else:
        origin = _git("remote", "get-url", "origin", cwd=dest, capture=True)
        if origin.rstrip("/") != spec.repo.rstrip("/"):
            raise RuntimeError(f"unexpected origin for {dest}: {origin}")

    head = _git("rev-parse", "HEAD", cwd=dest, capture=True)
    entrypoint = dest / spec.train_args[0]
    if head != spec.revision:
        _git("fetch", "--depth=1", "origin", spec.revision, cwd=dest)
    if head != spec.revision or not entrypoint.exists():
        _git("checkout", "--detach", spec.revision, cwd=dest)
    return dest
