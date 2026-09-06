#!/usr/bin/env python3
"""Run SegNeXt/RepSTDC with the paper's native MMSegmentation recipe."""

from __future__ import annotations

import argparse
import json
import os

# Kaggle notebooks export an inline backend that is invalid inside the isolated
# OpenMMLab environment and inherited by every torchrun rank. This runner is
# headless, so force Matplotlib's non-interactive backend before MMEngine imports.
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path

PAPER_BATCH_SIZE = 32
PAPER_ITERS = 80_000
PAPER_WARMUP_ITERS = 1_000
MAX_MICRO_BATCH_PER_GPU = 8
PYRAMIDMAMBA_EVAL_SPLIT = "val"
MSCAN_T_CHECKPOINT = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/"
    "mscan_t_20230227-119e8c9f.pth"
)


def region_for(filename: str) -> str:
    return filename.rsplit("_", 1)[0]


def normalize_oem_image_layout(image):
    """Normalize mixed OpenEarthMap TIFF layouts to HWC for MMCV/OpenCV."""
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        return image.transpose(1, 2, 0)
    return image


def _stage_native_file(source: Path, target: Path, kind: str) -> None:
    if kind == "images":
        import tifffile

        with tifffile.TiffFile(source) as tif:
            shape = tif.series[0].shape
        if len(shape) == 3 and shape[0] == 3 and shape[-1] != 3:
            image = normalize_oem_image_layout(tifffile.imread(source))
            if target.is_symlink():
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            if not target.exists():
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.tif")
                try:
                    tifffile.imwrite(temporary, image, photometric="rgb")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            return

    if not target.exists():
        try:
            target.symlink_to(source.resolve())
        except FileExistsError:
            pass  # another torchrun rank staged the same file first


def prepare_native_oem(data_root: Path, stage_root: Path) -> Path:
    """Expose the project's fixed splits in the flat layout expected by OEMDataset."""
    for split in ("train", "val"):
        names = [line.strip() for line in (data_root / f"{split}.txt").read_text().splitlines() if line.strip()]
        expected = 3000 if split == "train" else 500
        if len(names) != expected:
            raise RuntimeError(f"{split}: expected {expected} samples, got {len(names)}")
        for kind in ("images", "labels"):
            dest = stage_root / kind / split
            dest.mkdir(parents=True, exist_ok=True)
            for name in names:
                source = data_root / region_for(name) / kind / name
                target = dest / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                _stage_native_file(source, target, kind)
    return stage_root


def paper_training_plan(world_size: int) -> dict[str, int]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    micro_batch = min(MAX_MICRO_BATCH_PER_GPU, PAPER_BATCH_SIZE // world_size)
    if micro_batch < 1 or PAPER_BATCH_SIZE % (micro_batch * world_size):
        raise ValueError(f"paper batch {PAPER_BATCH_SIZE} is incompatible with world_size={world_size}")
    accumulation = PAPER_BATCH_SIZE // (micro_batch * world_size)
    return {
        "micro_batch_per_gpu": micro_batch,
        "accumulation": accumulation,
        "micro_iters": PAPER_ITERS * accumulation,
        "warmup_micro_iters": PAPER_WARMUP_ITERS * accumulation,
    }


def _segnext_model(data_preprocessor: dict) -> dict:
    sync_bn = dict(type="SyncBN", requires_grad=True)
    return dict(
        type="EncoderDecoder",
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type="MSCAN",
            init_cfg=dict(type="Pretrained", checkpoint=MSCAN_T_CHECKPOINT),
            embed_dims=[32, 64, 160, 256],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type="GELU"),
            norm_cfg=sync_bn,
        ),
        decode_head=dict(
            type="LightHamHead",
            in_channels=[64, 160, 256],
            in_index=[1, 2, 3],
            channels=256,
            ham_channels=256,
            ham_kwargs=dict(MD_R=16),
            dropout_ratio=0.1,
            num_classes=8,
            norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ),
        train_cfg=dict(),
        test_cfg=dict(mode="whole"),
    )


def build_config(model: str, data_root: Path, work_dir: Path, run_name: str, *, smoke: bool = False):
    from mmengine.config import Config

    project_root = Path(__file__).resolve().parents[1]
    upstream = project_root / ".vendor" / "RepSTDC"
    source_root = upstream / "mmseg_geo"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import mmseg_geo.datasets  # noqa: F401  # registers the upstream OEMDataset
    import mmseg_geo.models  # noqa: F401  # registers the upstream RepSTDC backbone

    cfg = Config.fromfile(str(upstream / "config/repstdc/repstdc-ca_512x512_80k_oem.py"))
    if model == "segnext":
        cfg.model = _segnext_model(dict(cfg.model.data_preprocessor))
    elif model != "repstdc":
        raise ValueError(f"unsupported native baseline: {model}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cfg.launcher = "pytorch" if world_size > 1 else "none"
    cfg.work_dir = str(work_dir)
    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = cfg[loader_name]
        loader.dataset.data_root = str(data_root)
    cfg.test_dataloader.dataset.data_prefix = dict(
        img_path=f"images/{PYRAMIDMAMBA_EVAL_SPLIT}",
        seg_map_path=f"labels/{PYRAMIDMAMBA_EVAL_SPLIT}",
    )
    plan = paper_training_plan(world_size)
    cfg.train_dataloader.batch_size = plan["micro_batch_per_gpu"]
    cfg.optim_wrapper.accumulative_counts = plan["accumulation"]
    cfg.train_cfg.max_iters = plan["micro_iters"]
    cfg.train_cfg.val_interval *= plan["accumulation"]
    cfg.default_hooks.checkpoint.interval *= plan["accumulation"]
    for scheduler in cfg.param_scheduler:
        scheduler.begin *= plan["accumulation"]
        scheduler.end *= plan["accumulation"]

    if smoke:
        cfg.optim_wrapper.accumulative_counts = 1
        cfg.train_cfg.max_iters = 2
        cfg.train_cfg.val_interval = 1
        cfg.train_dataloader.batch_size = min(cfg.train_dataloader.batch_size, 2)
        cfg.train_dataloader.dataset.indices = 4
        cfg.val_dataloader.dataset.indices = 4
        cfg.test_dataloader.dataset.indices = 4
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        cfg.val_dataloader.num_workers = 0
        cfg.val_dataloader.persistent_workers = False
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False

    cfg.visualizer.vis_backends = [
        dict(type="LocalVisBackend"),
        dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project=os.environ.get("WANDB_PROJECT", "sensing image segmentation"),
                entity=os.environ.get("WANDB_ENTITY", "phamdinhanhduy-university-of-information-and-technology"),
                name=run_name,
            ),
        ),
    ]
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("segnext", "repstdc"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(os.environ.get("OEM_NATIVE_DATA_ROOT", "/kaggle/tmp/oem-native"))
    if not str(stage_root).startswith("/kaggle/"):
        stage_root = run_dir / "native_data"
    prepare_native_oem(args.data_root, stage_root)

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(run_dir))

    cfg = build_config(args.model, stage_root, run_dir, args.run_name, smoke=args.smoke)
    from mmengine.runner import Runner
    from mmengine.dist import is_main_process

    runner = Runner.from_cfg(cfg)
    runner.train()
    metrics = runner.test()
    if is_main_process():
        summary = {
            "model": args.model,
            "training_protocol": "liu2024-repstdc-openearthmap",
            "evaluation_protocol": "pyramidmamba-openearthmap-500-val",
            "evaluation_split": PYRAMIDMAMBA_EVAL_SPLIT,
            "iterations": 2 if args.smoke else PAPER_ITERS,
            "global_batch_size": PAPER_BATCH_SIZE,
            "metrics": metrics,
        }
        (run_dir / "best_checkpoint_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"phase": "official_val", **metrics}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
