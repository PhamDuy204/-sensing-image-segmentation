#!/usr/bin/env python3
"""Complete OpenEarthMap with xBD RGB images and verify the public splits."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

SPLITS = {"train": 3000, "val": 500}


def region_for(filename: str) -> str:
    return filename.rsplit("_", 1)[0]


def mappings(csv_path: Path) -> list[tuple[str, str]]:
    with csv_path.open(newline="") as f:
        rows = [(a.strip(), b.strip()) for a, b in csv.reader(f) if a.strip() and b.strip()]
    if len(rows) != len({a for a, _ in rows}) or len(rows) != len({b for _, b in rows}):
        raise ValueError(f"Duplicate xBD or OEM names in {csv_path}")
    return rows


def index_xbd_images(xbd_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in xbd_root.rglob("*_pre_disaster.png"):
        if path.name in found:
            raise ValueError(f"Duplicate xBD image basename: {path.name}")
        found[path.name] = path
    return found


def copy_base(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{source}/.", str(destination)],
        check=True,
    )


def write_geotiff(png_path: Path, label_path: Path | None, output_path: Path) -> None:
    image = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if label_path is None:
        Image.fromarray(image).save(output_path, compression="tiff_deflate")
        return
    with rasterio.open(label_path) as src:
        profile = src.profile.copy()
    if image.shape[:2] != (profile["height"], profile["width"]):
        raise ValueError(
            f"Shape mismatch: {png_path} {image.shape[:2]} vs {label_path} "
            f"{(profile['height'], profile['width'])}"
        )
    profile.update(count=3, dtype="uint8", nodata=None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.moveaxis(image, -1, 0))


def split_entries(root: Path, split: str) -> list[str]:
    return [line.strip() for line in (root / f"{split}.txt").read_text().splitlines() if line.strip()]


def verify(root: Path) -> None:
    label_values: set[int] = set()
    for split, expected in SPLITS.items():
        names = split_entries(root, split)
        if len(names) != expected:
            raise ValueError(f"{split}.txt has {len(names)} entries, expected {expected}")
        missing_images, missing_labels = [], []
        for name in names:
            region = region_for(name)
            image = root / region / "images" / name
            label = root / region / "labels" / name
            if not image.exists():
                missing_images.append(name)
            if not label.exists():
                missing_labels.append(name)
            if label.exists():
                label_values.update(np.unique(np.asarray(Image.open(label))).tolist())
        if missing_images or missing_labels:
            raise FileNotFoundError(
                f"{split}: missing {len(missing_images)} images and {len(missing_labels)} labels; "
                f"examples={missing_images[:3] + missing_labels[:3]}"
            )
        print(f"{split}: {len(names)} paired images/labels")
    if label_values != set(range(9)):
        raise ValueError(f"Expected labels 0..8, found {sorted(label_values)}")
    test_names = split_entries(root, "test")
    missing_test_images = [
        name for name in test_names if not (root / region_for(name) / "images" / name).exists()
    ]
    if missing_test_images:
        raise FileNotFoundError(
            f"test: missing {len(missing_test_images)} images; examples={missing_test_images[:5]}"
        )
    print(f"test: {len(test_names)} images (labels are not public)")
    print("labels: 0..8")
    print("dataset verification: OK")


def prepare(source: Path, xbd_root: Path, destination: Path) -> None:
    rows = mappings(source / "xbd_files.csv")
    xbd = index_xbd_images(xbd_root)
    required = {src for src, _ in rows}
    missing = sorted(required - xbd.keys())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(required)} required xBD images. Examples: {missing[:10]}"
        )
    copy_base(source, destination)
    written = 0
    for source_name, oem_name in rows:
        region = region_for(oem_name)
        label = destination / region / "labels" / oem_name
        output = destination / region / "images" / oem_name
        if not output.exists():
            write_geotiff(xbd[source_name], label if label.exists() else None, output)
            written += 1
    print(f"inserted {written} missing xBD RGB images")
    verify(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/OpenEarthMap/OpenEarthMap_wo_xBD"),
    )
    parser.add_argument(
        "--xbd-root",
        type=Path,
        default=Path("datasets/OpenEarthMap/xBD_huggingface/extracted"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("datasets/OpenEarthMap/OpenEarthMap"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify(args.destination) if args.verify_only else prepare(args.source, args.xbd_root, args.destination)


if __name__ == "__main__":
    main()
