"""Best-checkpoint qualitative samples for W&B and local run artifacts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from oemseg.constants import CLASS_COLORS, CLASS_NAMES
from oemseg.data.dataset import OEMDataset
from oemseg.utils.tta import model_logits

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MASK_PALETTE = np.asarray(CLASS_COLORS, dtype=np.uint8)


def read_bad_prediction_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="") as input_file:
        return [
            str(row["filename"])
            for row in csv.DictReader(input_file, delimiter="\t")
            if row.get("filename")
        ]


def select_visualization_names(
    names: list[str],
    bad_names: list[str],
    count: int = 5,
    bad_count: int = 3,
    seed: int = 42,
) -> list[str]:
    if count < 1 or bad_count < 0:
        raise ValueError("count must be >= 1 and bad_count must be >= 0")

    rng = random.Random(seed)
    allowed = set(names)
    eligible_bad = list(dict.fromkeys(name for name in bad_names if name in allowed))
    rng.shuffle(eligible_bad)
    selected = eligible_bad[: min(count, bad_count, len(eligible_bad))]

    remaining = [name for name in names if name not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, min(count, len(names)) - len(selected))])
    return selected


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _image_from_tensor(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    array = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def _mask_image(mask: torch.Tensor | np.ndarray) -> Image.Image:
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    array = np.asarray(array, dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= len(MASK_PALETTE)):
        raise ValueError("mask contains class IDs outside the visualization palette")
    return Image.fromarray(MASK_PALETTE[array])


def render_label_legend(path: Path) -> Path:
    """Write one RGB legend image for the OpenEarthMap class palette."""
    row_height, swatch, width = 30, 22, 260
    image = Image.new("RGB", (width, row_height * len(CLASS_NAMES)), "white")
    draw = ImageDraw.Draw(image)
    for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, strict=True)):
        y = index * row_height
        draw.rectangle((6, y + 4, 6 + swatch, y + 4 + swatch), fill=color, outline="black")
        draw.text((38, y + 8), f"{index}: {name}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _compose_split_grid(
    originals: list[Image.Image],
    targets: list[Image.Image],
    predictions: list[Image.Image],
    names: list[str],
    tile_size: int = 256,
) -> Image.Image:
    if not originals or not (len(originals) == len(targets) == len(predictions) == len(names)):
        raise ValueError("visualization rows must be non-empty and aligned")

    label_width = 110
    canvas = Image.new("RGB", (label_width + tile_size * len(names), tile_size * 3), "white")
    draw = ImageDraw.Draw(canvas)
    rows = (("Original", originals), ("Ground truth", targets), ("Prediction", predictions))
    for row_index, (label, images) in enumerate(rows):
        y = row_index * tile_size
        draw.text((8, y + 8), label, fill="black")
        for column, image in enumerate(images):
            resized = image.resize((tile_size, tile_size), Image.Resampling.NEAREST if row_index else Image.Resampling.BILINEAR)
            x = label_width + column * tile_size
            canvas.paste(resized, (x, y))
            if row_index == 0:
                draw.rectangle((x, y, x + tile_size, y + 18), fill=(0, 0, 0))
                draw.text((x + 3, y + 3), names[column][:36], fill="white")
    return canvas


def render_best_checkpoint_visualizations(
    model,
    args,
    run_dir: Path,
    accelerator,
) -> dict[str, dict[str, object]]:
    """Render three 3xN grids using the already loaded best-validation checkpoint."""
    split_dir = run_dir / "splits"
    bad_sources = {
        "train": None,
        "val": (
            run_dir / "below_mean_val.tsv"
            if (run_dir / "below_mean_val.tsv").exists()
            else run_dir / "bad_predictions_val_best.tsv"
        ),
        "test": (
            run_dir / "below_mean_test.tsv"
            if (run_dir / "below_mean_test.tsv").exists()
            else run_dir / "bad_predictions_test_at_best_val.tsv"
            if (run_dir / "bad_predictions_test_at_best_val.tsv").exists()
            else run_dir / "bad_predictions_test.tsv"
        ),
    }
    seed_offsets = {"train": 101, "val": 202, "test": 303}
    output_dir = run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}

    model.eval()
    with torch.inference_mode():
        for split in ("train", "val", "test"):
            names = _read_manifest(split_dir / f"{split}.txt")
            if not names:
                continue
            bad_names = read_bad_prediction_names(bad_sources[split]) if bad_sources[split] else []
            selected = select_visualization_names(
                names,
                bad_names,
                count=5,
                bad_count=3,
                seed=args.seed + seed_offsets[split],
            )
            dataset = OEMDataset(args.data_root, selected, size=args.size, augment=False, return_name=True)
            originals: list[Image.Image] = []
            targets: list[Image.Image] = []
            predictions: list[Image.Image] = []

            for index in range(len(dataset)):
                image, target, _ = dataset[index]
                batch = image.unsqueeze(0).to(accelerator.device, non_blocking=True)
                if args.channels_last:
                    batch = batch.contiguous(memory_format=torch.channels_last)
                with accelerator.autocast():
                    logits = model_logits(model, batch, args.tta_scales, not args.no_tta_flips)
                prediction = logits.argmax(1)[0]
                originals.append(_image_from_tensor(image))
                targets.append(_mask_image(target))
                predictions.append(_mask_image(prediction))

            path = output_dir / f"best_{split}_examples.png"
            _compose_split_grid(originals, targets, predictions, selected).save(path)
            bad_set = set(bad_names)
            results[split] = {
                "path": str(path),
                "names": selected,
                "bad_names": [name for name in selected if name in bad_set],
            }

    (output_dir / "selection.json").write_text(json.dumps(results, indent=2))
    return results
