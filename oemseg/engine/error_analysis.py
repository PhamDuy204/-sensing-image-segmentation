"""Compact per-sample error reports without storing prediction images."""

from __future__ import annotations

import csv
import json
from pathlib import Path

FIELDS = ("filename", "region", "loss", "oa", "miou", "worst_class", "worst_class_iou")


def write_error_analysis(
    run_dir: Path,
    epoch: int,
    split: str,
    samples: list[dict[str, object]],
    top_n: int = 30,
    best_snapshot: bool = False,
) -> None:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    with (run_dir / "sample_scores.jsonl").open("a", buffering=1) as output:
        for sample in samples:
            output.write(json.dumps({"epoch": epoch, "split": split, **sample}) + "\n")

    worst = sorted(samples, key=lambda sample: (float(sample["miou"]), -float(sample["loss"])))[:top_n]
    path = run_dir / f"bad_predictions_{split}.tsv"
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(worst)

    if best_snapshot:
        snapshot_name = "bad_predictions_val_best.tsv" if split == "val" else "bad_predictions_test_at_best_val.tsv"
        (run_dir / snapshot_name).write_text(path.read_text())
