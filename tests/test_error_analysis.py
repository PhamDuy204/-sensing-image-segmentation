import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from oemseg.engine.error_analysis import write_error_analysis
from oemseg.engine.evaluator import evaluate


class NamedLogitDataset(Dataset):
    def __init__(self):
        good = torch.tensor(
            [
                [[9.0, -9.0], [-9.0, 9.0]],
                [[-9.0, 9.0], [9.0, -9.0]],
            ]
        )
        bad = -good
        self.images = torch.stack([good, bad])
        self.targets = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        self.names = ["alpha_1.tif", "beta_2.tif"]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.images[index], self.targets[index], self.names[index]


def test_evaluate_returns_per_sample_scores_with_names():
    loader = DataLoader(NamedLogitDataset(), batch_size=2, shuffle=False)
    accelerator = Accelerator(cpu=True)
    model, loader = accelerator.prepare(torch.nn.Identity(), loader)
    result = evaluate(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        accelerator,
        [1.0],
        False,
    )

    assert [sample["filename"] for sample in result.samples] == ["alpha_1.tif", "beta_2.tif"]
    assert result.samples[0]["miou"] > result.samples[1]["miou"]
    assert result.samples[0]["region"] == "alpha"
    assert "worst_class" in result.samples[1]


def test_error_analysis_writes_sorted_top_n_without_images(tmp_path: Path):
    samples = [
        {"filename": "a.tif", "region": "a", "loss": 0.1, "oa": 0.9, "miou": 0.8, "worst_class": "road", "worst_class_iou": 0.4},
        {"filename": "b.tif", "region": "b", "loss": 0.8, "oa": 0.4, "miou": 0.2, "worst_class": "water", "worst_class_iou": 0.0},
        {"filename": "c.tif", "region": "c", "loss": 0.5, "oa": 0.6, "miou": 0.5, "worst_class": "tree", "worst_class_iou": 0.1},
    ]
    write_error_analysis(tmp_path, epoch=3, split="val", samples=samples, top_n=2, best_snapshot=True)

    rows = (tmp_path / "bad_predictions_val.tsv").read_text().splitlines()
    assert rows[1].startswith("b.tif\t")
    assert rows[2].startswith("c.tif\t")
    assert len(rows) == 3
    assert (tmp_path / "bad_predictions_val_best.tsv").read_text() == (tmp_path / "bad_predictions_val.tsv").read_text()

    records = [json.loads(line) for line in (tmp_path / "sample_scores.jsonl").read_text().splitlines()]
    assert {record["epoch"] for record in records} == {3}
    assert {record["split"] for record in records} == {"val"}
    assert all("image" not in record and "prediction" not in record for record in records)
