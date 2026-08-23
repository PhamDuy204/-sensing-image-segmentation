from pathlib import Path

from oemseg.engine.error_analysis import write_best_checkpoint_analysis


def test_best_checkpoint_analysis_writes_full_scores_and_below_mean(tmp_path: Path):
    samples = [
        {"filename": "a.tif", "region": "a", "loss": 0.1, "oa": 0.9, "miou": 0.8, "worst_class": "road", "worst_class_iou": 0.4},
        {"filename": "b.tif", "region": "b", "loss": 0.8, "oa": 0.4, "miou": 0.2, "worst_class": "water", "worst_class_iou": 0.0},
        {"filename": "c.tif", "region": "c", "loss": 0.5, "oa": 0.6, "miou": 0.5, "worst_class": "tree", "worst_class_iou": 0.1},
    ]

    summary = write_best_checkpoint_analysis(tmp_path, "val", samples)

    assert summary == {"sample_mean_miou": 0.5, "sample_count": 3, "below_mean_count": 1}
    scores = (tmp_path / "best_checkpoint_val_scores.tsv").read_text().splitlines()
    below = (tmp_path / "below_mean_val.tsv").read_text().splitlines()
    assert [row.split("\t", 1)[0] for row in scores[1:]] == ["b.tif", "c.tif", "a.tif"]
    assert [row.split("\t", 1)[0] for row in below[1:]] == ["b.tif"]
