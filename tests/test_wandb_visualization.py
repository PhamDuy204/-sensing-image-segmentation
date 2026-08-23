from oemseg.utils.visualization import read_bad_prediction_names, select_visualization_names


def test_select_visualization_names_prefers_bad_examples_and_is_deterministic():
    names = [f"sample_{index}.png" for index in range(10)]
    bad_names = ["sample_1.png", "sample_3.png", "sample_5.png", "not_in_split.png"]

    first = select_visualization_names(names, bad_names, count=5, bad_count=3, seed=123)
    second = select_visualization_names(names, bad_names, count=5, bad_count=3, seed=123)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert len(set(first) & {"sample_1.png", "sample_3.png", "sample_5.png"}) == 3
    assert set(first) <= set(names)


def test_read_bad_prediction_names_handles_missing_file_and_reads_tsv(tmp_path):
    missing = tmp_path / "missing.tsv"
    assert read_bad_prediction_names(missing) == []

    path = tmp_path / "bad.tsv"
    path.write_text(
        "filename\tregion\tloss\toa\tmiou\tworst_class\tworst_class_iou\n"
        "a.png\tx\t1\t0\t0\troad\t0\n"
        "b.png\tx\t1\t0\t0\ttree\t0\n"
    )
    assert read_bad_prediction_names(path) == ["a.png", "b.png"]
