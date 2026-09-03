from pathlib import Path


def test_train_entrypoint_is_thin():
    text = Path("train.py").read_text()
    assert len(text.splitlines()) <= 20
    assert "class OEMDataset" not in text
    assert "def evaluate" not in text


def test_kaggle_notebook_reuses_shared_trainer():
    import json

    path = Path("notebooks/kaggle_multi_gpu.ipynb")
    assert path.exists()
    notebook = json.loads(path.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "scripts/launch.py" in code
    assert "distributed" in code
    assert "def train_one_epoch" not in code
    assert "def evaluate" not in code
    assert "torch.cuda" not in code


def test_unetformer_setup_pins_upstream_geoseg():
    path = Path("scripts/setup_unetformer.sh")
    assert path.exists()
    text = path.read_text()
    assert "WangLibo1995/GeoSeg" in text
    assert "9453fe48209c4626b29e35e61bab93b61212c4b1" in text
    assert ".vendor/GeoSeg" in text
