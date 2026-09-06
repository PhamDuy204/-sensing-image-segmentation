from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from oemseg.data.dataset import read_split, region_for
from oemseg.data.loaders import build_loaders, split_train_val, write_split_manifests


def test_region_name_uses_last_underscore():
    assert region_for("little_rock_12.tif") == "little_rock"


def test_real_oem_split_counts():
    root = Path("datasets/OpenEarthMap/OpenEarthMap")
    assert len(read_split(root, "train")) == 3000
    assert len(read_split(root, "val")) == 500


def test_region_stratified_split_is_deterministic_and_exact():
    names = [f"alpha_{i}.tif" for i in range(20)] + [f"beta_{i}.tif" for i in range(10)]
    train_a, val_a = split_train_val(names, 0.2, seed=42)
    train_b, val_b = split_train_val(names, 0.2, seed=42)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 24
    assert len(val_a) == 6
    assert not set(train_a) & set(val_a)
    assert set(map(region_for, val_a)) == {"alpha", "beta"}
    assert Counter(map(region_for, val_a)) == {"alpha": 4, "beta": 2}


def test_real_research_split_is_2700_300_500_and_disjoint():
    root = Path("datasets/OpenEarthMap/OpenEarthMap")
    official_train = read_split(root, "train")
    reported_test = read_split(root, "val")
    train, val = split_train_val(official_train, 0.1, seed=42)

    assert (len(train), len(val), len(reported_test)) == (2700, 300, 500)
    assert not set(train) & set(val)
    assert not set(train) & set(reported_test)
    assert not set(val) & set(reported_test)
    assert set(map(region_for, val)) == set(map(region_for, official_train))


def test_split_manifests_match_loader_datasets(tmp_path: Path):
    args = SimpleNamespace(
        data_root=Path("datasets/OpenEarthMap/OpenEarthMap"),
        internal_val_fraction=0.1,
        seed=42,
        size=32,
        workers=0,
        batch_size=2,
        eval_batch_size=2,
    )
    loaders = build_loaders(args)
    write_split_manifests(tmp_path, loaders)

    assert (tmp_path / "splits/train.txt").read_text().splitlines() == loaders.train.dataset.names
    assert (tmp_path / "splits/val.txt").read_text().splitlines() == loaders.internal_val.dataset.names
    assert (tmp_path / "splits/test.txt").read_text().splitlines() == loaders.test.dataset.names


def test_eval_loaders_expose_sample_names():
    args = SimpleNamespace(
        data_root=Path("datasets/OpenEarthMap/OpenEarthMap"),
        internal_val_fraction=0.1,
        seed=42,
        size=32,
        workers=0,
        batch_size=2,
        eval_batch_size=2,
    )
    loaders = build_loaders(args)
    assert loaders.internal_val.dataset.return_name is True
    assert loaders.test.dataset.return_name is True
    assert loaders.train.dataset.return_name is False


def test_resumable_train_loader_has_checkpointable_generator_and_nonpersistent_workers():
    args = SimpleNamespace(
        data_root=Path("datasets/OpenEarthMap/OpenEarthMap"),
        internal_val_fraction=0,
        seed=42,
        size=32,
        workers=2,
        batch_size=2,
        eval_batch_size=2,
        stop_after_epoch=15,
        resume_from=None,
    )
    loaders = build_loaders(args)

    assert loaders.train_generator is loaders.train.generator
    assert loaders.train_generator is not None
    assert loaders.train.worker_init_fn is not None
    assert loaders.train.persistent_workers is False
