from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_paper_recipe_is_accuracy_first():
    script = (ROOT / "scripts/kaggle_paper_repro.sh").read_text()

    for expected in (
        '--batch-size "$BATCH_SIZE"',
        'BATCH_SIZE=1',
        'GRAD_ACCUMULATION=1',
        '--val-fraction 0',
        '--patience 5',
        '--mixed-precision no',
        '--loss auto',
        '--wandb-mode offline',
    ):
        assert expected in script

    assert '--val-fraction 0.2' not in script
    assert '--mixed-precision fp16' not in script
    assert '--wandb-mode online' not in script


def test_kernel_files_enable_private_t4_and_keep_secrets_local():
    from scripts.kaggle_pipeline import build_kernel_files

    notebook, metadata = build_kernel_files(
        owner="duy18102004",
        slug="oem-unet-paper-repro-smoke",
        model="unet",
        smoke=True,
        repo_ref="feat/kaggle-auto-sync",
    )

    assert metadata["id"] == "duy18102004/oem-unet-paper-repro-smoke"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["duy18102004/oem-dataset"]

    source = "\n".join(notebook["cells"][0]["source"])
    assert "MODEL_NAME=unet" in source
    assert "SMOKE=1" in source
    assert "feat/kaggle-auto-sync" in source
    assert "kaggle_paper_repro.sh" in source
    assert "WANDB_API_KEY" not in source
    assert "KAGGLE_API_TOKEN" not in source


def test_normalize_status_handles_kaggle_cli_variants():
    from scripts.kaggle_pipeline import normalize_status

    assert normalize_status('Kernel status: RUNNING') == "RUNNING"
    assert normalize_status('duy/foo has status "complete"') == "COMPLETE"
    assert normalize_status('status: QUEUED') == "QUEUED"
    assert normalize_status('Kernel status: ERROR') == "ERROR"
    assert normalize_status('cancelled') == "CANCELLED"


def test_find_offline_runs_finds_downloaded_wandb_runs(tmp_path):
    from scripts.kaggle_pipeline import find_offline_runs

    first = tmp_path / "oem_outputs" / "run-a" / "wandb" / "offline-run-1"
    second = tmp_path / "oem_outputs" / "run-b" / "wandb" / "offline-run-2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "run-a.wandb").write_text("x")
    (second / "run-b.wandb").write_text("x")

    assert find_offline_runs(tmp_path) == [first, second]
