from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_paper_recipe_is_accuracy_first():
    script = (ROOT / "scripts/kaggle_paper_repro.sh").read_text()

    for expected in (
        '--batch-size "$BATCH_SIZE"',
        'BATCH_SIZE=1',
        'GRAD_ACCUMULATION=1',
        '--val-fraction 0',
        'PATIENCE=5',
        '--patience "$PATIENCE"',
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
    assert metadata["machine_shape"] == "NvidiaTeslaP100"
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


def test_generated_notebook_keeps_repository_out_of_kaggle_outputs():
    from scripts.kaggle_pipeline import build_kernel_files

    notebook, _ = build_kernel_files(
        owner="duy18102004",
        slug="oem-unet-paper-repro-smoke",
        model="unet",
        smoke=True,
        repo_ref="main",
    )
    source = "\n".join(notebook["cells"][0]["source"])
    assert "REPO_DIR=/kaggle/tmp/OEM_Segmentation" in source
    assert "REPO_DIR=/kaggle/working/OEM_Segmentation" not in source


def test_unet_repro_uses_single_p100_but_multi_gpu_models_keep_t4x2():
    from scripts.kaggle_pipeline import build_kernel_files

    _, unet = build_kernel_files(
        owner="duy18102004", slug="oem-unet-paper-repro", model="unet", smoke=False, repo_ref="main"
    )
    _, mask2former = build_kernel_files(
        owner="duy18102004", slug="oem-mask2former-paper-repro", model="mask2former", smoke=False, repo_ref="main"
    )

    assert unet["machine_shape"] == "NvidiaTeslaP100"
    assert mask2former["machine_shape"] == "NvidiaTeslaT4"


def test_kaggle_repro_script_distinguishes_smoke_and_keeps_temp_env_out_of_outputs():
    script = (ROOT / "scripts/kaggle_paper_repro.sh").read_text()
    pipeline = (ROOT / "scripts/kaggle_pipeline.py").read_text()

    assert 'RUN_NAME="${RUN_NAME}${SMOKE_SUFFIX}"' in script
    assert 'SMOKE_SUFFIX="-smoke"' in script
    assert "/kaggle/tmp/.micromamba" in script
    assert "/kaggle/working/.micromamba" not in script
    assert '"--file-pattern"' in pipeline
    assert 'r"^oem_outputs/"' in pipeline


def test_unet_p100_bootstrap_and_mask2former_optimizer_recipe_are_explicit():
    script = (ROOT / "scripts/kaggle_paper_repro.sh").read_text()

    for expected in (
        "torch==2.6.0",
        "torchvision==0.21.0",
        "https://download.pytorch.org/whl/cu118",
        '[[ "$MODEL_NAME" == "unet" ]]',
        "BATCH_SIZE=2",
        'GPU_IDS="0"',
        "LR=1e-4",
        "ENCODER_LR=1e-5",
        "WEIGHT_DECAY=0.05",
        "MAX_GRAD_NORM=0.01",
        "WARMUP_EPOCHS=0",
        "PATIENCE=0",
    ):
        assert expected in script
