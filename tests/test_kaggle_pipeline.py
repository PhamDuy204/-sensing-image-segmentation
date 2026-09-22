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
    assert "MODEL_VARIANT=''" in source
    assert "SMOKE=1" in source
    assert "feat/kaggle-auto-sync" in source
    assert "kaggle_paper_repro.sh" in source
    assert "WANDB_API_KEY" not in source
    assert "KAGGLE_API_TOKEN" not in source



def test_kernel_files_forward_optional_model_variant_to_notebook():
    from scripts.kaggle_pipeline import build_kernel_files

    notebook, _ = build_kernel_files(
        owner="duy18102004",
        slug="oem-unet-resnet34-paper-repro",
        model="unet",
        model_variant="resnet34",
        smoke=False,
        repo_ref="main",
    )

    source = "\n".join(notebook["cells"][0]["source"])
    assert "MODEL_NAME=unet" in source
    assert "MODEL_VARIANT=resnet34" in source

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
    assert 'MODEL_VARIANT_ARGS=(--model-variant "$MODEL_VARIANT")' in script
    assert 'RUN_MODEL_NAME="${MODEL_NAME}-$(printf' in script
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


def test_cancel_acknowledged_is_a_terminal_status():
    from scripts.kaggle_pipeline import TERMINAL_STATUSES, normalize_status

    assert normalize_status(
        'foo/bar has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'
    ) == "CANCEL_ACKNOWLEDGED"
    assert "CANCEL_ACKNOWLEDGED" in TERMINAL_STATUSES


def test_chunk_end_epochs_cover_full_training_without_crossing_chunk_size():
    from scripts.kaggle_pipeline import chunk_end_epochs

    assert chunk_end_epochs(45, 15) == [15, 30, 45]
    assert chunk_end_epochs(45, 20) == [20, 40, 45]


def test_chunked_kernel_attaches_previous_output_and_requests_resume():
    from scripts.kaggle_pipeline import build_kernel_files

    notebook, metadata = build_kernel_files(
        owner="ovnduytrng",
        slug="oem-unetformer-paper-repro-part2",
        model="unetformer",
        smoke=False,
        repo_ref="main",
        chunk_end_epoch=30,
        previous_kernel="ovnduytrng/oem-unetformer-paper-repro-part1",
    )

    assert metadata["kernel_sources"] == [
        "ovnduytrng/oem-unetformer-paper-repro-part1"
    ]
    source = "\n".join(notebook["cells"][0]["source"])
    assert "CHUNK_END_EPOCH=30" in source
    assert "RESUME_FROM_INPUT=1" in source


def test_chunked_repro_script_resumes_checkpoint_and_defers_eval_to_final_epoch():
    script = (ROOT / "scripts/kaggle_paper_repro.sh").read_text()

    for expected in (
        'CHUNK_END_EPOCH="${CHUNK_END_EPOCH:-0}"',
        'RESUME_FROM_INPUT="${RESUME_FROM_INPUT:-0}"',
        '--stop-after-epoch "$CHUNK_END_EPOCH"',
        '--resume-from "$RESUME_CHECKPOINT"',
        'EVAL_START_EPOCH=44',
        '--eval-start-epoch "$EVAL_START_EPOCH"',
        '*/oem_outputs/${RUN_NAME}/last.pt',
    ):
        assert expected in script


def test_wandb_sync_command_can_append_chunks_to_one_run(tmp_path):
    from scripts.kaggle_pipeline import wandb_sync_command

    command = wandb_sync_command(
        tmp_path / "wandb",
        tmp_path / "offline-run-abc",
        target_id="deadbeef",
        append=True,
    )
    assert command == [
        str(tmp_path / "wandb"),
        "sync",
        "--legacy",
        "--id",
        "deadbeef",
        "--append",
        str(tmp_path / "offline-run-abc"),
    ]


def test_foreground_pins_repo_ref_to_commit_before_submission(monkeypatch, tmp_path):
    import argparse
    from scripts import kaggle_pipeline as pipeline

    pinned = "a" * 40
    captured = {}
    monkeypatch.setattr(pipeline, "resolve_repo_ref", lambda root, ref: pinned, raising=False)
    monkeypatch.setattr(pipeline, "_load_account", lambda path: ("owner", "token"))
    monkeypatch.setattr(pipeline, "_tool", lambda client_dir, name: tmp_path / name)

    def fake_run_kernel_once(**kwargs):
        captured["repo_ref"] = kwargs["args"].repo_ref
        return {}

    monkeypatch.setattr(pipeline, "_run_kernel_once", fake_run_kernel_once)
    args = argparse.Namespace(
        token_file=tmp_path / "token",
        client_dir=tmp_path,
        smoke=False,
        slug="oem-unet-paper-repro",
        model="unet",
        state_root=tmp_path / "state",
        chunk_epochs=0,
        repo_ref="main",
    )

    assert pipeline._foreground(args) == 0
    assert captured["repo_ref"] == pinned


def test_recover_existing_run_polls_and_downloads_without_resubmitting(monkeypatch, tmp_path):
    import argparse
    import json
    import subprocess
    from scripts import kaggle_pipeline as pipeline

    run_root = tmp_path / "run"
    output_dir = run_root / "output"
    output_dir.mkdir(parents=True)
    state_path = run_root / "state.json"
    state_path.write_text(json.dumps({
        "kernel": "owner/oem-unet-resnet34-paper-repro",
        "output_dir": str(output_dir),
        "status": "RUNNING",
    }))

    monkeypatch.setattr(pipeline, "_load_account", lambda path: ("owner", "token"))
    monkeypatch.setattr(pipeline, "_tool", lambda client_dir, name: tmp_path / name)
    commands = []

    def fake_run(command, *, env=None, capture=False):
        commands.append(command)
        if command[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="Kernel status: COMPLETE", stderr="")
        if command[1:3] == ["kernels", "output"]:
            (output_dir / "oem_outputs").mkdir()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(pipeline, "_sync_wandb", lambda *a, **k: ["offline-run-x"])
    args = argparse.Namespace(
        recover_run_root=run_root,
        token_file=tmp_path / "token",
        client_dir=tmp_path,
        poll_seconds=1,
    )

    assert pipeline._recover_existing(args) == 0
    assert not any(cmd[1:3] == ["kernels", "push"] for cmd in commands)
    assert any(cmd[1:3] == ["kernels", "status"] for cmd in commands)
    assert any(cmd[1:3] == ["kernels", "output"] for cmd in commands)
    recovered = json.loads(state_path.read_text())
    assert recovered["status"] == "SYNCED"
    assert recovered["synced_wandb_runs"] == ["offline-run-x"]


def test_u2net_kernel_enables_expandable_cuda_segments_only_for_u2net():
    from scripts.kaggle_pipeline import build_kernel_files

    u2net_notebook, _ = build_kernel_files(
        owner="duy18102004", slug="oem-u2net", model="u2net", smoke=False, repo_ref="main"
    )
    unet_notebook, _ = build_kernel_files(
        owner="duy18102004", slug="oem-unet", model="unet", smoke=False, repo_ref="main"
    )

    u2net_source = "\n".join(u2net_notebook["cells"][0]["source"])
    unet_source = "\n".join(unet_notebook["cells"][0]["source"])

    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in u2net_source
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" not in unet_source


def test_fleet_output_download_retries_without_force(monkeypatch, tmp_path):
    import subprocess
    from scripts import kaggle_fleet as fleet

    commands = []
    sleeps = []

    def fake_run(command, *, env=None, check=False, text=False, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    monkeypatch.setattr(fleet.time, "sleep", lambda seconds: sleeps.append(seconds))

    fleet._download_kernel_outputs_with_retry(
        kaggle_bin=tmp_path / "kaggle",
        kernel="owner/kernel",
        output_dir=tmp_path / "output",
        token="secret",
        attempts=3,
        base_delay=2,
    )

    assert len(commands) == 2
    assert all("-o" not in command for command in commands)
    assert all("--file-pattern" in command for command in commands)
    assert sleeps == [2]


def test_fleet_recovers_completed_part_and_reuses_persistent_wandb_id(monkeypatch, tmp_path):
    import json
    from scripts import kaggle_fleet as fleet

    model_root = tmp_path / "u2net"
    part_root = model_root / "part-01"
    output_dir = part_root / "output"
    output_dir.mkdir(parents=True)
    state_path = part_root / "state.json"
    state_path.write_text(json.dumps({
        "kernel": "owner/oem-u2net-fleet-test-part1",
        "output_dir": str(output_dir),
        "status": "COMPLETE",
    }))
    marker = output_dir / "oem_outputs" / "u2net-paper-repro-t4x2" / "chunk_state.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"epoch": 15, "complete": False}))

    calls = []
    monkeypatch.setattr(
        fleet,
        "_token_for_owner",
        lambda owner: "token" if owner == "owner" else None,
    )
    monkeypatch.setattr(
        fleet,
        "_download_kernel_outputs_with_retry",
        lambda **kwargs: calls.append(("download", kwargs["kernel"])),
    )
    monkeypatch.setattr(
        fleet.kp,
        "_sync_wandb",
        lambda *args, **kwargs: calls.append(("sync", kwargs["target_id"])) or ["offline-run-x"],
    )

    wandb_id = fleet._load_or_create_wandb_target_id(model_root)
    assert fleet._load_or_create_wandb_target_id(model_root) == wandb_id

    result = fleet._recover_existing_part(
        part_root=part_root,
        kaggle_bin=tmp_path / "kaggle",
        wandb_bin=tmp_path / "wandb",
        wandb_target_id=wandb_id,
        wandb_append=False,
    )

    assert result is not None
    assert result["kernel"] == "owner/oem-u2net-fleet-test-part1"
    assert result["output_dir"] == output_dir
    recovered = json.loads(state_path.read_text())
    assert recovered["status"] == "SYNCED"
    assert recovered["synced_wandb_runs"] == ["offline-run-x"]
    assert calls == [
        ("download", "owner/oem-u2net-fleet-test-part1"),
        ("sync", wandb_id),
    ]


def test_fleet_recovers_in_process_when_completed_kernel_download_fails(monkeypatch, tmp_path):
    import subprocess
    from scripts import kaggle_fleet as fleet

    class Lock:
        def close(self):
            pass

    recovered_result = {
        "kernel": "owner/kernel-part1",
        "output_dir": tmp_path / "output",
        "state_path": tmp_path / "state.json",
        "synced_wandb_runs": ["offline-run-x"],
    }
    recover_calls = []

    monkeypatch.setattr(fleet.kp, "_tool", lambda client_dir, name: tmp_path / name)
    monkeypatch.setattr(fleet, "_load_or_create_wandb_target_id", lambda root: "deadbeef")
    monkeypatch.setattr(fleet, "phase_plan", lambda model: [None])
    monkeypatch.setattr(
        fleet,
        "choose_account",
        lambda **kwargs: (3, Lock(), "owner", "token", 20.0),
    )

    def fake_recover(**kwargs):
        recover_calls.append(kwargs["part_root"])
        return None if len(recover_calls) == 1 else recovered_result

    monkeypatch.setattr(fleet, "_recover_existing_part", fake_recover)
    monkeypatch.setattr(
        fleet.kp,
        "_run_kernel_once",
        lambda **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["kaggle", "kernels", "output"])
        ),
    )

    assert fleet.run_model("segnext", tmp_path / "fleet", "a" * 40, 1) == 0
    assert len(recover_calls) == 2
