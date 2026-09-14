from pathlib import Path
import zipfile


def test_fleet_phase_plan_chunks_native_openmmlab_by_effective_iteration():
    from scripts.kaggle_fleet import phase_plan

    assert phase_plan("unet") == [15, 30, 45]
    assert phase_plan("mask2former") == [15, 30, 45]
    assert phase_plan("segnext") == [26666, 53333, 80000]
    assert phase_plan("repstdc") == [26666, 53333, 80000]


def test_fleet_parses_kaggle_gpu_hours():
    from scripts.kaggle_fleet import _hours

    assert _hours("30.00h") == 30.0
    assert _hours(" 12.5h ") == 12.5


def test_resume_dataset_is_mounted_without_private_kernel_source():
    from scripts.kaggle_pipeline import build_kernel_files

    notebook, metadata = build_kernel_files(
        owner="new-owner",
        slug="oem-unet-part2",
        model="unet",
        smoke=False,
        repo_ref="main",
        chunk_end_epoch=30,
        resume_dataset="new-owner/private-resume",
    )
    source = "\n".join(notebook["cells"][0]["source"])
    assert metadata["dataset_sources"][-1] == "new-owner/private-resume"
    assert metadata["kernel_sources"] == []
    assert "RESUME_FROM_INPUT=1" in source
    assert "RESUME_ARCHIVE=" in source


def test_resume_zip_preserves_checkpoint_and_accelerator_state(tmp_path: Path):
    from scripts.kaggle_fleet import _resume_zip

    run = tmp_path / "download" / "oem_outputs" / "run"
    state = run / "accelerator_state"
    state.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"checkpoint")
    (run / "metrics.jsonl").write_text("{}\n")
    (run / "best_train_loss.pt").write_bytes(b"best")
    (state / "random_states_0.pkl").write_bytes(b"rng")

    archive = _resume_zip(tmp_path / "download", tmp_path / "relay")
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {
            "last.pt",
            "metrics.jsonl",
            "best_train_loss.pt",
            "accelerator_state/random_states_0.pkl",
        }


def test_resume_zip_packages_native_mmengine_last_checkpoint(tmp_path: Path):
    from scripts.kaggle_fleet import _resume_zip

    run = tmp_path / "download" / "oem_outputs" / "segnext-paper-repro-t4x2"
    run.mkdir(parents=True)
    checkpoint = run / "iter_53332.pth"
    checkpoint.write_bytes(b"mmengine-checkpoint")
    (run / "last_checkpoint").write_text(
        "/kaggle/working/oem_outputs/segnext-paper-repro-t4x2/iter_53332.pth\n"
    )

    archive = _resume_zip(tmp_path / "download", tmp_path / "relay", native=True)
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("resume_checkpoint.pth") == b"mmengine-checkpoint"


def test_phase_boundary_routes_epoch_and_native_chunks_to_distinct_pipeline_args():
    from scripts.kaggle_fleet import phase_boundary

    assert phase_boundary("unet", 15) == {"chunk_end_epoch": 15, "chunk_end_iter": None}
    assert phase_boundary("segnext", 26_666) == {"chunk_end_epoch": None, "chunk_end_iter": 26_666}
    assert phase_boundary("repstdc", 80_000) == {"chunk_end_epoch": None, "chunk_end_iter": 80_000}


def test_wait_for_dataset_ready_polls_until_ready(monkeypatch):
    from scripts import kaggle_fleet

    calls = []

    class Result:
        def __init__(self, stdout: str, returncode: int = 0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    responses = iter([
        Result('{"status": "pending", "current_version_number": 1}\n'),
        Result('{"status": "ready", "current_version_number": 1}\n'),
    ])

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return next(responses)

    monkeypatch.setattr(kaggle_fleet.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_fleet.time, "sleep", lambda _: None)

    kaggle_fleet._wait_for_dataset_ready(
        dataset="new-owner/private-resume",
        token="secret-token",
        kaggle_bin=Path("/fake/kaggle"),
        timeout_seconds=10,
        poll_seconds=1,
    )

    assert len(calls) == 2
    assert calls[0][1:4] == ["datasets", "status", "new-owner/private-resume"]
    assert calls[0][-2:] == ["--format", "json"]


def test_resume_from_completed_native_part_skips_earlier_parts(monkeypatch, tmp_path: Path):
    import json
    from scripts import kaggle_fleet

    fleet_root = tmp_path / "fleet"
    model_root = fleet_root / "repstdc"
    part2_root = model_root / "part-02"
    output_dir = part2_root / "output"
    native_run = output_dir / "oem_outputs" / "repstdc-paper-repro-t4x2"
    native_run.mkdir(parents=True)
    (native_run / "chunk_state.json").write_text(
        json.dumps({"iteration": 53333, "complete": False})
    )
    (native_run / "last_checkpoint").write_text(
        "/kaggle/working/oem_outputs/repstdc-paper-repro-t4x2/iter_106666.pth\n"
    )
    (native_run / "iter_106666.pth").write_bytes(b"checkpoint")
    part2_root.mkdir(parents=True, exist_ok=True)
    (part2_root / "state.json").write_text(
        json.dumps(
            {
                "kernel": "oldowner/oem-repstdc-part2",
                "output_dir": str(output_dir),
                "status": "SYNC_WARNING",
            }
        )
    )

    class Lock:
        def close(self):
            pass

    monkeypatch.setattr(kaggle_fleet.kp, "_tool", lambda *_: Path("/fake/tool"))
    monkeypatch.setattr(
        kaggle_fleet,
        "choose_account",
        lambda **_: (3, Lock(), "newowner", "token", 20.0),
    )
    relays = []

    def fake_create_resume_dataset(**kwargs):
        relays.append(kwargs)
        return "newowner/relay-dataset"

    monkeypatch.setattr(kaggle_fleet, "create_resume_dataset", fake_create_resume_dataset)
    submissions = []

    def fake_run_kernel_once(**kwargs):
        submissions.append(kwargs)
        out = kwargs["run_root"] / "output"
        marker = out / "oem_outputs" / "repstdc-paper-repro-t4x2"
        marker.mkdir(parents=True)
        (marker / "chunk_state.json").write_text(
            json.dumps({"iteration": 80000, "complete": True})
        )
        return {
            "kernel": "newowner/oem-repstdc-part3",
            "output_dir": out,
            "state_path": kwargs["run_root"] / "state.json",
            "synced_wandb_runs": [],
            "wandb_sync_error": None,
        }

    monkeypatch.setattr(kaggle_fleet.kp, "_run_kernel_once", fake_run_kernel_once)

    rc = kaggle_fleet.run_model(
        "repstdc",
        fleet_root,
        "fixed-sha",
        1,
        resume_from_part=2,
    )

    assert rc == 0
    assert len(submissions) == 1
    assert submissions[0]["chunk_end_iter"] == 80000
    assert submissions[0]["resume_dataset"] == "newowner/relay-dataset"
    assert relays[0]["part"] == 3
    assert relays[0]["previous_output"] == output_dir
