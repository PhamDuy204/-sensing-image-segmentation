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
