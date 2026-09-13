import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import scripts.openmmlab_paper_baseline as native_baseline

from scripts.openmmlab_paper_baseline import (
    normalize_oem_image_layout,
    PAPER_BATCH_SIZE,
    PAPER_ITERS,
    PYRAMIDMAMBA_EVAL_SPLIT,
    paper_training_plan,
)


def test_two_gpu_plan_preserves_paper_batch_and_optimizer_updates():
    plan = paper_training_plan(world_size=2)

    assert plan["micro_batch_per_gpu"] == 8
    assert plan["accumulation"] == 2
    assert plan["micro_batch_per_gpu"] * 2 * plan["accumulation"] == PAPER_BATCH_SIZE
    assert plan["micro_iters"] // plan["accumulation"] == PAPER_ITERS
    assert plan["warmup_micro_iters"] // plan["accumulation"] == 1000


def test_pyramidmamba_reports_on_public_500_image_validation_split():
    assert PYRAMIDMAMBA_EVAL_SPLIT == "val"


def test_native_runner_forces_headless_matplotlib_backend():
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    result = subprocess.run(
        [sys.executable, "-c", "import os; import scripts.openmmlab_paper_baseline; print(os.environ['MPLBACKEND'])"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "Agg"


def test_oem_channel_first_tiffs_are_normalized_to_hwc():
    import numpy as np

    chw = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    hwc = normalize_oem_image_layout(chw)

    assert hwc.shape == (4, 5, 3)
    assert np.array_equal(hwc[..., 0], chw[0])
    assert np.array_equal(hwc[..., 1], chw[1])
    assert np.array_equal(hwc[..., 2], chw[2])

    already_hwc = np.zeros((4, 5, 3), dtype=np.uint8)
    assert normalize_oem_image_layout(already_hwc) is already_hwc


def test_native_chunk_plan_uses_effective_iterations_without_shortening_scheduler_horizon():
    plan = native_baseline.native_chunk_plan(world_size=2, chunk_end_iter=26_666)

    assert plan["stop_micro_iters"] == 53_332
    assert plan["full_micro_iters"] == 160_000
    assert plan["complete"] is False

    final = native_baseline.native_chunk_plan(world_size=2, chunk_end_iter=80_000)
    assert final["stop_micro_iters"] == 160_000
    assert final["full_micro_iters"] == 160_000
    assert final["complete"] is True


def test_native_chunk_state_records_effective_and_micro_iteration():
    state = native_baseline.native_chunk_state(
        world_size=2,
        chunk_end_iter=26_666,
        runner_iter=53_332,
    )

    assert state == {
        "iteration": 26_666,
        "total_iterations": 80_000,
        "micro_iteration": 53_332,
        "total_micro_iterations": 160_000,
        "complete": False,
        "reason": "chunk_boundary",
    }


def test_apply_native_chunk_control_sets_mmengine_resume_and_boundary(tmp_path):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        train_cfg=SimpleNamespace(max_iters=160_000),
        default_hooks=SimpleNamespace(checkpoint=SimpleNamespace(save_last=False)),
        resume=False,
        load_from=None,
    )
    resume = tmp_path / "iter_53332.pth"

    native_baseline.apply_native_chunk_control(
        cfg,
        world_size=2,
        chunk_end_iter=53_333,
        resume_from=resume,
        smoke=False,
    )

    assert cfg.train_cfg.max_iters == 106_666
    assert cfg.default_hooks.checkpoint.save_last is True
    assert cfg.resume is True
    assert cfg.load_from == str(resume)
