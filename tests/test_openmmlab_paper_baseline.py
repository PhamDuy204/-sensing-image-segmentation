import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
