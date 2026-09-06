from scripts.openmmlab_paper_baseline import (
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
