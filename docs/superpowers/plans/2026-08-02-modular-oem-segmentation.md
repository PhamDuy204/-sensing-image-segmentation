# Modular OpenEarthMap Segmentation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the verified OpenEarthMap training pipeline into a modular registry-based package while preserving UNet++ behavior and adding SegFormer-B0 plus MambaVision-T + UPerNet.

**Architecture:** Keep `train.py` as a thin argparse entry point and move data, models, losses, optimization, metrics, evaluation, checkpointing, logging, and training into focused modules under `oemseg/`. All model adapters return `[B, 9, H, W]` logits and expose separate backbone/main parameter groups, allowing one shared engine to train UNet++, SegFormer, and MambaVision.

**Tech Stack:** Python 3.11, PyTorch in conda `work-env`, torchvision, segmentation-models-pytorch, Hugging Face Transformers, official NVIDIA MambaVision code/weights, timm, einops, mamba-ssm, rasterio, Pillow, tqdm, W&B, pytest.

## Global Constraints

- Use only `/home/duypham/miniconda3/envs/work-env`; do not modify system Python.
- Preserve 9 OEM classes including background and labels `[0, 8]`.
- Keep official OEM `val.txt` as the externally reported `test_*` split because official test labels are unavailable.
- Keep `test_oa` as overall pixel accuracy; do not add a duplicate `test_acc`.
- Preserve delayed evaluation defaults: first evaluation at `ceil(epochs * 2/3)`, then every 3 epochs, and always at the final epoch.
- Preserve default paper settings, TTA, W&B behavior, output layout, metric names, and checkpoint top-level keys.
- `python train.py` must still select UNet++/ResNet18/ImageNet and current default hyperparameters.
- New optional model dependencies must be imported lazily so UNet++ remains usable when they are absent.
- Do not install MMEngine, MMCV, MMSegmentation, MMDetection, or MMPretrain in `work-env` unless the standalone implementation is proven impossible.
- Use official NVIDIA MambaVision-T backbone and pretrained weights; do not rewrite the backbone.
- Implement only the compact UPerNet decoder needed by this project, following the official four stages `[80, 160, 320, 640]`.
- Do not rerun the full 45-epoch experiment; use unit tests and one-batch real-data GPU smoke tests.
- Run `chmod -R 777 /home/duypham/workspace/OEM_Segmentation` after final verification.

---

## File Map

**Create:**

- `oemseg/__init__.py` — package metadata and public entry points.
- `oemseg/constants.py` — OEM class names and class count.
- `oemseg/config.py` — argparse parser, normalization, validation, and legacy CLI mapping.
- `oemseg/data/dataset.py` — OEM sample discovery and `OEMDataset`.
- `oemseg/data/loaders.py` — deterministic train/internal-val/test loader creation.
- `oemseg/models/base.py` — common adapter contract and parameter-group helper.
- `oemseg/models/registry.py` — model registry and lazy builders.
- `oemseg/models/unetpp.py` — SMP UNet++ adapter.
- `oemseg/models/segformer.py` — Hugging Face SegFormer adapter.
- `oemseg/models/upernet.py` — compact PSP + FPN UPerNet decoder.
- `oemseg/models/mambavision.py` — official MambaVision feature backbone adapter plus UPerNet.
- `oemseg/losses/registry.py` — loss-name normalization and builder.
- `oemseg/losses/segmentation.py` — CE, Dice, and CE+Dice modules.
- `oemseg/optimizers/factory.py` — Adam/AdamW factory.
- `oemseg/schedulers/factory.py` — warmup + polynomial scheduler.
- `oemseg/metrics/segmentation.py` — confusion matrix and flattened metrics.
- `oemseg/engine/evaluator.py` — model-independent evaluation and TTA.
- `oemseg/engine/checkpoint.py` — compatible checkpoint save/load helpers.
- `oemseg/engine/trainer.py` — shared AMP/accumulation training loop.
- `oemseg/utils/logging.py` — file/console logger and JSONL formatting.
- `oemseg/utils/reproducibility.py` — deterministic seeding.
- `oemseg/utils/tta.py` — multi-scale and flip logits aggregation.
- package `__init__.py` files for each subdirectory.
- `tests/test_config.py`
- `tests/test_data.py`
- `tests/test_losses.py`
- `tests/test_metrics.py`
- `tests/test_factories.py`
- `tests/test_models.py`
- `tests/test_engine.py`
- `tests/test_training_smoke.py`

**Modify:**

- `train.py` — replace the 430-line implementation with a thin entry point.
- `requirements.txt` — add bounded optional model/test dependencies actually verified in `work-env`.
- `README.md` — document architecture, CLI, models, metrics, memory guidance, and licenses.
- `.gitignore` — ignore Hugging Face caches and temporary smoke outputs if needed.
- `tests/test_pipeline.py` — either remove after equivalent pytest coverage exists or retain as a tiny compatibility launcher.

---

### Task 1: Establish the package contract, registries, and CLI configuration

**Files:**
- Create: `oemseg/__init__.py`
- Create: `oemseg/constants.py`
- Create: `oemseg/config.py`
- Create: `oemseg/models/base.py`
- Create: `oemseg/models/registry.py`
- Create: package `__init__.py` files
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `CLASS_NAMES`, `NUM_CLASSES`.
- Produces: `SegmentationModelAdapter.parameter_groups(base_lr: float, backbone_lr: float) -> list[dict[str, object]]`.
- Produces: `register_model(name: str, aliases: tuple[str, ...] = ())`, `build_model(args: argparse.Namespace) -> SegmentationModelAdapter`, and `build_model_from_values(name: str, variant: str, pretrained: bool, decoder: str = "upernet") -> SegmentationModelAdapter`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `parse_args(argv: list[str] | None = None) -> argparse.Namespace`.

- [x] **Step 1: Write failing CLI and registry tests**

```python
# tests/test_config.py
from oemseg.config import parse_args
from oemseg.models.registry import available_models, normalize_name


def test_default_cli_preserves_unetpp_behavior():
    args = parse_args([])
    assert args.model == "unetpp"
    assert args.model_variant == "resnet18"
    assert args.loss == "ce_dice"
    assert args.optimizer == "adamw"
    assert args.epochs == 45
    assert args.grad_accumulation == 1


def test_legacy_encoder_maps_to_model_variant():
    args = parse_args(["--encoder", "resnet34", "--encoder-weights", "none"])
    assert args.model == "unetpp"
    assert args.model_variant == "resnet34"
    assert args.pretrained is False


def test_component_names_are_normalized():
    assert normalize_name("CE-Dice") == "ce_dice"
    assert "unetpp" in available_models()
```

- [x] **Step 2: Run tests and verify import failures**

Run:

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_config.py
```

Expected: collection fails because `oemseg` does not exist.

- [x] **Step 3: Implement constants, base adapter, registry, and parser**

Use this stable adapter contract:

```python
class SegmentationModelAdapter(nn.Module):
    backbone: nn.Module
    head: nn.Module

    def parameter_groups(self, base_lr: float, backbone_lr: float):
        backbone = list(self.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        main = [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]
        if not backbone or not main or backbone_ids & {id(parameter) for parameter in main}:
            raise RuntimeError("Model parameter groups must be nonempty and nonoverlapping")
        return [
            {"params": backbone, "lr": backbone_lr, "group_name": "backbone"},
            {"params": main, "lr": base_lr, "group_name": "main"},
        ]
```

Parser additions must include:

```text
--model {unetpp,segformer,mambavision}
--model-variant
--decoder
--pretrained / --no-pretrained
--loss
--optimizer
--grad-accumulation
```

Validation must reject `grad_accumulation < 1`, invalid evaluation fractions, and invalid component names using `parser.error(...)` with valid names in the message.

- [x] **Step 4: Run the tests**

Run:

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_config.py
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add oemseg tests/test_config.py
git commit -m "feat: add modular configuration and model registry"
```

---

### Task 2: Extract OEM data loading, metrics, reproducibility, and logging

**Files:**
- Create: `oemseg/data/dataset.py`
- Create: `oemseg/data/loaders.py`
- Create: `oemseg/metrics/segmentation.py`
- Create: `oemseg/utils/reproducibility.py`
- Create: `oemseg/utils/logging.py`
- Test: `tests/test_data.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `region_for(filename: str) -> str`.
- Produces: `read_split(root: Path, split: str) -> list[str]`.
- Produces: `OEMDataset(root: Path, names: list[str], size: int, augment: bool)`.
- Produces: `build_loaders(args) -> LoaderBundle` with `train`, `internal_val`, `test`, and sample counts.
- Produces: `ConfusionMatrix.compute() -> SegmentationMetrics`.
- Produces: `flatten_metrics(prefix: str, loss: float, metrics: SegmentationMetrics) -> dict[str, float]`.

- [x] **Step 1: Write focused data and metric tests**

```python
# tests/test_metrics.py
import torch
from oemseg.metrics.segmentation import ConfusionMatrix, flatten_metrics


def test_perfect_prediction_has_unit_metrics_and_test_oa():
    target = torch.tensor([[[0, 1], [2, 3]]])
    matrix = ConfusionMatrix(classes=9)
    matrix.update(target, target)
    metrics = matrix.compute()
    flattened = flatten_metrics("test", 0.25, metrics)
    assert metrics.oa == 1.0
    assert metrics.miou == 1.0
    assert flattened["test_oa"] == 1.0
    assert "test_acc" not in flattened
```

```python
# tests/test_data.py
from oemseg.data.dataset import region_for


def test_region_name_uses_last_underscore():
    assert region_for("little_rock_12.tif") == "little_rock"
```

- [x] **Step 2: Verify the tests fail before extraction**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_data.py tests/test_metrics.py
```

Expected: missing modules.

- [x] **Step 3: Move behavior without changing semantics**

Copy the proven image resize, nearest-neighbor mask resize, H/V augmentation, ImageNet normalization, split validation, confusion-matrix formulas, and metric prefixes from the old `train.py` into the new modules. Use a `LoaderBundle` dataclass so the trainer does not know split implementation details.

- [x] **Step 4: Verify tests and real dataset counts**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_data.py tests/test_metrics.py
/home/duypham/miniconda3/bin/conda run -n work-env python - <<'PY'
from pathlib import Path
from oemseg.data.dataset import read_split
root = Path('datasets/OpenEarthMap/OpenEarthMap')
assert len(read_split(root, 'train')) == 3000
assert len(read_split(root, 'val')) == 500
print('OEM split verification: OK')
PY
```

- [x] **Step 5: Commit**

```bash
git add oemseg/data oemseg/metrics oemseg/utils tests/test_data.py tests/test_metrics.py
git commit -m "refactor: extract OEM data metrics and utilities"
```

---

### Task 3: Add loss, optimizer, and scheduler factories

**Files:**
- Create: `oemseg/losses/segmentation.py`
- Create: `oemseg/losses/registry.py`
- Create: `oemseg/optimizers/factory.py`
- Create: `oemseg/schedulers/factory.py`
- Test: `tests/test_losses.py`
- Test: `tests/test_factories.py`

**Interfaces:**
- Produces: `build_loss(name: str) -> nn.Module`.
- Produces: `build_optimizer(name: str, parameter_groups, weight_decay: float) -> torch.optim.Optimizer`.
- Produces: `build_scheduler(optimizer, epochs: int, warmup_epochs: int, power: float)`.
- Produces: `should_evaluate(epoch: int, total_epochs: int, start_fraction: float, every: int) -> bool`.

- [x] **Step 1: Write failing factory tests**

```python
# tests/test_losses.py
import torch
from oemseg.losses.registry import build_loss


def test_registered_losses_have_finite_backward():
    for name in ("ce", "dice", "ce_dice", "ce-dice", "cedice"):
        logits = torch.randn(2, 9, 16, 16, requires_grad=True)
        target = torch.randint(0, 9, (2, 16, 16))
        loss = build_loss(name)(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
```

```python
# tests/test_factories.py
import torch
from oemseg.optimizers.factory import build_optimizer
from oemseg.schedulers.factory import build_scheduler, should_evaluate


def test_optimizer_and_poly_schedule():
    p1 = torch.nn.Parameter(torch.tensor(1.0))
    p2 = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = build_optimizer(
        "adamw",
        [{"params": [p1], "lr": 1e-4}, {"params": [p2], "lr": 1e-3}],
        weight_decay=0.01,
    )
    scheduler = build_scheduler(optimizer, epochs=10, warmup_epochs=2, power=0.9)
    for _ in range(10):
        optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert should_evaluate(30, 45, 2 / 3, 3)
    assert should_evaluate(45, 45, 2 / 3, 3)
```

- [x] **Step 2: Run and confirm failure**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_losses.py tests/test_factories.py
```

- [x] **Step 3: Implement minimal registered components**

Implement CE with `torch.nn.CrossEntropyLoss`, Dice with `smp.losses.DiceLoss(mode="multiclass", from_logits=True)`, and CE+Dice as an additive module. Normalize aliases through the same lowercase/underscore function used by model names. Optimizers must initially support exactly `adam` and `adamw`.

- [x] **Step 4: Run tests**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_losses.py tests/test_factories.py
```

- [x] **Step 5: Commit**

```bash
git add oemseg/losses oemseg/optimizers oemseg/schedulers tests/test_losses.py tests/test_factories.py
git commit -m "feat: add configurable losses optimizers and scheduler"
```

---

### Task 4: Implement the UNet++ adapter and prove old checkpoint compatibility

**Files:**
- Create: `oemseg/models/unetpp.py`
- Modify: `oemseg/models/registry.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `UNetPlusPlusAdapter(variant: str, pretrained: bool, num_classes: int = 9)`.
- `forward(images)` returns logits at exactly the input spatial size.
- `backbone` refers to `model.encoder`; `head` refers to the remaining model path for parameter grouping.

- [x] **Step 1: Write the UNet++ adapter tests**

```python
# tests/test_models.py
import torch
from oemseg.models.registry import build_model_from_values


def assert_adapter_contract(model):
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 9, 64, 64)
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


def test_unetpp_adapter_contract():
    model = build_model_from_values("unetpp", "resnet18", pretrained=False)
    assert_adapter_contract(model)
```

- [x] **Step 2: Verify failure**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_models.py::test_unetpp_adapter_contract
```

- [x] **Step 3: Implement adapter and registry builder**

Wrap `smp.UnetPlusPlus`; do not copy its implementation. Make the registry lazy-import `oemseg.models.unetpp` only when selected.

- [x] **Step 4: Verify old checkpoint state dictionary loads**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python - <<'PY'
import torch
from oemseg.models.registry import build_model_from_values
checkpoint = torch.load('outputs/unetplusplus-oem-paper/last.pt', map_location='cpu', weights_only=False)
model = build_model_from_values('unetpp', 'resnet18', pretrained=False)
model.load_state_dict(checkpoint['model'], strict=True)
print('legacy UNet++ checkpoint: OK')
PY
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_models.py
```

- [x] **Step 5: Commit**

```bash
git add oemseg/models tests/test_models.py
git commit -m "feat: add compatible UNet++ model adapter"
```

---

### Task 5: Extract evaluator, checkpointing, trainer, and thin entry point

**Files:**
- Create: `oemseg/utils/tta.py`
- Create: `oemseg/engine/evaluator.py`
- Create: `oemseg/engine/checkpoint.py`
- Create: `oemseg/engine/trainer.py`
- Modify: `train.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_training_smoke.py`

**Interfaces:**
- Produces: `model_logits(model, images, scales, flips) -> Tensor`.
- Produces: `evaluate(model, loader, criterion, device, scales, flips, max_batches=None)`.
- Produces: `save_checkpoint(path, model, optimizer, scheduler, epoch, args, metadata=None)`.
- Produces: `run_training(args) -> Path`.
- `train.py` contains only imports, `args = parse_args()`, and `run_training(args)`.

- [x] **Step 1: Write failing engine tests**

```python
# tests/test_engine.py
import argparse
import torch
from pathlib import Path
from oemseg.engine.checkpoint import save_checkpoint


def test_checkpoint_keeps_legacy_keys(tmp_path: Path):
    model = torch.nn.Conv2d(3, 9, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scheduler, 3, argparse.Namespace(model="unetpp"))
    checkpoint = torch.load(path, weights_only=False)
    assert {"epoch", "model", "optimizer", "scheduler", "args"} <= checkpoint.keys()
```

Add a gradient accumulation test with a two-batch synthetic loader and assert optimizer step count equals one when `grad_accumulation=2`.

- [x] **Step 2: Run and verify failure**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_engine.py tests/test_training_smoke.py
```

- [x] **Step 3: Move the proven engine and add accumulation correctly**

For each batch:

```python
scaled_loss = loss / args.grad_accumulation
scaler.scale(scaled_loss).backward()
should_step = ((batch_index + 1) % args.grad_accumulation == 0) or batch_index + 1 == len(train_loader)
if should_step:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

Record the unscaled loss for logging. Preserve `last.pt`, `best_train_loss.pt`, optional `best_val_miou.pt`, JSONL, W&B, `val_*`, `test_*`, TTA, and final `run_complete` logging.

- [x] **Step 4: Verify thin entry point and default smoke behavior**

```bash
/home/duypham/miniconda3/envs/work-env/bin/python - <<'PY'
from pathlib import Path
text = Path('train.py').read_text()
assert len(text.splitlines()) <= 20
assert 'class OEMDataset' not in text
assert 'def evaluate' not in text
print('thin train.py: OK')
PY
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_engine.py tests/test_training_smoke.py
/home/duypham/miniconda3/bin/conda run --no-capture-output -n work-env python train.py --smoke --run-name modular-unetpp-smoke
```

Verify `outputs/modular-unetpp-smoke/metrics.jsonl` contains `train_loss`, `test_oa`, `test_miou`, and all nine `test_iou_*` keys.

- [x] **Step 5: Commit**

```bash
git add train.py oemseg/engine oemseg/utils/tta.py tests/test_engine.py tests/test_training_smoke.py
git commit -m "refactor: move training and evaluation into shared engine"
```

---

### Task 6: Install and integrate SegFormer-B0

**Files:**
- Modify: `requirements.txt`
- Create: `oemseg/models/segformer.py`
- Modify: `oemseg/models/registry.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `SegFormerAdapter(variant="b0", pretrained=True, num_classes=9)`.
- Uses `nvidia/mit-b0` for the ImageNet-pretrained MiT-B0 encoder.
- Returns resized `[B, 9, H, W]` logits.
- Exposes `model.segformer` as backbone and `model.decode_head` as main head.

- [x] **Step 1: Add a failing no-download model-contract test**

Construct a tiny `SegformerConfig` in the test and inject it into the adapter through a private test-only constructor or classmethod. Assert forward/backward shape and parameter groups without internet access.

```python
def test_segformer_adapter_contract_without_download():
    model = build_model_from_values("segformer", "b0", pretrained=False)
    assert_adapter_contract(model)
```

- [x] **Step 2: Install the smallest verified dependency set in `work-env`**

Start with:

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python -m pip install \
  'transformers>=4.50,<5' 'safetensors>=0.5,<1' 'pytest>=8,<9'
```

Record the resolved versions with:

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python -m pip show transformers safetensors pytest
```

Pin/bound only versions that actually pass the tests.

- [x] **Step 3: Implement the adapter with verified pretrained loading**

Use `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=9, ignore_mismatched_sizes=True, output_loading_info=True)`. Validate loading information so missing keys are confined to the new decode/classifier head; raise a clear error when backbone keys are missing unexpectedly. For `pretrained=False`, instantiate `SegformerConfig` for B0 with `num_labels=9`.

Set `id2label` and `label2id` from OEM constants and keep label zero as a real background class. Do not use `do_reduce_labels=True`.

- [x] **Step 4: Run unit and pretrained GPU smoke tests**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_models.py
/home/duypham/miniconda3/bin/conda run --no-capture-output -n work-env python train.py \
  --model segformer --model-variant b0 --smoke \
  --run-name modular-segformer-smoke
```

Verify output shape, finite backward pass, checkpoint creation, and `test_oa` logging.

- [x] **Step 5: Commit**

```bash
git add requirements.txt oemseg/models/segformer.py oemseg/models/registry.py tests/test_models.py
git commit -m "feat: add SegFormer-B0 segmentation adapter"
```

---

### Task 7: Implement compact UPerNet and integrate official MambaVision-T

**Files:**
- Create: `oemseg/models/upernet.py`
- Create: `oemseg/models/mambavision.py`
- Modify: `oemseg/models/registry.py`
- Modify: `requirements.txt`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `UPerNetHead(in_channels=(80,160,320,640), channels=512, pool_scales=(1,2,3,6), num_classes=9)`.
- Produces: `MambaVisionAdapter(variant="tiny", pretrained=True, decoder="upernet", num_classes=9)`.
- Official backbone output must be four feature maps with channels `[80,160,320,640]`.

- [x] **Step 1: Write UPerNet shape and gradient tests**

```python
def test_upernet_decoder_contract():
    from oemseg.models.upernet import UPerNetHead
    decoder = UPerNetHead((80, 160, 320, 640), channels=64, num_classes=9)
    features = [
        torch.randn(1, 80, 32, 32, requires_grad=True),
        torch.randn(1, 160, 16, 16, requires_grad=True),
        torch.randn(1, 320, 8, 8, requires_grad=True),
        torch.randn(1, 640, 4, 4, requires_grad=True),
    ]
    logits = decoder(features)
    assert logits.shape == (1, 9, 32, 32)
    logits.mean().backward()
```

Add a fake-backbone adapter test that returns the same four feature shapes so the project integration is testable without downloading weights.

- [x] **Step 2: Implement only the required UPerNet components**

Implement:

- `PyramidPoolingModule`: adaptive pooling at scales `(1,2,3,6)`, `1x1 conv + BN + ReLU`, bilinear upsample, concatenate, bottleneck conv.
- top-down FPN: lateral `1x1` projections for stages 1–3, add upsampled deeper feature, `3x3` FPN conv.
- resize all FPN outputs to stage-1 size, concatenate, bottleneck, dropout, final `1x1` classifier.

Do not copy MMEngine runtime code or add auxiliary loss in this cycle.

- [x] **Step 3: Install and probe official MambaVision dependencies**

Use the official requirements as the starting point while preserving installed PyTorch:

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python -m pip install \
  'timm==1.0.15' 'einops==0.8.1' 'mamba-ssm==2.2.4' 'mambavision>=1.2,<2'
```

Do not allow pip to replace torch/torchvision. Use `--no-deps` for `mambavision` after manually satisfying its verified dependencies if resolver output attempts to downgrade or replace them.

Probe the official feature API:

```bash
/home/duypham/miniconda3/bin/conda run --no-capture-output -n work-env python - <<'PY'
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/MambaVision-T-1K', trust_remote_code=True)
model.eval()
with torch.inference_mode():
    _, features = model(torch.randn(1, 3, 224, 224))
assert [x.shape[1] for x in features] == [80, 160, 320, 640]
print([tuple(x.shape) for x in features])
PY
```

- [x] **Step 4: Implement MambaVision adapter**

Load the official model through `AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)` for pretrained mode. For random initialization, use the official package constructor `create_model("mamba_vision_T", pretrained=False)` and expose a feature-returning path; do not duplicate the backbone source.

Normalize backbone outputs into NCHW feature maps, validate exactly four stages and channel dimensions, pass them to `UPerNetHead`, and resize logits to input size. Missing optional packages must raise an error such as:

```text
MambaVision requires transformers, timm, einops, mamba-ssm, and the official mambavision package. Install requirements.txt in conda environment work-env.
```

- [x] **Step 5: Run CPU unit tests and RTX 3060 GPU smoke tests**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q tests/test_models.py
/home/duypham/miniconda3/bin/conda run --no-capture-output -n work-env python train.py \
  --model mambavision --model-variant tiny --decoder upernet \
  --batch-size 1 --eval-batch-size 1 --grad-accumulation 2 \
  --smoke --run-name modular-mambavision-smoke
```

Measure peak memory:

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
```

If 1024² smoke exceeds 12 GB, reduce UPerNet internal `channels` from 512 to 256 through an explicit model option while keeping the official MambaVision-T backbone and UPerNet topology. Document the verified value; do not silently reduce input size.

- [x] **Step 6: Commit**

```bash
git add requirements.txt oemseg/models/upernet.py oemseg/models/mambavision.py oemseg/models/registry.py tests/test_models.py
git commit -m "feat: add MambaVision-T with UPerNet"
```

---

### Task 8: Run full modular integration verification and compatibility checks

**Files:**
- Modify: tests as required by verified behavior
- Modify: implementation only for defects found by tests

**Interfaces:**
- Consumes all preceding modules.
- Produces a verified framework with identical external metrics/checkpoints and three functioning model choices.

- [x] **Step 1: Run the complete test suite and compilation**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python -m compileall -q train.py oemseg tests
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q
```

Expected: all tests pass without network-dependent unit tests.

- [x] **Step 2: Verify CLI discoverability and invalid-name messages**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python train.py --help
set +e
/home/duypham/miniconda3/bin/conda run -n work-env python train.py --model nonexistent 2>&1 | tee /tmp/oem-invalid-model.log
status=${PIPESTATUS[0]}
set -e
test "$status" -ne 0
grep -E 'unetpp|segformer|mambavision' /tmp/oem-invalid-model.log
```

- [x] **Step 3: Run one real OEM train/eval batch for all models**

Use unique run names and delete only failed temporary smoke directories before reruns:

```bash
for command in \
  "--model unetpp --model-variant resnet18 --run-name final-unetpp-smoke" \
  "--model segformer --model-variant b0 --run-name final-segformer-smoke" \
  "--model mambavision --model-variant tiny --decoder upernet --batch-size 1 --grad-accumulation 2 --run-name final-mambavision-smoke"
do
  /home/duypham/miniconda3/bin/conda run --no-capture-output -n work-env python train.py --smoke $command
done
```

- [x] **Step 4: Programmatically validate every smoke artifact**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python - <<'PY'
import json
from pathlib import Path
for name in ('final-unetpp-smoke', 'final-segformer-smoke', 'final-mambavision-smoke'):
    root = Path('outputs') / name
    record = json.loads((root / 'metrics.jsonl').read_text().splitlines()[-1])
    assert {'train_loss', 'test_oa', 'test_miou'} <= record.keys(), (name, record.keys())
    assert 'test_acc' not in record
    assert len([key for key in record if key.startswith('test_iou_')]) == 9
    checkpoint = __import__('torch').load(root / 'last.pt', map_location='cpu', weights_only=False)
    assert {'epoch', 'model', 'optimizer', 'scheduler', 'args'} <= checkpoint.keys()
print('all smoke artifacts: OK')
PY
```

- [x] **Step 5: Verify optional W&B remains operational**

```bash
WANDB_MODE=offline /home/duypham/miniconda3/bin/conda run --no-capture-output -n work-env \
  python train.py --model unetpp --smoke --wandb --wandb-mode offline \
  --run-name final-wandb-smoke
```

- [x] **Step 6: Commit any integration fixes**

```bash
git add oemseg train.py tests requirements.txt
git commit -m "test: verify modular segmentation integration"
```

Skip the commit only when there are no changes.

---

### Task 9: Document usage, versions, licensing, and finish the repository

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `docs/superpowers/plans/2026-08-02-modular-oem-segmentation.md` only to mark completed checkboxes during execution

**Interfaces:**
- Produces user-facing commands for all supported components and a reproducible environment record.

- [x] **Step 1: Capture exact verified versions**

```bash
/home/duypham/miniconda3/bin/conda run -n work-env python - <<'PY'
import importlib.metadata as m
for package in ('torch','torchvision','segmentation-models-pytorch','transformers','timm','einops','mamba-ssm','mambavision','rasterio','wandb','pytest'):
    try: print(f'{package}=={m.version(package)}')
    except m.PackageNotFoundError: print(f'{package}: NOT INSTALLED')
PY
```

Update `requirements.txt` with bounded ranges that include exactly the tested versions, while continuing to omit torch/torchvision because they are preinstalled in `work-env`.

- [x] **Step 2: Update README with verified commands**

Document:

```bash
python train.py
python train.py --model unetpp --model-variant resnet34 --loss ce_dice --optimizer adamw --epochs 30
python train.py --model segformer --model-variant b0 --loss ce_dice --epochs 30
python train.py --model mambavision --model-variant tiny --decoder upernet --batch-size 1 --grad-accumulation 2
```

Also explain:

- project tree and extension points,
- how to register a model/loss/optimizer,
- `test_oa` means overall pixel accuracy,
- official OEM validation is reported as `test_*`,
- delayed evaluation defaults,
- MambaVision source license is NVIDIA Source Code License-NC,
- pretrained MambaVision weights are CC-BY-NC-SA-4.0,
- memory settings verified on RTX 3060 12 GB.

- [x] **Step 3: Run final documentation and repository checks**

```bash
grep -nE 'unetpp|segformer|mambavision|test_oa|grad-accumulation' README.md
/home/duypham/miniconda3/envs/work-env/bin/python - <<'PY'
from pathlib import Path
needles = ['T' + 'BD', 'TO' + 'DO', 'FIX' + 'ME', 'X' + 'XX']
paths = [Path('README.md'), Path('requirements.txt')]
paths += [p for root in ('oemseg', 'tests') for p in Path(root).rglob('*') if p.is_file()]
for path in paths:
    if any(word in path.read_text(errors='ignore') for word in needles):
        raise SystemExit(f'placeholder found: {path}')
PY
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q
git diff --check
```

- [x] **Step 4: Commit documentation**

```bash
git add README.md requirements.txt .gitignore docs/superpowers/plans/2026-08-02-modular-oem-segmentation.md
git commit -m "docs: document modular OEM segmentation framework"
```

- [x] **Step 5: Apply requested permissions and verify clean state**

```bash
chmod -R 777 /home/duypham/workspace/OEM_Segmentation
git config core.fileMode false
git status --short
git log --oneline -12
```

Expected: `git status --short` is empty, all project paths are accessible, and no full training process is running.

---

## Final Acceptance Verification

Run once after all tasks:

```bash
cd /home/duypham/workspace/OEM_Segmentation
/home/duypham/miniconda3/bin/conda run -n work-env python -m compileall -q train.py oemseg tests
/home/duypham/miniconda3/bin/conda run -n work-env pytest -q
/home/duypham/miniconda3/bin/conda run -n work-env python - <<'PY'
from oemseg.config import parse_args
from oemseg.models.registry import available_models
args = parse_args([])
assert (args.model, args.model_variant, args.loss, args.optimizer) == ('unetpp', 'resnet18', 'ce_dice', 'adamw')
assert {'unetpp', 'segformer', 'mambavision'} <= set(available_models())
print('final framework contract: OK')
PY
test -z "$(git status --short)"
```

The implementation is complete only after all commands succeed and real-data GPU smoke outputs exist for UNet++, SegFormer-B0, and MambaVision-T + UPerNet.


## Execution Notes

- Implemented on branch `modular-oem-segmentation` in an isolated worktree.
- SegFormer uses `transformers==4.50.0` and `nvidia/mit-b0`.
- `mamba-ssm==2.2.4` could not build with CUDA 13.2 because its build script emits GPU architectures removed by CUDA 13. The compatible official 2.2-line release `mamba-ssm==2.2.6.post3` was verified instead.
- Both pretrained and random-initialized MambaVision-T use NVIDIA's Hugging Face remote implementation pinned to revision `b1de77e17599566d98efb701c0231b1095dc3a67`. This removes the dependency conflict introduced by the standalone `mambavision==1.2.0` package while retaining official model code.
- Real OEM 1024×1024 smoke runs were completed for UNet++, SegFormer-B0, and MambaVision-T + UPerNet. The MambaVision smoke used UPerNet channels 512 and observed approximately 3555 MiB process VRAM.
- The verified feature branch was fast-forward merged into `unetplusplus-oem`; its worktree and branch were removed, and project permissions were applied recursively as `777`.
