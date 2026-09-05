#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/duy18102004/oem-dataset/OpenEarthMap_Prepared}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/oem_outputs}"
WANDB_ENTITY="${WANDB_ENTITY:-phamdinhanhduy-university-of-information-and-technology}"
WANDB_PROJECT="${WANDB_PROJECT:-sensing image segmentation}"
SMOKE="${SMOKE:-0}"

EPOCHS=45
IMAGE_SIZE=1024
BATCH_SIZE=1
GRAD_ACCUMULATION=1
EVAL_BATCH_SIZE=1
WORKERS=2
RUN_NAME="${MODEL_NAME}-paper-repro-t4x2"

export TORCH_HOME="${TORCH_HOME:-/kaggle/tmp/torch_cache}"
export HF_HOME="${HF_HOME:-/kaggle/tmp/hf_cache}"
export WANDB_MODE=offline
mkdir -p "$TORCH_HOME" "$HF_HOME" "$OUTPUT_ROOT"

case "$MODEL_NAME" in
  unet|unetformer|segformer|segnext|repstdc|mambavision|pyramidmamba|mask2former) ;;
  *) echo "ERROR: unsupported MODEL_NAME=$MODEL_NAME" >&2; exit 2 ;;
esac

if [[ ! -d "$DATA_ROOT" ]]; then
  fallback="/kaggle/input/oem-dataset/OpenEarthMap_Prepared"
  [[ -d "$fallback" ]] && DATA_ROOT="$fallback"
fi
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: OpenEarthMap dataset not mounted" >&2; exit 3; }

python3 - "$DATA_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for split, expected in {"train": 3000, "val": 500, "test": 1500}.items():
    path = root / f"{split}.txt"
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(names) != expected:
        raise SystemExit(f"ERROR: {split} expected {expected}, got {len(names)}")
    print(f"{split}: {len(names)}")
print("DATASET_OK")
PY

python3 - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit(f"ERROR: need T4 x2; CUDA={torch.cuda.is_available()} count={torch.cuda.device_count()}")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])
PY

setup_standard() {
  local req=/tmp/oem_requirements_no_mamba.txt
  grep -vE '^mamba-ssm([<=> ]|$)' requirements.txt > "$req"
  python3 -m pip install --disable-pip-version-check -r "$req"
  RUN_PYTHON=(python3)
}

setup_openmmlab() {
  local runner
  if command -v conda >/dev/null 2>&1; then
    runner="$(command -v conda)"
  else
    local bin_dir=/kaggle/working/.micromamba-bin
    local mm="$bin_dir/micromamba"
    export MAMBA_ROOT_PREFIX=/kaggle/working/.micromamba
    mkdir -p "$bin_dir" "$MAMBA_ROOT_PREFIX"
    if [[ ! -x "$mm" ]]; then
      local tmp
      tmp="$(mktemp -d)"
      curl --fail --location --silent --show-error \
        https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$tmp" bin/micromamba
      mv "$tmp/bin/micromamba" "$mm"
      rm -rf "$tmp"
      chmod +x "$mm"
    fi
    "$mm" config append channels conda-forge >/dev/null 2>&1 || true
    "$mm" config set channel_priority strict >/dev/null 2>&1 || true
    runner="$mm"
  fi
  CONDA_EXE="$runner" bash scripts/setup_openmmlab_baselines.sh
  RUN_PYTHON=("$runner" run -n oem-openmmlab python)
}

case "$MODEL_NAME" in
  segnext|repstdc)
    setup_openmmlab
    ;;
  mambavision|pyramidmamba)
    bash scripts/setup_env.sh
    RUN_PYTHON=(python3)
    ;;
  unetformer)
    bash scripts/setup_unetformer.sh
    setup_standard
    ;;
  *)
    setup_standard
    ;;
esac

SMOKE_ARGS=()
[[ "$SMOKE" == "1" ]] && SMOKE_ARGS=(--smoke)

printf '%s\n' \
  "model=$MODEL_NAME" \
  "epochs=$EPOCHS" \
  "image_size=$IMAGE_SIZE" \
  "batch_per_gpu=$BATCH_SIZE" \
  "global_batch=2" \
  "grad_accumulation=$GRAD_ACCUMULATION" \
  "precision=fp32" \
  "internal_val_fraction=0" \
  "checkpoint_selection=train_loss" \
  "loss=auto" \
  "wandb=offline"

"${RUN_PYTHON[@]}" scripts/launch.py distributed \
  --gpus 0,1 \
  --model "$MODEL_NAME" \
  -- \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --grad-accumulation "$GRAD_ACCUMULATION" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --workers "$WORKERS" \
  --optimizer adamw \
  --lr 6e-4 \
  --encoder-lr 6e-5 \
  --weight-decay 0.01 \
  --warmup-epochs 5 \
  --poly-power 0.9 \
  --val-fraction 0 \
  --patience 5 \
  --mixed-precision no \
  --loss auto \
  --wandb \
  --wandb-mode offline \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT" \
  "${SMOKE_ARGS[@]}"

find "$OUTPUT_ROOT" -name best_checkpoint_summary.json -print -exec cat {} \; || true
find "$OUTPUT_ROOT" -type d -path '*/wandb/offline-run-*' -print || true
