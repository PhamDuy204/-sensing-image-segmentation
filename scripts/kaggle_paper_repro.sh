#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/duy18102004/oem-dataset/OpenEarthMap_Prepared}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/oem_outputs}"
WANDB_ENTITY="${WANDB_ENTITY:-phamdinhanhduy-university-of-information-and-technology}"
WANDB_PROJECT="${WANDB_PROJECT:-sensing image segmentation}"
SMOKE="${SMOKE:-0}"
ACCELERATOR_KIND="${ACCELERATOR_KIND:-T4X2}"
CHUNK_END_EPOCH="${CHUNK_END_EPOCH:-0}"
RESUME_FROM_INPUT="${RESUME_FROM_INPUT:-0}"

EPOCHS=45
IMAGE_SIZE=1024
BATCH_SIZE=1
GRAD_ACCUMULATION=1
EVAL_BATCH_SIZE=1
WORKERS=2
GPU_IDS="0,1"
EXPECTED_GPU_COUNT=2
RUN_SUFFIX="t4x2"
LR=6e-4
ENCODER_LR=6e-5
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=0
WARMUP_EPOCHS=5
PATIENCE=5
EVAL_START_EPOCH=30

if (( CHUNK_END_EPOCH > 0 )); then
  EVAL_START_EPOCH=44
fi

if [[ "$MODEL_NAME" == "unet" ]]; then
  BATCH_SIZE=2
  GPU_IDS="0"
  EXPECTED_GPU_COUNT=1
  RUN_SUFFIX="p100"
elif [[ "$MODEL_NAME" == "mask2former" ]]; then
  LR=1e-4
  ENCODER_LR=1e-5
  WEIGHT_DECAY=0.05
  MAX_GRAD_NORM=0.01
  WARMUP_EPOCHS=0
  PATIENCE=0
fi

RUN_NAME="${MODEL_NAME}-paper-repro-${RUN_SUFFIX}"
SMOKE_SUFFIX=""
[[ "$SMOKE" == "1" ]] && SMOKE_SUFFIX="-smoke"
RUN_NAME="${RUN_NAME}${SMOKE_SUFFIX}"
GLOBAL_BATCH=$((BATCH_SIZE * EXPECTED_GPU_COUNT * GRAD_ACCUMULATION))

export TORCH_HOME="${TORCH_HOME:-/kaggle/tmp/torch_cache}"
export HF_HOME="${HF_HOME:-/kaggle/tmp/hf_cache}"
export WANDB_MODE=offline
mkdir -p "$TORCH_HOME" "$HF_HOME" "$OUTPUT_ROOT"

case "$MODEL_NAME" in
  unet|unetformer|segformer|segnext|repstdc|mambavision|pyramidmamba|mask2former) ;;
  *) echo "ERROR: unsupported MODEL_NAME=$MODEL_NAME" >&2; exit 2 ;;
esac

if [[ "$MODEL_NAME" == "unet" ]]; then
  [[ "$ACCELERATOR_KIND" == "P100" ]] || { echo "ERROR: U-Net reproduction requires P100 single-GPU topology" >&2; exit 2; }
else
  [[ "$ACCELERATOR_KIND" == "T4X2" ]] || { echo "ERROR: $MODEL_NAME requires T4x2 in this workflow" >&2; exit 2; }
fi

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

if [[ "$MODEL_NAME" == "unet" ]]; then
  # Kaggle's current default cu128 image omits Pascal/sm_60 kernels used by P100.
  python3 -m pip install --disable-pip-version-check --upgrade \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu118
fi

python3 - "$EXPECTED_GPU_COUNT" "$MODEL_NAME" <<'PY'
import sys
import torch
expected = int(sys.argv[1])
model = sys.argv[2]
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count < expected:
    raise SystemExit(f"ERROR: need {expected} GPU(s); CUDA={torch.cuda.is_available()} count={count}")
# Exercise CUDA, because Kaggle's default P100 image can report CUDA available yet fail on first kernel.
probe = torch.ones(1, device="cuda")
if probe.item() != 1:
    raise SystemExit("ERROR: CUDA compute probe failed")
if model == "unet" and torch.cuda.get_device_capability(0) == (6, 0) and "sm_60" not in torch.cuda.get_arch_list():
    raise SystemExit(f"ERROR: installed PyTorch lacks P100/sm_60 kernels: {torch.cuda.get_arch_list()}")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(expected)])
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "arch:", torch.cuda.get_arch_list())
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
    local bin_dir=/kaggle/tmp/.micromamba-bin
    local mm="$bin_dir/micromamba"
    export MAMBA_ROOT_PREFIX=/kaggle/tmp/.micromamba
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

if [[ "$MODEL_NAME" == "segnext" || "$MODEL_NAME" == "repstdc" ]]; then
  if (( CHUNK_END_EPOCH > 0 )) || [[ "$RESUME_FROM_INPUT" == "1" ]]; then
    echo "ERROR: native OpenMMLab SegNeXt/RepSTDC runs are iteration-based and do not use epoch chunks" >&2
    exit 4
  fi
  export WANDB_PROJECT WANDB_ENTITY WANDB_MODE
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  NATIVE_SMOKE_ARGS=()
  [[ "$SMOKE" == "1" ]] && NATIVE_SMOKE_ARGS=(--smoke)
  "${RUN_PYTHON[@]}" -m torch.distributed.run --standalone --nproc_per_node="$EXPECTED_GPU_COUNT" \
    scripts/openmmlab_paper_baseline.py \
    --model "$MODEL_NAME" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --run-name "$RUN_NAME" \
    "${NATIVE_SMOKE_ARGS[@]}"
  find "$OUTPUT_ROOT" -name best_checkpoint_summary.json -print -exec cat {} \; || true
  find "$OUTPUT_ROOT" -type d -path '*/wandb/offline-run-*' -print || true
  exit 0
fi

RESUME_ARGS=()
if [[ "$RESUME_FROM_INPUT" == "1" ]]; then
  mapfile -t RESUME_CANDIDATES < <(
    find /kaggle/input -type f -path "*/oem_outputs/${RUN_NAME}/last.pt" -print
  )
  [[ ${#RESUME_CANDIDATES[@]} -eq 1 ]] || {
    echo "ERROR: expected exactly one resume checkpoint, found ${#RESUME_CANDIDATES[@]}" >&2
    printf '%s\n' "${RESUME_CANDIDATES[@]}" >&2
    exit 4
  }
  RESUME_CHECKPOINT="${RESUME_CANDIDATES[0]}"
  RESUME_ARGS=(--resume-from "$RESUME_CHECKPOINT")
fi

CHUNK_ARGS=()
if (( CHUNK_END_EPOCH > 0 )); then
  CHUNK_ARGS=(--stop-after-epoch "$CHUNK_END_EPOCH")
fi

SMOKE_ARGS=()
[[ "$SMOKE" == "1" ]] && SMOKE_ARGS=(--smoke)

printf '%s\n' \
  "model=$MODEL_NAME" \
  "epochs=$EPOCHS" \
  "image_size=$IMAGE_SIZE" \
  "accelerator=$ACCELERATOR_KIND" \
  "gpus=$GPU_IDS" \
  "batch_per_gpu=$BATCH_SIZE" \
  "global_batch=$GLOBAL_BATCH" \
  "grad_accumulation=$GRAD_ACCUMULATION" \
  "precision=fp32" \
  "lr=$LR" \
  "encoder_lr=$ENCODER_LR" \
  "weight_decay=$WEIGHT_DECAY" \
  "max_grad_norm=$MAX_GRAD_NORM" \
  "warmup_epochs=$WARMUP_EPOCHS" \
  "patience=$PATIENCE" \
  "eval_start_epoch=$EVAL_START_EPOCH" \
  "chunk_end_epoch=$CHUNK_END_EPOCH" \
  "resume_from_input=$RESUME_FROM_INPUT" \
  "internal_val_fraction=0" \
  "checkpoint_selection=train_loss" \
  "loss=auto" \
  "wandb=offline"

"${RUN_PYTHON[@]}" scripts/launch.py distributed \
  --gpus "$GPU_IDS" \
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
  --lr "$LR" \
  --encoder-lr "$ENCODER_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --poly-power 0.9 \
  --eval-start-epoch "$EVAL_START_EPOCH" \
  --val-fraction 0 \
  --patience "$PATIENCE" \
  --mixed-precision no \
  --loss auto \
  --wandb \
  --wandb-mode offline \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT" \
  "${RESUME_ARGS[@]}" \
  "${CHUNK_ARGS[@]}" \
  "${SMOKE_ARGS[@]}"

find "$OUTPUT_ROOT" -name best_checkpoint_summary.json -print -exec cat {} \; || true
find "$OUTPUT_ROOT" -type d -path '*/wandb/offline-run-*' -print || true
