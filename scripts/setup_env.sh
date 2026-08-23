#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MAX_JOBS="${MAX_JOBS:-2}"
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

"$PYTHON" - <<'PY'
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA-enabled PyTorch is required; do not let pip replace Kaggle's preinstalled torch")
PY

"$PYTHON" -m pip install -U setuptools wheel packaging ninja
grep -vE '^mamba-ssm([<=> ]|$)' "$ROOT/requirements.txt" > "$TMP_REQ"
"$PYTHON" -m pip install -r "$TMP_REQ"
MAX_JOBS="$MAX_JOBS" "$PYTHON" -m pip install --no-build-isolation 'mamba-ssm==2.2.6.post3'

"$PYTHON" - <<'PY'
import accelerate, mamba_ssm, torch, wandb
print(f"environment OK: torch={torch.__version__} accelerate={accelerate.__version__} wandb={wandb.__version__}")
print(f"mamba_ssm={getattr(mamba_ssm, '__version__', 'installed')}")
PY
