#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/setup_unetformer.sh"
PYTHON="${PYTHON:-python}"
MAMBA_VERSION="2.3.2.post1"
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

"$PYTHON" - <<'PY'
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}"
)
if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: CUDA-enabled PyTorch is required; do not let pip replace the preinstalled torch"
    )
PY

"$PYTHON" -m pip install -U setuptools wheel packaging ninja
grep -vE '^mamba-ssm([<=> ]|$)' "$ROOT/requirements.txt" > "$TMP_REQ"
"$PYTHON" -m pip install -r "$TMP_REQ"

mapfile -t MAMBA_INFO < <("$PYTHON" - <<'PY'
import platform
import sys
import torch

if not torch.version.cuda:
    raise SystemExit("ERROR: torch.version.cuda is unavailable")

python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_parts = torch.__version__.split("+", 1)[0].split(".")
torch_tag = ".".join(torch_parts[:2])
cuda_major = torch.version.cuda.split(".", 1)[0]
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
platform_tag = f"linux_{platform.machine()}"

print(f"cu{cuda_major}torch{torch_tag}cxx11abi{abi}")
print(python_tag)
print(platform_tag)
PY
)

MAMBA_TAG="${MAMBA_INFO[0]}"
PYTHON_TAG="${MAMBA_INFO[1]}"
PLATFORM_TAG="${MAMBA_INFO[2]}"
MAMBA_WHEEL="mamba_ssm-${MAMBA_VERSION}+${MAMBA_TAG}-${PYTHON_TAG}-${PYTHON_TAG}-${PLATFORM_TAG}.whl"
MAMBA_WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VERSION}/${MAMBA_WHEEL}"

echo "Installing prebuilt Mamba wheel:"
echo "  ${MAMBA_WHEEL_URL}"
if ! "$PYTHON" -m pip install --no-deps "$MAMBA_WHEEL_URL"; then
    echo >&2 "ERROR: no compatible prebuilt mamba-ssm wheel was installable for this Python/PyTorch/CUDA/ABI stack."
    echo >&2 "Refusing to fall back to a long source compilation."
    echo >&2 "Detected wheel tag: ${MAMBA_TAG}-${PYTHON_TAG}-${PLATFORM_TAG}"
    exit 1
fi

"$PYTHON" - <<'PY'
import accelerate
import mamba_ssm
import selective_scan_cuda
import torch
import wandb
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

print(
    f"environment OK: torch={torch.__version__} "
    f"accelerate={accelerate.__version__} wandb={wandb.__version__}"
)
print(f"mamba_ssm={getattr(mamba_ssm, '__version__', 'installed')}")
print(f"selective_scan_cuda={selective_scan_cuda.__file__}")

# Run the actual CUDA extension once so ABI/import success alone cannot hide a runtime mismatch.
device = torch.device("cuda")
u = torch.randn(1, 2, 4, device=device)
delta = torch.randn(1, 2, 4, device=device)
A = -torch.rand(2, 2, device=device)
B = torch.randn(1, 2, 4, device=device)
C = torch.randn(1, 2, 4, device=device)
D = torch.ones(2, device=device)
out = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=True)
torch.cuda.synchronize()
if out.shape != u.shape or not torch.isfinite(out).all():
    raise SystemExit("ERROR: selective_scan CUDA smoke test returned an invalid result")
print("selective_scan_cuda_smoke=OK")
PY
