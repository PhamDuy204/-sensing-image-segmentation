#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.vendor/GeoSeg"
REPO="https://github.com/WangLibo1995/GeoSeg.git"
REV="9453fe48209c4626b29e35e61bab93b61212c4b1"
WEIGHT="$DEST/pretrain_weights/stseg_base.pth"
WEIGHT_URL="https://drive.usercontent.google.com/download?id=1tHNxQUffwNIfWFDKa4ql1klKGjCnbhwv&export=download&confirm=t"
WEIGHT_SHA256="00585aef2972df877f53db302eebbb9b9162821a328f8200bf10f758a9d8ed78"
SOURCE_ONLY="${1:-}"

if [[ -d "$DEST/.git" ]] && [[ "$(git -c safe.directory="$DEST" -C "$DEST" rev-parse HEAD)" == "$REV" ]]; then
    echo "GeoSeg source already pinned at $REV"
else
    rm -rf "$DEST"
    mkdir -p "$(dirname "$DEST")"
    git clone --filter=blob:none --no-checkout "$REPO" "$DEST"
    git -c safe.directory="$DEST" -C "$DEST" checkout --detach "$REV"
    echo "GeoSeg source pinned at $REV"
fi

if [[ "$SOURCE_ONLY" == "--source-only" ]]; then
    exit 0
fi

mkdir -p "$(dirname "$WEIGHT")"
if [[ -f "$WEIGHT" ]] && echo "$WEIGHT_SHA256  $WEIGHT" | sha256sum --check --status; then
    echo "GeoSeg Swin-B weight already verified"
else
    tmp="$WEIGHT.tmp"
    rm -f "$tmp"
    curl --fail --location --retry 3 --show-error "$WEIGHT_URL" --output "$tmp"
    echo "$WEIGHT_SHA256  $tmp" | sha256sum --check --status
    mv "$tmp" "$WEIGHT"
    echo "GeoSeg Swin-B weight downloaded and verified"
fi
