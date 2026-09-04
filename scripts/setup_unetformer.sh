#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.vendor/GeoSeg"
REPO="https://github.com/WangLibo1995/GeoSeg.git"
REV="9453fe48209c4626b29e35e61bab93b61212c4b1"

if [[ -d "$DEST/.git" ]] && [[ "$(git -C "$DEST" rev-parse HEAD)" == "$REV" ]]; then
    echo "GeoSeg source already pinned at $REV"
    exit 0
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
git clone --filter=blob:none --no-checkout "$REPO" "$DEST"
git -C "$DEST" checkout --detach "$REV"
echo "GeoSeg source pinned at $REV"
