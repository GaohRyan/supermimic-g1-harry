#!/usr/bin/env bash
# One-time local setup for the CPU-only offline tools (no mjlab runtime, no GPU).
#   * pinned mjlab source (math helpers + G1 MJCF/meshes) -> upstream/references/mjlab-8bb2bc9 (gitignored)
#   * a stub `mjlab` package exposing only utils/lab_api/math.py and asset_zoo -> tools/offline/.mjlab_stub (gitignored)
# Then:  source tools/offline/env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
REV=8bb2bc9e187797689ec82c78c35019afd2fd0376
SRC=upstream/references/mjlab-8bb2bc9
if [ ! -f "$SRC/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml" ]; then
  mkdir -p upstream/references
  git clone -q --filter=blob:none https://github.com/mujocolab/mjlab "$SRC.tmp"
  git -C "$SRC.tmp" checkout -q "$REV"
  mv "$SRC.tmp" "$SRC"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$REV" ] || { echo "mjlab checkout is not at $REV" >&2; exit 1; }
STUB=tools/offline/.mjlab_stub/mjlab
mkdir -p "$STUB/utils/lab_api"
: > "$STUB/__init__.py"; : > "$STUB/utils/__init__.py"; : > "$STUB/utils/lab_api/__init__.py"
ln -sfn "$PWD/$SRC/src/mjlab/utils/lab_api/math.py" "$STUB/utils/lab_api/math.py"
ln -sfn "$PWD/$SRC/src/mjlab/asset_zoo" "$STUB/asset_zoo"
echo "ok: $SRC @ ${REV:0:7}; g1.xml sha256 $(shasum -a 256 "$SRC/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml" | cut -c1-16)"
