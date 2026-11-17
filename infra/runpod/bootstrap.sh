#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo "Run bootstrap on the Linux NVIDIA training Pod." >&2
  exit 1
fi
command -v uv >/dev/null || { echo "Install uv in the Pod image before bootstrap." >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA driver is not visible." >&2; exit 1; }
bash "$project_root/infra/runpod/system_dependencies.sh"
python3 - "$project_root" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts'))
from new_run import source_state
state = source_state(root)
assert state['upstream']['commit'] == 'ba6105349069bf040f1fdad60873d13138d505a2', 'Unexpected source revision'
print('Source integrity verified:', state['mode'])
PY
cd "$project_root/upstream/SuperMimic-Reproduced"
uv sync --frozen --group dev --python 3.12
env -u PYTHONPATH uv run --frozen python "$project_root/infra/runpod/preflight.py"
