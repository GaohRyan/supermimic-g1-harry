#!/usr/bin/env bash
# Hook run by infra/runpod/session_driver.py after each recovered session: offline summaries (no GPU).
# Usage: tools/offline/after_session.sh <session_dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
S="$1"; mkdir -p "$S/analysis"
python3 -I tools/offline/session_summary.py "$S" > "$S/analysis/session_summary.log" 2>&1
source tools/offline/env.sh
CK=$(ls "$S"/recovered/out/train_logs/*/ckpt_latest.pt 2>/dev/null | head -1)
TR=$(ls "$S"/recovered/out/eval/full_*ckpt_latest/trajectories.npz 2>/dev/null | head -1)
if [ -n "$CK" ] && [ -n "$TR" ]; then
  python3 tools/offline/failure_diagnosis.py --ckpt "$CK" --traj "$TR" --out "$S/analysis/failure_diagnosis_final.json" > "$S/analysis/failure_diagnosis.log" 2>&1
fi
# terrain task: per-tile evaluation tables and the curriculum (only when the session has terrain evaluations)
if ls "$S"/recovered/out/eval/terrain_*/results.json > /dev/null 2>&1; then
  python3 -I tools/offline/terrain_summary.py "$S" > "$S/analysis/terrain_summary.log" 2>&1
fi
tail -4 "$S/analysis/session_summary.log" 2>/dev/null | cut -c1-220
exit 0
