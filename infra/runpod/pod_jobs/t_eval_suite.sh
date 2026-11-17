#!/usr/bin/env bash
# T (night 2026-11-10): the same check-up for every policy shipped in the bundle as EV_<name>.pt. No training.
#   For each policy: robust-v1 full clips on LAFAN40 (sensor noise + DR + reset noise, seed 0) and BONES40 holdout
#   (nominal, whole clips; these are short, 2-33 s). Policies named *_final also get robust-v1 10 s windows.
# Extra bundle file: bones-40csv-2f59b20.tar.gz
# Runs inside the generated run_remote.sh (cwd = $W/code). Every step is non-fatal and logs its exit code.
set -uo pipefail
W=/workspace/job; B=$W/bundle; O=$W/out
PY="env -u PYTHONPATH uv run --frozen python"
step(){ echo "=== $(date -u +%FT%TZ) $*"; }
EV(){ MUJOCO_GL=egl timeout ${5:-900} $PY scripts/eval_fullclip_traj.py --ckpt $1 --out $2 --npz-dir $3 $4 > $2.log 2>&1; echo "eval $(basename $2) exit=$?" | tee -a $O/exits.txt; tail -1 $2.log | cut -c1-200
      [ -f $2/trajectories.npz ] && $PY $B/traj_geom.py $2/trajectories.npz $2/geom.json > /dev/null 2>&1; rm -f $2/render_input.npz $2/trajectories.npz; }
L="--data-label LAFAN40-train-coverage"; R="--obs-noise --domain-randomization --reset-noise"

step bones40-convert
mkdir -p $W/rawbones && tar xzf $B/bones-40csv-2f59b20.tar.gz -C $W/rawbones && timeout 900 $PY scripts/data/csv_to_npz_batch.py --input-dir $W/rawbones/csv --output-dir $W/npzbones --input-fps 120 --output-fps 50 --workers 8 > $O/convert_bones.log 2>&1; echo "convert_bones_exit=$?" | tee -a $O/exits.txt
(cd $W/npzbones && sha256sum *.npz) > $O/npzbones.sha256

for f in $B/EV_*.pt; do
  [ -f "$f" ] || continue
  k=$(basename $f .pt); k=${k#EV_}
  step check-up-$k
  EV $f $O/eval/full_robust_$k $W/npz40 "$R $L"
  EV $f $O/eval/full_bones40_$k $W/npzbones "--data-label BONES40-holdout-unseen-clips"
done
for f in $B/EV_*_final.pt; do
  [ -f "$f" ] || continue
  k=$(basename $f .pt); k=${k#EV_}
  EV $f $O/eval/win_robust_$k $W/npz40 "--window-steps 500 $R $L" 600
done
