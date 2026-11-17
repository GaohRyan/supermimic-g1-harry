#!/usr/bin/env bash
# Evaluation only: the terrain checkpoint shipped as $B/init_ckpt.pt under the training-time perturbations
# (sensor / action / scan noise, domain randomisation incl. sim-to-real randomisation and action delay, reset noise),
# next to the nominal protocol on the same Pod. Runs inside the generated run_remote.sh (cwd = $W/code); steps are non-fatal.
set -uo pipefail
W=/workspace/job; B=$W/bundle; O=$W/out
PY="env -u PYTHONPATH uv run --frozen python"
step(){ echo "=== $(date -u +%FT%TZ) $*"; }
C=$B/init_ckpt.pt
TEV(){ MUJOCO_GL=egl timeout 900 $PY scripts/eval_terrain.py --ckpt $C --out $O/eval/$1 --npz-dir $W/npz40 $2 > $O/eval/$1.log 2>&1; echo "terrain eval $1 exit=$?" | tee -a $O/exits.txt; grep terrain_summary $O/eval/$1.log | cut -c1-200; rm -f $O/eval/$1/per_env.pt; }
EV(){ MUJOCO_GL=egl timeout ${3:-900} $PY scripts/eval_fullclip_traj.py --ckpt $C --out $O/eval/$1 --npz-dir $W/npz40 $2 > $O/eval/$1.log 2>&1; echo "eval $1 exit=$?" | tee -a $O/exits.txt; tail -1 $O/eval/$1.log | cut -c1-200; rm -f $O/eval/$1/trajectories.npz $O/eval/$1/render_input.npz; }
step terrain-nominal; TEV terrain_nominal "--per-tile 128"
step terrain-robust; TEV terrain_robust "--per-tile 128 --robust"
step terrain-robust-seed1; TEV terrain_robust_seed1 "--per-tile 128 --robust --seed 1"
step flat-robust; EV full_robust "--obs-noise --domain-randomization --reset-noise --action-noise-std 0.1 --data-label LAFAN40-train-coverage" 1200
step robust-eval-done
