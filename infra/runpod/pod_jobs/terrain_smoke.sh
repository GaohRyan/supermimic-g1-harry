#!/usr/bin/env bash
# Terrain level 1, first contact with the real training stack (mjlab + MuJoCo-Warp). No real training.
#   Bundle: built with --no-train from the terrain config, plus --init-ckpt (the flat-ground checkpoint the line starts
#   from) and this script as --post-cmd. The generated run_remote.sh has already run the unit tests and
#   check_terrain.py (simulator-side reward / termination / height map vs the analytic path) before this script.
#   1. init        the Trainer builds the terrain env (4096 envs), expands the networks, one iteration  -> logs/init
#   2. train       up to 500 iterations (at most 6 minutes) from the warm start: the whole loop once; it/s and the
#                  sim / wm / policy split
#   3. terrain evaluation of the warm start: per tile, and once more with the height-field tiles added
#      (is foot contact on a flat height field different from the flat box tile in this simulator?)
#   4. flat-ground windows of the warm start: must reproduce the flat-ground checkpoint it came from
#   5. resume      about 2 minutes more from the training checkpoint (terrain levels restored)
#   6. terrain evaluation of the checkpoint after the short training (sanity only)
# Runs inside the generated run_remote.sh (cwd = $W/code). Every step is non-fatal and logs its exit code.
set -uo pipefail
W=/workspace/job; B=$W/bundle; O=$W/out
PY="env -u PYTHONPATH uv run --frozen python"
step(){ echo "=== $(date -u +%FT%TZ) $*"; }
EV(){ MUJOCO_GL=egl timeout ${4:-900} $PY scripts/eval_fullclip_traj.py --ckpt $1 --out $2 --npz-dir $W/npz40 $3 > $2.log 2>&1; echo "eval $(basename $2) exit=$?" | tee -a $O/exits.txt; tail -1 $2.log | cut -c1-200; rm -f $2/trajectories.npz $2/render_input.npz; }
TEV(){ MUJOCO_GL=egl timeout ${4:-900} $PY scripts/eval_terrain.py --ckpt $1 --out $2 --npz-dir $W/npz40 $3 > $2.log 2>&1; echo "terrain eval $(basename $2) exit=$?" | tee -a $O/exits.txt; grep -B30 terrain_summary $2.log | cut -c1-140; rm -f $2/per_env.pt; }
CFG=$(ls $B/terrain-*.json | head -1)
# The frozen training arguments of the config, without the run name, the wall-clock limit and the checkpoint interval.
ARGS=$(python3 - "$CFG" <<'PYEOF'
import json, shlex, sys
argv = json.load(open(sys.argv[1]))["argv"][4:]
out, skip = [], 0
for i, x in enumerate(argv):
    if skip:
        skip -= 1
        continue
    if x in ("--run-name", "--schedule.max-hours", "--schedule.checkpoint-interval"):
        skip = 1
        continue
    out.append("$W/npz40" if x == "<npz40>" else shlex.quote(x))
print(" ".join(out))
PYEOF
)
ARGS=$(eval echo "$ARGS")
echo "smoke args: $ARGS"

step smoke-init
timeout 900 env -u PYTHONPATH uv run --frozen sm-train $ARGS --run-name init --schedule.total-iterations 1 --init-from $B/init_ckpt.pt > $O/init_train.log 2>&1; echo "init_exit=$?" | tee -a $O/exits.txt
grep -E "^\[INFO\]|^\[WARN\]|Traceback|Error" $O/init_train.log | cut -c1-400 | tail -14

step smoke-train
# 500 iterations, or 6 minutes of wall clock, whichever comes first (the simulation speed with the terrain is not known yet)
timeout 900 env -u PYTHONPATH uv run --frozen sm-train $ARGS --run-name smoke --schedule.total-iterations 500 --schedule.max-hours 0.1 --schedule.checkpoint-interval 250 --init-from $B/init_ckpt.pt > $O/train.log 2>&1; echo "train_exit=$?" | tee -a $O/exits.txt
grep '^it ' $O/train.log | tail -5 | cut -c1-330; grep -E "Traceback|Error|WARN" $O/train.log | cut -c1-300 | tail -5

step smoke-terrain-eval-warm-start
TEV logs/init/ckpt_latest.pt $O/eval/terrain_init ""
TEV logs/init/ckpt_latest.pt $O/eval/terrain_init_heightfields "--heightfields"

step smoke-flat-eval-warm-start
EV logs/init/ckpt_latest.pt $O/eval/win_init "--window-steps 500 --data-label LAFAN40-train-coverage" 600

step smoke-resume
timeout 600 env -u PYTHONPATH uv run --frozen sm-train $ARGS --run-name smoke-resume --schedule.total-iterations 2000 --schedule.max-hours 0.04 --schedule.checkpoint-interval 250 --resume logs/smoke/ckpt_latest.pt > $O/resume.log 2>&1; echo "resume_exit=$?" | tee -a $O/exits.txt
grep -E "resumed|WARN|Traceback|Error" $O/resume.log | cut -c1-300 | tail -5; grep '^it ' $O/resume.log | tail -2 | cut -c1-330

step smoke-terrain-eval-after-training
TEV logs/smoke/ckpt_latest.pt $O/eval/terrain_smoke ""

mkdir -p $O/train_logs/smoke && cp -r logs/smoke/config.json logs/smoke/tb $O/train_logs/smoke/ 2>/dev/null
mkdir -p $O/train_logs/init && cp logs/init/config.json $O/train_logs/init/ 2>/dev/null
step smoke-done
