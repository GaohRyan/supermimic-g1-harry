#!/usr/bin/env bash
# S (night 2026-11-10): does policy weight averaging hold up? No training. LAFAN40 +anchor seed-0 line.
#  A. earlier stages: average of the newest 4 checkpoints of sessions 1, 2, 3 (vs the single final checkpoint of each)
#  B. averaging width at session 4: 2, (4, 7 already scored in R1), 12 checkpoints
#  C. the 7-checkpoint average under robust-v1 (sensor noise + DR + reset noise; action noise; second seed)
#  D. the 7-checkpoint average on BONES40 (clips never trained on)
# Runs inside the generated run_remote.sh (cwd = $W/code). Every step is non-fatal and logs its exit code.
set -uo pipefail
W=/workspace/job; B=$W/bundle; O=$W/out
PY="env -u PYTHONPATH uv run --frozen python"
step(){ echo "=== $(date -u +%FT%TZ) $*"; }
EV(){ MUJOCO_GL=egl timeout ${5:-900} $PY scripts/eval_fullclip_traj.py --ckpt $1 --out $2 --npz-dir $3 $4 > $2.log 2>&1; echo "eval $(basename $2) exit=$?" | tee -a $O/exits.txt; tail -1 $2.log | cut -c1-200; rm -f $2/render_input.npz $2/trajectories.npz; }
L="--data-label LAFAN40-train-coverage"

step swa-robust
EV $B/swa7_s4.pt $O/eval/full_robust_swa7 $W/npz40 "--obs-noise --domain-randomization --reset-noise $L"
EV $B/swa7_s4.pt $O/eval/full_robust_seed1_swa7 $W/npz40 "--obs-noise --domain-randomization --reset-noise --seed 1 $L"
EV $B/swa7_s4.pt $O/eval/full_robust_actnoise_swa7 $W/npz40 "--obs-noise --domain-randomization --reset-noise --action-noise-std 0.1 $L"
EV $B/swa7_s4.pt $O/eval/win_robust_swa7 $W/npz40 "--window-steps 500 --obs-noise --domain-randomization --reset-noise $L" 600

step swa-bones40-holdout
mkdir -p $W/rawbones && tar xzf $B/bones-40csv-2f59b20.tar.gz -C $W/rawbones && timeout 900 $PY scripts/data/csv_to_npz_batch.py --input-dir $W/rawbones/csv --output-dir $W/npzbones --input-fps 120 --output-fps 50 --workers 8 > $O/convert_bones.log 2>&1; echo "convert_bones_exit=$?" | tee -a $O/exits.txt
(cd $W/npzbones && sha256sum *.npz) > $O/npzbones.sha256
EV $B/swa7_s4.pt $O/eval/full_bones40_swa7 $W/npzbones "--data-label BONES40-holdout-unseen-clips"

step swa-earlier-stages
for k in swa4_s3 swa4_s2 swa4_s1; do
  EV $B/$k.pt $O/eval/full_$k $W/npz40 "$L"
  EV $B/$k.pt $O/eval/win_$k $W/npz40 "--window-steps 500 $L" 600
done

step swa-width
for k in swa2_s4 swa12_s34; do
  EV $B/$k.pt $O/eval/full_$k $W/npz40 "$L"
  EV $B/$k.pt $O/eval/win_$k $W/npz40 "--window-steps 500 $L" 600
done
