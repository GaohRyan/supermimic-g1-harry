#!/usr/bin/env bash
# R1 (night 2026-11-10): no-training evaluation suite for the LAFAN40 +anchor seed-0 final checkpoint.
# Runs inside the generated run_remote.sh (cwd = $W/code). Every step is non-fatal and logs its exit code.
set -uo pipefail
W=/workspace/job; B=$W/bundle; O=$W/out
PY="env -u PYTHONPATH uv run --frozen python"
step(){ echo "=== $(date -u +%FT%TZ) $*"; }
EV(){ # EV <ckpt> <outdir> <npz-dir> <extra args> [timeout]
  MUJOCO_GL=egl timeout ${5:-900} $PY scripts/eval_fullclip_traj.py --ckpt $1 --out $2 --npz-dir $3 $4 > $2.log 2>&1; echo "eval $(basename $2) exit=$?" | tee -a $O/exits.txt; tail -1 $2.log | cut -c1-200; }
slim(){ rm -f $1/render_input.npz; [ "${2:-}" = keep ] || rm -f $1/trajectories.npz; }

step export-sim2sim-contract
MUJOCO_GL=egl timeout 300 $PY scripts/export_sim2sim_contract.py --npz-dir $W/npz40 --out $O/sim2sim > $O/export.log 2>&1; echo "export_exit=$?" | tee -a $O/exits.txt; tail -1 $O/export.log | cut -c1-200

step eval-robust-v1   # training-distribution perturbations on the 40 training clips
EV $B/slim_final.pt $O/eval/full_robust_final $W/npz40 "--obs-noise --domain-randomization --reset-noise --data-label LAFAN40-train-coverage" 900; slim $O/eval/full_robust_final keep
EV $B/slim_final.pt $O/eval/win_robust_final $W/npz40 "--window-steps 500 --obs-noise --domain-randomization --reset-noise --data-label LAFAN40-train-coverage" 600; slim $O/eval/win_robust_final
EV $B/slim_final.pt $O/eval/full_robust_actnoise_final $W/npz40 "--obs-noise --domain-randomization --reset-noise --action-noise-std 0.1 --data-label LAFAN40-train-coverage" 900; slim $O/eval/full_robust_actnoise_final
EV $B/slim_final.pt $O/eval/full_robust_seed1_final $W/npz40 "--obs-noise --domain-randomization --reset-noise --seed 1 --data-label LAFAN40-train-coverage" 900; slim $O/eval/full_robust_seed1_final

step eval-bones40-holdout   # clips never seen in training (policy trained on LAFAN40 only)
mkdir -p $W/rawbones && tar xzf $B/bones-40csv-2f59b20.tar.gz -C $W/rawbones && timeout 900 $PY scripts/data/csv_to_npz_batch.py --input-dir $W/rawbones/csv --output-dir $W/npzbones --input-fps 120 --output-fps 50 --workers 8 > $O/convert_bones.log 2>&1; echo "convert_bones_exit=$?" | tee -a $O/exits.txt
(cd $W/npzbones && sha256sum *.npz) > $O/npzbones.sha256
EV $B/slim_final.pt $O/eval/full_bones40_final $W/npzbones "--data-label BONES40-holdout-unseen-clips" 900; slim $O/eval/full_bones40_final keep

step eval-checkpoint-variance   # full-clip on neighbouring checkpoints and on weight averages of them
for k in 38000 40000 42000 swa4 swa7; do EV $B/slim_$k.pt $O/eval/full_s4_$k $W/npz40 "--data-label LAFAN40-train-coverage" 900; slim $O/eval/full_s4_$k; done
EV $B/slim_swa4.pt $O/eval/win_s4_swa4 $W/npz40 "--window-steps 500 --data-label LAFAN40-train-coverage" 600; slim $O/eval/win_s4_swa4
EV $B/slim_swa7.pt $O/eval/win_s4_swa7 $W/npz40 "--window-steps 500 --data-label LAFAN40-train-coverage" 600; slim $O/eval/win_s4_swa7

step speed-test   # same checkpoint, same host: torch.compile default vs reduce-overhead (CUDA graphs)
T="--seed 0 --data.npz-dir $W/npz40 --env.anchor-obs --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 0.17 --schedule.checkpoint-interval 100000 --resume $B/parent_ckpt.pt"
timeout 780 env -u PYTHONPATH uv run --frozen sm-train --run-name speed_default $T > $O/speed_default.log 2>&1; echo "speed_default_exit=$?" | tee -a $O/exits.txt; grep '^it ' $O/speed_default.log | tail -3 | cut -c1-200
timeout 780 env -u PYTHONPATH uv run --frozen sm-train --run-name speed_reduce_overhead $T --policy-trainer.compile reduce-overhead --wm-trainer.compile reduce-overhead > $O/speed_reduce_overhead.log 2>&1; echo "speed_reduce_overhead_exit=$?" | tee -a $O/exits.txt; grep '^it ' $O/speed_reduce_overhead.log | tail -3 | cut -c1-200
