#!/usr/bin/env python3
"""Build a session bundle (run_remote.sh + inputs + SHA256SUMS) from a frozen training config.

The training argv comes from the frozen config JSON, never from editing another session's script.

  python3 -I infra/runpod/make_bundle.py <session_dir> --config configs/cuboi-g1/<frozen>.json \
      --code-tar <code-<sha7>.tar.gz> --data-tar <lafan1-40csv-*.tar.gz> [--parent-ckpt <ckpt_latest.pt>] \
      [--session-index N] [--max-hours 1.15] [--extra-script path ...] [--render-clips a b] [--no-train --eval-ckpt <pt> ...]

What the generated script guarantees on the Pod:
  * bundle sha256 verification before anything runs;
  * tests (done labels, FK) and, for anchor configs, the mjlab-vs-analytic anchor observation check gate training;
  * 150 s after training starts, config.json is asserted against the frozen config (anchor_obs, goal_dim, clips, seed,
    num_envs) and training is killed on mismatch;
  * window10s eval on every checkpoint, full-clip eval on mid + final. Each eval leaves results.json + geom.json (yaw /
    position / joint error summary); trajectories.npz is kept only for the final checkpoint's full-clip eval from session 4 on.
  * weight average of the newest 4 checkpoints (ckpt_swa4.pt, policy only), scored on both protocols (non-fatal).
  * push-recovery test on the final checkpoint and on the average (non-fatal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def argv_value(argv: list[str], flag: str, n: int = 1):
    if flag not in argv:
        return None
    i = argv.index(flag)
    vals = []
    for a in argv[i + 1 :]:
        if a.startswith("--"):
            break
        vals.append(a)
    return vals if n != 1 else (vals[0] if vals else None)


def copy_clone(src, dst) -> None:
    """Copy into the bundle; on APFS use a clone (no extra disk space, identical bytes), else a plain copy."""
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode == 0:
        return
    shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--code-tar", type=Path, required=True)
    ap.add_argument("--data-tar", type=Path, required=True)
    ap.add_argument("--parent-ckpt", type=Path)
    ap.add_argument("--init-ckpt", type=Path, help="session 1 of a new line starts from another run's checkpoint (sm-train --init-from): weights and optimizer moments, counters at 0")
    ap.add_argument("--session-index", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=1.15)
    ap.add_argument("--train-extra", nargs="*", default=[], help="extra sm-train flags (recorded as a deviation in bundle.json)")
    ap.add_argument("--extra-script", type=Path, nargs="*", default=[])
    ap.add_argument("--render-clips", nargs="*", default=[])
    ap.add_argument("--post-cmd", nargs="*", default=[], help="shell lines run after evaluation (extra evals); cwd = code dir")
    ap.add_argument("--pre-cmd", nargs="*", default=[], help="shell lines run before training (after tests)")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--nonfatal-checks", action="store_true", help="smoke runs: a failing unit test or numeric check is logged and the script goes on (default: stop)")
    ap.add_argument("--keep-full-traj", choices=["auto", "yes", "no"], default="auto", help="keep the final checkpoint's full-clip trajectories.npz (0.34 GB); auto = from session 4 on")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    argv = list(cfg["argv"])
    assert argv[:4] == ["uv", "run", "--frozen", "sm-train"], argv[:4]
    base_run = argv_value(argv, "--run-name")
    run = base_run if a.session_index == 1 else f"{base_run}-s{a.session_index}"
    anchor = "--env.anchor-obs" in argv
    anchor_mode = (argv_value(argv, "--env.anchor-obs-mode") if "--env.anchor-obs-mode" in argv else "full") if anchor else None  # "ori": variant branch opt/anchor-ori
    assert anchor_mode in (None, "full", "ori", "ori_window"), anchor_mode
    official = argv_value(argv, "--env.goal-contract") == "official"
    assert official == (anchor_mode == "ori_window"), "goal-contract official goes with anchor-obs-mode ori_window"
    seed = int(argv_value(argv, "--seed"))
    clips = argv_value(argv, "--data.clips", n=0) or []
    num_envs = int(argv_value(argv, "--env.num-envs"))
    assert bool(cfg.get("paper_contract", {}).get("anchor_obs", False)) == anchor, "config paper_contract.anchor_obs disagrees with argv"
    if a.session_index > 1:
        assert a.parent_ckpt, "warm-restart sessions need --parent-ckpt"
    terrain = "--env.terrain-task" in argv  # terrain task: extra tests, a pre-train numeric check and the per-tile terrain evaluation
    init_ckpt = a.init_ckpt if not a.parent_ckpt else None  # only the first session of a line is initialised from another run

    # training argv: frozen argv with run name, npz dir, max-hours override, optional resume
    t = argv[3:]
    t[t.index("--run-name") + 1] = "$RUN"
    t[t.index("--data.npz-dir") + 1] = "$W/npz40"
    t[t.index("--schedule.max-hours") + 1] = str(a.max_hours)
    t += list(a.train_extra)
    if a.parent_ckpt:
        t += ["--resume", "$B/parent_ckpt.pt"]
    elif init_ckpt:
        t += ["--init-from", "$B/init_ckpt.pt"]
    train_cmd = " ".join(x if x.startswith("$") else shlex.quote(x) for x in t)
    init_argv = [x for x in argv[3:]]
    init_argv[init_argv.index("--run-name") + 1] = "init"
    init_argv[init_argv.index("--data.npz-dir") + 1] = "$W/npz40"
    i = init_argv.index("--schedule.max-hours")
    del init_argv[i : i + 2]
    init_argv += ["--schedule.total-iterations", "1"]
    if init_ckpt:
        init_argv += ["--init-from", "$B/init_ckpt.pt"]  # logs/init then holds the warm start itself: its evaluation is the "before" score
    init_cmd = " ".join(x if x.startswith("$") else shlex.quote(x) for x in init_argv)
    clip_flag = ("--clips " + " ".join(clips) + " ") if clips else ""
    label = "4clip-train-coverage" if clips else "LAFAN40-train-coverage"

    sd = a.session_dir
    b = sd / "bundle"
    b.mkdir(parents=True, exist_ok=True)
    files = {}
    for src, name in ((a.code_tar, "code.tar.gz"), (a.data_tar, "data.tar.gz"), (ROOT / "infra/runpod/system_dependencies.sh", "system_dependencies.sh"),
                      (ROOT / "infra/runpod/budget_watchdog.py", "budget_watchdog.py"), (ROOT / "infra/runpod/pod_jobs/swa_policy.py", "swa_policy.py"), (ROOT / "infra/runpod/pod_jobs/traj_geom.py", "traj_geom.py"), (a.config, a.config.name)):
        copy_clone(src, b / name)
    if a.parent_ckpt:
        copy_clone(a.parent_ckpt, b / "parent_ckpt.pt")
    if init_ckpt:
        copy_clone(init_ckpt, b / "init_ckpt.pt")
    if terrain:
        copy_clone(ROOT / "infra/runpod/pod_checks/check_terrain.py", b / "check_terrain.py")
    for s in a.extra_script:
        copy_clone(s, b / s.name)

    L: list[str] = []
    w = L.append
    w("#!/usr/bin/env bash")
    w(f"# Generated by infra/runpod/make_bundle.py from {a.config.name} (session {a.session_index}). Do not edit by hand.")
    w("set -uo pipefail")
    w("W=/workspace/job; B=$W/bundle; O=$W/out; mkdir -p $O/eval; exec > >(tee -a $O/workload.log) 2>&1")
    w(f"RUN={shlex.quote(run)}")
    w('step(){ echo "=== $(date -u +%FT%TZ) $*"; }')
    w('fail(){ echo "FATAL: $*"; touch $O/CORE_DONE $O/DONE; exit ${2:-1}; }')
    w("step verify-bundle; (cd $B && sha256sum -c SHA256SUMS) || exit 10")
    w("step extract; cd $W && mkdir -p code raw40 && tar xzf $B/code.tar.gz -C $W && tar xzf $B/data.tar.gz -C raw40 || exit 11")
    w("step system-deps; bash $B/system_dependencies.sh >/dev/null 2>&1; bash $B/system_dependencies.sh | tail -2 || exit 12")
    w("command -v ffmpeg >/dev/null || (apt-get update >/dev/null && apt-get install -y --no-install-recommends ffmpeg >/dev/null)")
    w("step host-info; { grep -m1 'model name' /proc/cpuinfo; nproc; free -g | sed -n 2p; } | tee $O/host-info.txt")
    w("python3 - <<'PY' | tee -a $O/host-info.txt\nimport time\nt=time.perf_counter(); s=0\nfor i in range(20_000_000): s+=i*i\nprint('py_bench_s %.2f' % (time.perf_counter()-t))\nPY")
    w("step uv-sync; cd $W/code && timeout 1200 uv sync --frozen --group dev --python 3.12 || exit 13")
    w("PY=\"env -u PYTHONPATH uv run --frozen python\"")
    w("$PY -c \"import torch,mujoco,mjlab;print('torch',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),'mujoco',mujoco.__version__)\" | tee $O/runtime-versions.txt")
    w("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a $O/runtime-versions.txt; uv pip freeze > $O/pip-freeze.txt 2>/dev/null")
    w("step convert-40; timeout 900 $PY scripts/data/csv_to_npz_batch.py --input-dir $W/raw40/lafan1_g1 --output-dir $W/npz40 --input-fps 30 --output-fps 50 --workers 8 || exit 14")
    w("(cd $W/npz40 && sha256sum *.npz) > $O/npz40.sha256")
    w(": > $O/exits.txt")
    tests = "tests/test_done_labels.py tests/test_g1_fk.py" + (" tests/test_heightmap.py tests/test_terrain_lifts.py tests/test_terrain_curriculum.py tests/test_sampler_clip_end_norm.py tests/test_optim.py" if terrain else "")
    stop = (lambda msg, code: f"echo 'CHECK FAILED, continuing (smoke): {msg}'") if a.nonfatal_checks else (lambda msg, code: f"fail '{msg}' {code}")
    w(f"step tests; $PY -m pytest -q -p no:cacheprovider {tests} 2>&1 | tail -1 | tee $O/tests.txt; grep -q ' passed' $O/tests.txt && ! grep -q 'failed' $O/tests.txt || " + stop("unit tests failed", 15))
    if anchor:
        w("step check-anchor-obs")
        w("MUJOCO_GL=egl timeout 600 $PY $B/check_anchor_obs.py --npz-dir $W/npz40 --out $O/check_anchor_obs.json --steps 60 --num-envs 64" + (f" --anchor-obs-mode {anchor_mode}" if anchor_mode in ("ori", "ori_window") else "") + " > $O/check_anchor_obs.log 2>&1; rc=$?; echo \"check_anchor_obs_exit=$rc\" | tee -a $O/exits.txt; tail -1 $O/check_anchor_obs.log | cut -c1-300")
        w("[ $rc -eq 0 ] || " + stop("anchor observation mismatch between mjlab and the analytic (imagination) path", 20))
    if terrain:
        # Reward / termination of the simulator on the lifted references vs the analytic heads, height map vs ray casts, reset placement.
        w("step check-terrain")
        w("MUJOCO_GL=egl timeout 900 $PY $B/check_terrain.py --npz-dir $W/npz40 --out $O/check_terrain.json > $O/check_terrain.log 2>&1; rc=$?; echo \"check_terrain_exit=$rc\" | tee -a $O/exits.txt; tail -1 $O/check_terrain.log | cut -c1-400")
        w("echo \"terrain check: nconmax / njmax overflow messages in the log: $(grep -ci overflow $O/check_terrain.log)\"")
        w("[ $rc -eq 0 ] || " + stop("terrain check: simulator-side reward / termination / height map disagree with the analytic (imagination) path", 22))
    for c in a.pre_cmd:
        w(c)
    w("(nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader -l 5 > $O/gpu-samples.csv 2>&1 &)")
    if not a.no_train:
        if not a.parent_ckpt:
            w("step initial-ckpt")
            w(f"timeout 900 env -u PYTHONPATH uv run --frozen {init_cmd} > $O/init_train.log 2>&1; echo \"init_exit=$?\" | tee -a $O/exits.txt")
        w("step train")
        w(f"export SM_WANDB_MODE=offline SM_RUN_ID={shlex.quote(sd.name)} SM_WANDB_TAGS=cuboi-g1")
        w(f"timeout {int(a.max_hours * 3600) + 560} env -u PYTHONPATH uv run --frozen {train_cmd} > $O/train.log 2>&1 &")
        w("TPID=$!; sleep 150; step assert-config")
        exp = {"anchor_obs": anchor, "goal_dim": 640 if official else 680 + ((6 if anchor_mode == "ori" else 9) if anchor else 0), "seed": seed, "num_envs": num_envs, "clips": clips or None}
        if anchor_mode in ("ori", "ori_window"):
            exp["anchor_obs_mode"] = anchor_mode
        if "--env.sim2real-dr" in argv:  # optimisation-line switches must be on in the run's config and the context must be 38-D
            exp["sim2real_dr"] = True
            exp["action_delay_max"] = int(argv_value(argv, "--env.action-delay-max") or 0)
            exp["xi_dim"] = 38
        if "--env.sampler-clip-end-norm" in argv:
            exp["sampler_clip_end_norm"] = True
        if "--wm-trainer.hard-ratio" in argv:
            exp["wm_hard_ratio"] = float(argv_value(argv, "--wm-trainer.hard-ratio"))
        if "--wm-trainer.batches-per-update" in argv:
            exp["wm_batches_per_update"] = int(argv_value(argv, "--wm-trainer.batches-per-update"))
        if "--wm-trainer.hidden-optimizer" in argv or "--policy-trainer.hidden-optimizer" in argv:
            exp["hidden_optimizer"] = [argv_value(argv, "--wm-trainer.hidden-optimizer") or "adamw", argv_value(argv, "--policy-trainer.hidden-optimizer") or "adamw"]
        if terrain:  # the scan must reach the world model (187) and the actor in the mode the config names
            exp["terrain"] = [True, 187, argv_value(argv, "--policy.scan-mode") or "none", float(argv_value(argv, "--env.foot-clearance-weight") or 30.0), float(argv_value(argv, "--env.progress-weight") or 1.0)]
        w("python3 - <<PY\nimport json,sys\nc=json.load(open(\"logs/$RUN/config.json\"))\nexp=json.loads('''" + json.dumps(exp) + "''')\n"
          "got={'anchor_obs':c['env']['anchor_obs'],'goal_dim':c['policy']['goal_dim'],'seed':c['seed'],'num_envs':c['env']['num_envs'],'clips':c['data']['clips']}\n"
          "if 'anchor_obs_mode' in exp: got['anchor_obs_mode']=c['env'].get('anchor_obs_mode')\n"
          "if 'sim2real_dr' in exp: got.update(sim2real_dr=c['env'].get('sim2real_dr'), action_delay_max=c['env'].get('action_delay_max'), xi_dim=len(c['world_model']['state_spec']['xi_scale']))\n"
          "if 'sampler_clip_end_norm' in exp: got['sampler_clip_end_norm']=c['env'].get('sampler_clip_end_norm')\n"
          "if 'wm_hard_ratio' in exp: got['wm_hard_ratio']=c['wm_trainer'].get('hard_ratio')\n"
          "if 'wm_batches_per_update' in exp: got['wm_batches_per_update']=c['wm_trainer'].get('batches_per_update')\n"
          "if 'hidden_optimizer' in exp: got['hidden_optimizer']=[c['wm_trainer'].get('hidden_optimizer'), c['policy_trainer'].get('hidden_optimizer')]\n"
          "if 'terrain' in exp: got['terrain']=[c['env'].get('terrain_task'), c['world_model'].get('scan_dim'), c['policy'].get('scan_mode'), c['env'].get('foot_clearance_weight'), c['env'].get('progress_weight')]\n"
          "ok=got==exp and c['policy_trainer']['anchor_obs']==exp['anchor_obs']\nprint('config', got, 'OK' if ok else 'MISMATCH expected %s' % exp)\nsys.exit(0 if ok else 21)\nPY")
        w("rc=$?; echo \"config_assert_exit=$rc\" | tee -a $O/exits.txt")
        w("[ $rc -eq 0 ] || { kill $TPID; sleep 5; fail 'training config does not match the frozen config' 21; }")
        w("wait $TPID; echo \"train_exit=$?\" | tee -a $O/exits.txt")
        w("unset SM_WANDB_MODE; grep '^it ' $O/train.log | tail -1 | cut -c1-200; ls logs/$RUN")
        w("EV(){ MUJOCO_GL=egl timeout ${4:-900} $PY scripts/eval_fullclip_traj.py --ckpt $1 --out $2 --npz-dir $W/npz40 $3 > $2.log 2>&1; echo \"eval $(basename $2) exit=$?\"; tail -1 $2.log | cut -c1-200; }")
        keep_full = a.keep_full_traj == "yes" or (a.keep_full_traj == "auto" and a.session_index >= 4)
        # Yaw / position / joint error summary of a trajectories.npz, so the big file itself need not be brought back.
        w("GEOM(){ [ -f $1/trajectories.npz ] && $PY $B/traj_geom.py $1/trajectories.npz $1/geom.json > /dev/null 2>&1; true; }")
        w("step eval-windows")
        pre = "" if a.parent_ckpt else "logs/init/ckpt_latest.pt "
        w(f"for c in {pre}logs/$RUN/ckpt_0*.pt logs/$RUN/ckpt_latest.pt; do [ -f \"$c\" ] || continue; n=$(echo $c | tr / _ | sed 's/.pt$//'); "
          f"EV $c $O/eval/win_$n \"{clip_flag}--window-steps 500 --data-label {label}\" 600; "
          "GEOM $O/eval/win_$n; rm -f $O/eval/win_$n/trajectories.npz $O/eval/win_$n/render_input.npz; done")
        w("step eval-fullclip")
        w("MID=$(ls logs/$RUN/ckpt_0*.pt 2>/dev/null | sort | awk '{a[NR]=$0} END{print a[int((NR+1)/2)]}')")
        w(f"for c in $MID logs/$RUN/ckpt_latest.pt; do [ -f \"$c\" ] || continue; n=$(echo $c | tr / _ | sed 's/.pt$//'); "
          f"EV $c $O/eval/full_$n \"{clip_flag}--data-label {label}\" 900; "
          "GEOM $O/eval/full_$n; if [ \"$c\" = \"logs/$RUN/ckpt_latest.pt\" ]; then " + ("true" if keep_full else "rm -f $O/eval/full_$n/trajectories.npz") + "; "
          "else rm -f $O/eval/full_$n/trajectories.npz $O/eval/full_$n/render_input.npz; fi; done")
        if terrain:
            # Per-tile windows on the training terrain (scripts/eval_terrain.py). The warm start itself is scored in session 1
            # (also with the height-field tiles added, to compare their contact with the flat box tile in this simulator);
            # the newest checkpoint also with the actor shown flat ground and with a 10 % height mismatch.
            w("TEV(){ MUJOCO_GL=egl timeout ${4:-900} $PY scripts/eval_terrain.py --ckpt $1 --out $2 --npz-dir $W/npz40 $3 > $2.log 2>&1; echo \"terrain eval $(basename $2) exit=$?\"; grep terrain_summary $2.log | cut -c1-200; }")
            w("step eval-terrain")
            if not a.parent_ckpt:
                w("TEV logs/init/ckpt_latest.pt $O/eval/terrain_init \"\"; TEV logs/init/ckpt_latest.pt $O/eval/terrain_init_heightfields \"--heightfields\"")
            w("TEV logs/$RUN/ckpt_latest.pt $O/eval/terrain_latest \"\"; TEV logs/$RUN/ckpt_latest.pt $O/eval/terrain_latest_flatscan \"--scan flat\"")
            w("TEV logs/$RUN/ckpt_latest.pt $O/eval/terrain_latest_relief090 \"--scan-relief 0.9\"; TEV logs/$RUN/ckpt_latest.pt $O/eval/terrain_latest_relief110 \"--scan-relief 1.1\"")
            w("rm -f $O/eval/terrain_*/per_env.pt")
        if a.render_clips:
            w("step render; F=$O/eval/full_logs_${RUN}_ckpt_latest")
            w("MUJOCO_GL=egl timeout 600 $PY scripts/media/render_rollout.py --traj $F/render_input.npz --out-dir $O/mp4 --fps 25 --width 960 --height 540 --clips " + " ".join(a.render_clips) + "; echo \"render_exit=$?\"")
        w("rm -f $O/eval/*/render_input.npz")
        w("mkdir -p $O/train_logs/$RUN && cp -r logs/$RUN/ckpt_latest.pt logs/$RUN/config.json logs/$RUN/tb $O/train_logs/$RUN/ && (cd logs/$RUN && sha256sum ckpt_*.pt) > $O/ckpt_sha256_pod.txt")
        # Weight average of this session's newest 4 checkpoints, scored on both protocols. Non-fatal: never blocks DONE.
        w("step swa; S=$O/train_logs/$RUN/ckpt_swa4.pt")
        w("timeout 300 $PY $B/swa_policy.py --out $S --last 4 logs/$RUN/ckpt_0*.pt logs/$RUN/ckpt_latest.pt > $O/swa.log 2>&1; echo \"swa_exit=$?\" | tee -a $O/exits.txt; tail -1 $O/swa.log | cut -c1-200")
        w(f"if [ -f $S ]; then EV $S $O/eval/win_swa4 \"{clip_flag}--window-steps 500 --data-label {label}\" 600; GEOM $O/eval/win_swa4; EV $S $O/eval/full_swa4 \"{clip_flag}--data-label {label}\" 900; GEOM $O/eval/full_swa4; fi")
        if terrain:
            w("if [ -f $S ]; then TEV $S $O/eval/terrain_swa4 \"\"; rm -f $O/eval/terrain_swa4/per_env.pt; fi")
        w("rm -f $O/eval/win_swa4/trajectories.npz $O/eval/full_swa4/trajectories.npz $O/eval/*/render_input.npz")
        # Push recovery (paper section 4.3; upstream eval_push.py: T-pose, pelvis velocity kick at 1 s). Non-fatal.
        w("step push; for c in logs/$RUN/ckpt_latest.pt $S; do [ -f $c ] || continue; n=$(basename $c .pt); "
          "MUJOCO_GL=egl timeout 400 $PY scripts/eval_push.py --ckpt $c > $O/push_$n.log 2>&1; echo \"push_${n}_exit=$?\" | tee -a $O/exits.txt; grep '^push' $O/push_$n.log | cut -c1-160; done")
    for c in a.post_cmd:
        w(c)
    w("touch $O/CORE_DONE; pkill -f 'nvidia-smi --query-gpu=timestamp'; step done; touch $O/DONE")
    (b / "run_remote.sh").write_text("\n".join(L) + "\n")

    names = sorted(p.name for p in b.iterdir() if p.name != "SHA256SUMS")
    (b / "SHA256SUMS").write_text("".join(f"{sha256(b / n)}  {n}\n" for n in names))
    meta = {"schema_version": 1, "session": sd.name, "config_id": cfg["config_id"], "config_sha256": sha256(a.config), "run_name": run, "session_index": a.session_index,
            "anchor_obs": anchor, "anchor_obs_mode": anchor_mode, "seed": seed, "clips": clips, "num_envs": num_envs, "max_hours": a.max_hours, "train_extra": a.train_extra,
            "deviation_from_frozen_argv": ([f"max_hours {a.max_hours} (frozen {argv_value(argv, '--schedule.max-hours')})"] if str(a.max_hours) != argv_value(argv, "--schedule.max-hours") else []) + [f"extra flag {x}" for x in a.train_extra],
            "parent_ckpt_sha256": sha256(a.parent_ckpt) if a.parent_ckpt else None, "init_ckpt_sha256": sha256(init_ckpt) if init_ckpt else None, "code_tar_sha256": sha256(a.code_tar), "data_tar_sha256": sha256(a.data_tar),
            "train_cmd": train_cmd, "files": {n: sha256(b / n) for n in names}}
    (sd / "bundle.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("session", "run_name", "anchor_obs", "seed", "parent_ckpt_sha256", "deviation_from_frozen_argv")}))


if __name__ == "__main__":
    main()
