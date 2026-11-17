"""sim2sim: run the trained tracker closed loop in plain MuJoCo (CPU, C solver) instead of MuJoCo-Warp (PIPELINE §7).

  source tools/offline/env.sh
  .venv-sim/bin/python tools/sim2sim/sim2sim_mujoco_cpu.py validate --contract-dir <dir> --warp-traj <trajectories.npz> --out <json>
  .venv-sim/bin/python tools/sim2sim/sim2sim_mujoco_cpu.py openloop --contract-dir <dir> --warp-traj <trajectories.npz> --out <json>
  .venv-sim/bin/python tools/sim2sim/sim2sim_mujoco_cpu.py closed   --contract-dir <dir> --warp-traj <trajectories.npz> --ckpt <pt> --out-dir <dir>

contract-dir = output of the variant's scripts/export_sim2sim_contract.py (g1_nominal.mjb + contract.json), i.e. the
same compiled model the Warp env uses. Needs a local `mujoco` of the exporting version (.venv-sim).

validate : analytic reward / termination on Warp-logged states vs the logged env values (fixes the frame alignment
           and proves the scoring code here equals the env's).
openloop : replay Warp-logged executed actions from Warp-logged states; state divergence vs horizon.
closed   : policy in the loop from the same initial states and reference frames as the Warp evaluation; same
           termination, same fixed-denominator return. Observations use the analytic path that reproduces mjlab's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
import sm_offline as smo  # noqa: E402
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_relative_obs, flatten_history_term_major  # noqa: E402

MAX_R = 0.1


def load_contract(d: Path):
    c = json.loads((d / "contract.json").read_text())
    assert mujoco.__version__ == c["mujoco_version"], f"local mujoco {mujoco.__version__} != exporter {c['mujoco_version']}"
    m = mujoco.MjModel.from_binary_path(str(d / "g1_nominal.mjb"))
    assert m.nq == 36 and m.nv == 35 and m.nu == 29, (m.nq, m.nv, m.nu)
    assert c["action_joint_names"] == c["robot_joint_names"], "action joint order differs from robot joint order"
    # ctrl index for each action dim: actuator whose joint is that action joint
    act_joint = [j.split("/")[-1] for j in c["actuator_joint_names"]]  # model names carry the entity prefix "robot/"
    ctrl_of = np.array([act_joint.index(j) for j in c["action_joint_names"]])  # actuator order != joint order
    assert sorted(ctrl_of) == list(range(m.nu)), "every actuator must be driven by exactly one action dim"
    assert [j.split("/")[-1] for j in c["model_joint_names"][1:]] == c["robot_joint_names"], "model joint order differs from robot joint order (qpos layout)"
    return m, c, ctrl_of


def ref_at(lib, frame: torch.Tensor) -> ReferenceFrame:
    r, b = lib.ref(frame), lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


def setup(a):
    m, c, ctrl_of = load_contract(a.contract_dir)
    lib, heads, names = smo.build(Path(a.npz_dir), c["default_joint_pos"], c["soft_joint_pos_limits"])
    z = np.load(a.warp_traj)
    t = {k: z[k] for k in z.files}  # NpzFile decompresses the whole array on every access; load once
    wnames = [str(x) for x in t["names"]]
    return m, c, ctrl_of, lib, heads, names, t, wnames


def check_library(lib, t, i, T):
    """Local library must reproduce the logged reference at the logged frame indices."""
    fr = torch.as_tensor(t["frame"][i, : T + 1]).long()
    q = lib.ref_qpos_qvel(fr)[0].numpy()
    return float(np.abs(q - t["ref_qpos"][i, : T + 1]).max())


@torch.no_grad()
def validate(a):
    m, c, ctrl_of, lib, heads, names, t, wnames = setup(a)
    rows = []
    for i, nm in enumerate(wnames):
        T = int(t["executed_steps"][i])
        if T < 20:
            continue
        s = torch.as_tensor(np.concatenate([t["qpos"][i, : T + 1], t["qvel"][i, : T + 1]], -1)).float()
        act = torch.as_tensor(t["action_clean"][i, :T]).float()
        prev = torch.cat([torch.zeros(1, 29), act[:-1]], 0)
        fr = torch.as_tensor(t["frame"][i, : T + 1]).long()
        lib_err = check_library(lib, t, i, T)
        logged = t["reward"][i, :T]
        res = {"clip": nm, "steps": T, "library_ref_maxabs": lib_err}
        for tag, fidx in (("ref_frame_before_step", fr[:T]), ("ref_frame_after_step", fr[1 : T + 1])):
            r = heads.reward(s[1 : T + 1], act, prev, ref_at(lib, fidx))["total"].numpy()
            res[f"reward_maxabs_{tag}"] = float(np.abs(r - logged).max())
            res[f"reward_sum_{tag}"] = float(r.sum())
            term = heads.terminated(s[1 : T + 1], ref_at(lib, fidx)).numpy()
            tn = [str(x) for x in t["term_names"]]
            phys = t["term_flags"][i, :T][:, [tn.index("root_height"), tn.index("root_tilt")]].any(-1)
            res[f"term_mismatch_steps_{tag}"] = int((term != phys).sum())
        res["reward_sum_logged"] = float(logged.sum())
        rows.append(res)
    keys = ("ref_frame_before_step", "ref_frame_after_step")
    summ = {k: {"reward_maxabs_max": max(r[f"reward_maxabs_{k}"] for r in rows), "reward_maxabs_median": float(np.median([r[f"reward_maxabs_{k}"] for r in rows])),
                "term_mismatch_steps": sum(r[f"term_mismatch_steps_{k}"] for r in rows)} for k in keys}
    summ["library_ref_maxabs_max"] = max(r["library_ref_maxabs"] for r in rows)
    summ["best_alignment"] = min(keys, key=lambda k: summ[k]["reward_maxabs_median"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "clips": rows}, indent=1))
    print(json.dumps(summ, indent=1))


def step_ctrl(m, d, c, ctrl_of, action: np.ndarray):
    d.ctrl[ctrl_of] = np.asarray(c["action_offset"]) + np.asarray(c["action_scale"]) * action
    for _ in range(c["decimation"]):
        mujoco.mj_step(m, d)


def set_state(m, d, qpos, qvel, origin):
    d.qpos[:] = qpos
    d.qpos[:3] += origin
    d.qvel[:] = qvel
    d.act[:] = 0 if m.na else d.act
    d.time = 0.0
    mujoco.mj_forward(m, d)


@torch.no_grad()
def openloop(a):
    m, c, ctrl_of, lib, heads, names, t, wnames = setup(a)
    origin = np.asarray(c["env_origin"])
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    H = [1, 2, 4, 8, 16, 32, 50]
    acc = {h: {"root_pos_m": [], "root_rot_deg": [], "joint_rad": [], "joint_vel_rad_s": [], "root_linvel_m_s": []} for h in H}
    n_starts = 0
    for i, nm in enumerate(wnames):
        T = int(t["executed_steps"][i])
        if T < 200:
            continue
        for k in rng.integers(20, T - 60, size=a.starts_per_clip):
            set_state(m, d, t["qpos"][i, k], t["qvel"][i, k], origin)
            if a.warm_prev_ctrl and k > 0:
                d.ctrl[ctrl_of] = np.asarray(c["action_offset"]) + np.asarray(c["action_scale"]) * t["action_executed"][i, k - 1]
            n_starts += 1
            for j in range(1, max(H) + 1):
                step_ctrl(m, d, c, ctrl_of, t["action_executed"][i, k + j - 1])
                if j in acc:
                    qw, vw = t["qpos"][i, k + j], t["qvel"][i, k + j]
                    qc = d.qpos.copy(); qc[:3] -= origin
                    dq = np.zeros(3); mujoco.mju_subQuat(dq, qc[3:7].astype(np.float64), qw[3:7].astype(np.float64))
                    acc[j]["root_pos_m"].append(np.linalg.norm(qc[:3] - qw[:3]))
                    acc[j]["root_rot_deg"].append(np.degrees(np.linalg.norm(dq)))
                    acc[j]["joint_rad"].append(np.abs(qc[7:] - qw[7:]).mean())
                    acc[j]["joint_vel_rad_s"].append(np.abs(d.qvel[6:] - vw[6:]).mean())
                    acc[j]["root_linvel_m_s"].append(np.linalg.norm(d.qvel[:3] - vw[:3]))
    out = {"starts": n_starts, "control_dt": c["control_dt"],
           "horizon_steps": {str(h): {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v))} for k, v in acc[h].items()} for h in H}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for h in H:
        x = out["horizon_steps"][str(h)]
        print(f"h={h:3d} ({h * c['control_dt']:.2f}s) root_pos med {x['root_pos_m']['median'] * 1000:7.2f} mm p90 {x['root_pos_m']['p90'] * 1000:7.2f} | joint med {x['joint_rad']['median']:.4f} rad p90 {x['joint_rad']['p90']:.4f} | root_rot med {x['root_rot_deg']['median']:.3f} deg")


@torch.no_grad()
def closed(a):
    m, c, ctrl_of, lib, heads, names, t, wnames = setup(a)
    pol, cfg, meta = smo.load_policy(Path(a.ckpt))
    anchor = bool(cfg["env"].get("anchor_obs", False))
    anchor_ori_only = cfg["env"].get("anchor_obs_mode", "full") == "ori"  # variant anchor-ori: rot6d only
    official = cfg["env"].get("goal_contract", "cuboi") == "official"  # variant official-contract: 640-D goal, pelvis anchor
    heads_torso = heads  # Cuboi reward definition (torso anchor): the common ruler across variants
    if official:
        from supermimic.world_model.heads import anchor_ori_window_obs

        _, heads, _ = smo.build(Path(a.npz_dir), c["default_joint_pos"], c["soft_joint_pos_limits"], anchor="pelvis")
    rsum_torso = None
    # ---- model-error / sensing perturbations (all default to "none"); recorded in results.json
    pert = {"friction_scale": a.friction_scale, "mass_scale": a.mass_scale, "motor_scale": a.motor_scale, "action_delay_steps": a.action_delay_steps,
            "yaw_bias_deg": a.yaw_bias_deg, "yaw_drift_deg_per_s": a.yaw_drift_deg_per_s}
    # Foot friction: the exported .mjb carries the MJCF default (0.6); the nominal eval protocol sets the foot geoms to 1.0 at
    # runtime (startup event), which is not baked into the export. Until 2026-11-12 this tool ran at 0.6 without saying so.
    import re as _re
    foot = [i for i in range(m.ngeom) if _re.match(c.get("foot_geom_regex", r".*foot\d+_collision"), (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").split("/")[-1])]
    assert len(foot) == 14, len(foot)
    if a.foot_friction is not None:
        m.geom_friction[foot, 0] = a.foot_friction
    m.geom_friction[foot, 0] *= a.friction_scale
    pert["foot_friction_effective"] = float(m.geom_friction[foot[0], 0])
    m.body_mass[:] *= a.mass_scale
    m.body_inertia[:] *= a.mass_scale
    m.actuator_gainprm[:, 0] *= a.motor_scale  # position actuators: kp ...
    m.actuator_biasprm[:, 1] *= a.motor_scale  # ... and its -kp bias term
    m.actuator_forcerange[:] *= a.motor_scale  # torque limit
    delay_buf = []

    def yaw_err_quat(k):  # heading-estimate error (IMU yaw offset + drift), applied only to the orientation the goal is built from
        ang = np.radians(a.yaw_bias_deg + a.yaw_drift_deg_per_s * k * c["control_dt"])
        return torch.tensor([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)], dtype=torch.float32)

    def qmul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)
    Hh, K = cfg["policy"]["history"], cfg["env"]["future_frames"]
    origin = np.asarray(c["env_origin"])
    q0 = torch.as_tensor(c["default_joint_pos"], dtype=torch.float32)
    ai = heads.anchor_index
    sel = [i for i, nm in enumerate(wnames) if (not a.clips or nm in a.clips)]
    n = len(sel)
    planned = np.array([int(t["planned_steps"][i]) for i in sel])
    Tmax = int(planned.max())
    # reference frame index per step, taken from the Warp log where available, else continued (+1, clamped at the clip end)
    frames = np.zeros((n, Tmax + 1), dtype=np.int64)
    for e, i in enumerate(sel):
        ex = int(t["executed_steps"][i])
        f0 = int(t["frame"][i, 0])
        full = np.minimum(f0 + np.arange(Tmax + 1), f0 + planned[e] - 2)  # clip = frames f0-1 .. f0+planned-2; the env clamps at the last one
        frames[e] = full
        assert np.array_equal(full[:ex], t["frame"][i, :ex]), f"frame schedule mismatch for {wnames[i]}"
    datas = [mujoco.MjData(m) for _ in range(n)]
    for e, i in enumerate(sel):
        set_state(m, datas[e], t["qpos"][i, 0], t["qvel"][i, 0], origin)

    def read():
        s = np.stack([np.concatenate([d.qpos, d.qvel]) for d in datas]).astype(np.float32)
        s[:, :3] -= origin
        return torch.as_tensor(s)

    s = read()
    xi = torch.zeros(n, 33)
    prev = torch.zeros(n, 29)
    alive = np.ones(n, bool)
    steps = np.zeros(n, int)
    rsum = np.zeros(n)
    reason = [None] * n
    hist = None
    Q = np.zeros((n, Tmax + 1, 36), np.float32); Q[:, 0] = s[:, :36].numpy()
    A = np.zeros((n, Tmax, 29), np.float32)
    R = np.zeros((n, Tmax), np.float32)
    t0 = time.time()
    for k in range(Tmax):
        fr = torch.as_tensor(frames[:, k])
        obs = analytic_observation(s, prev, xi, q0)
        hist = obs[None].repeat(Hh, 1, 1) if hist is None else torch.cat([hist[1:], obs[None]], 0)
        goal = lib.goal_window(fr, K)
        if official:
            q_est = qmul(yaw_err_quat(k).expand(n, 4), s[:, 3:7]) if (a.yaw_bias_deg or a.yaw_drift_deg_per_s) else s[:, 3:7]
            goal = torch.cat([lib.goal_window_official(fr), anchor_ori_window_obs(q_est, lib.root_quat_window_official(fr))], -1)
        elif anchor:
            b = heads.robot_bodies(s)
            rb = lib.bodies(fr)
            rel = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], rb.pos_w[:, ai], rb.quat_w[:, ai])
            goal = torch.cat([goal, rel[:, 3:] if anchor_ori_only else rel], -1)
        act = pol(flatten_history_term_major(hist), goal)
        an = act.numpy()
        delay_buf.append(an)
        applied = delay_buf[max(0, len(delay_buf) - 1 - a.action_delay_steps)]  # action computed d control steps ago
        for e in range(n):
            if alive[e]:
                step_ctrl(m, datas[e], c, ctrl_of, applied[e])
        s = read()
        ref = ref_at(lib, torch.as_tensor(frames[:, k + (1 if a.ref_after_step else 0)]))
        r = heads.reward(s, act, prev, ref)["total"].numpy()
        term = heads.terminated(s, ref).numpy()
        if official:
            rt = heads_torso.reward(s, act, prev, ref)["total"].numpy()
            rsum_torso = np.zeros(n) if rsum_torso is None else rsum_torso
            rsum_torso += np.where(alive, rt, 0.0)
        hz = (ref.root_pos_w[:, 2] - s[:, 2]).abs().numpy() > 0.25
        for e in range(n):
            if not alive[e]:
                continue
            rsum[e] += r[e]; steps[e] += 1
            A[e, k] = an[e]; R[e, k] = r[e]; Q[e, k + 1] = s[e, :36].numpy()
            if term[e]:
                alive[e] = False; reason[e] = "root_height" if hz[e] else "root_tilt"
            elif steps[e] >= planned[e]:
                alive[e] = False; reason[e] = "clip_end"
        prev = act
        if not alive.any():
            break
        if k % 2000 == 0:
            print(f"step {k}/{Tmax} alive {int(alive.sum())}/{n} {time.time() - t0:.0f}s", flush=True)
    rows = []
    for e, i in enumerate(sel):
        w_ex = int(t["executed_steps"][i]); w_ret = float(t["reward"][i, :w_ex].sum() / (MAX_R * planned[e]))
        tn = [str(x) for x in t["term_names"]]
        wf = [tn[j] for j in range(len(tn)) if t["term_flags"][i, max(w_ex - 1, 0), j]]
        rows.append({"clip": wnames[i], "planned_steps": int(planned[e]), "cpu_executed_steps": int(steps[e]), "cpu_termination": reason[e], "cpu_completed": reason[e] == "clip_end",
                     "cpu_norm_return": float(rsum[e] / (MAX_R * planned[e])), "warp_executed_steps": w_ex, "warp_termination": "+".join(wf) or "?", "warp_completed": bool(w_ex == planned[e] and "clip_end" in wf),
                     "warp_norm_return": w_ret})
    extra = {} if rsum_torso is None else {"cpu_macro_return_torso_anchor_ruler": float(np.mean(rsum_torso / (MAX_R * planned)))}
    summ = {**extra, "clips": n, "cpu_completed": sum(r["cpu_completed"] for r in rows), "warp_completed": sum(r["warp_completed"] for r in rows),
            "cpu_macro_return": float(np.mean([r["cpu_norm_return"] for r in rows])), "warp_macro_return": float(np.mean([r["warp_norm_return"] for r in rows])),
            "both_completed": sum(r["cpu_completed"] and r["warp_completed"] for r in rows), "only_cpu_failed": [r["clip"] for r in rows if r["warp_completed"] and not r["cpu_completed"]],
            "only_warp_failed": [r["clip"] for r in rows if r["cpu_completed"] and not r["warp_completed"]],
            "return_abs_diff_on_both_completed_mean": float(np.mean([abs(r["cpu_norm_return"] - r["warp_norm_return"]) for r in rows if r["cpu_completed"] and r["warp_completed"]] or [np.nan]))}
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "trajectories.npz", names=np.array([wnames[i] for i in sel]), qpos=Q, action=A, reward=R, executed_steps=steps, planned_steps=planned)
    (od / "results.json").write_text(json.dumps({"schema_version": 1, "protocol_id": "cuboi-g1/sim2sim-mujoco-cpu-fullclip-nominal-v1", "simulator": f"mujoco {mujoco.__version__} (C, CPU)", "checkpoint": str(a.ckpt), "checkpoint_meta": meta,
                                               "anchor_obs": anchor, "contract_dir": str(a.contract_dir), "warp_traj": str(a.warp_traj), "ref_after_step": a.ref_after_step, "perturbation": pert, "wall_seconds": time.time() - t0,
                                               "summary": summ, "clips": rows}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "openloop", "closed"):
        p = sub.add_parser(name)
        p.add_argument("--contract-dir", type=Path, required=True)
        p.add_argument("--warp-traj", type=Path, required=True)
        p.add_argument("--npz-dir", default=None)
        if name != "closed":
            p.add_argument("--out", required=True)
        if name == "openloop":
            p.add_argument("--starts-per-clip", type=int, default=20)
            p.add_argument("--warm-prev-ctrl", action="store_true", default=True)
        if name == "closed":
            p.add_argument("--ckpt", required=True)
            p.add_argument("--out-dir", required=True)
            p.add_argument("--clips", nargs="*", default=[])
            p.add_argument("--ref-after-step", action="store_true")
            p.add_argument("--friction-scale", type=float, default=1.0, help="multiplies the foot-geom sliding friction")
            p.add_argument("--foot-friction", type=float, default=None, help="absolute foot friction (nominal protocol = 1.0; exported model default = 0.6)")
            p.add_argument("--mass-scale", type=float, default=1.0)
            p.add_argument("--motor-scale", type=float, default=1.0, help="scales kp and the torque limit of every actuator")
            p.add_argument("--action-delay-steps", type=int, default=0, help="control steps (20 ms each) between computing and applying an action")
            p.add_argument("--yaw-bias-deg", type=float, default=0.0, help="constant heading-estimate error fed to the goal (official-contract policies)")
            p.add_argument("--yaw-drift-deg-per-s", type=float, default=0.0)
    a = ap.parse_args()
    import os

    a.npz_dir = a.npz_dir or os.environ["NPZ40_LOCAL"]
    {"validate": validate, "openloop": openloop, "closed": closed}[a.cmd](a)


if __name__ == "__main__":
    main()
