"""Where is the learned world model wrong? CPU-only, on logged closed-loop full-clip trajectories.

python3 tools/offline/wm_error_map.py --ckpt <ckpt.pt> --traj <trajectories.npz> --out <json> [--stride 10]

Open-loop k-step normalised squared error (k = 1, 8, 32; 32 = imagination horizon) from start states every `stride`
steps of every clip. Reported per clip category, per reference root height band (standing / low / on the ground),
per error component (root pos, root rot, root vel, root ang vel, joint pos, joint vel) and against the per-step reward.
PYTHONPATH as for failure_diagnosis.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

import failure_diagnosis as fd

COMP = {"root_pos": slice(0, 3), "root_rot": slice(3, 6), "root_vel": slice(6, 9), "root_angvel": slice(9, 12), "joint_pos": slice(12, 41), "joint_vel": slice(41, 70)}
KS = (1, 8, 32)


@torch.no_grad()
def rollout_err(wm, s, a, starts, kmax=32):
    xi = torch.zeros(len(starts), len(wm.spec.xi_scale))
    xi[:, 0] = 1.0
    if xi.shape[1] > 33:
        xi[:, 33:36] = 1.0  # nominal armature / kp / kd scale of the sim2real-dr context
    cur = s[starts]
    tot, comp = {}, {}
    for j in range(kmax):
        cur = wm.step(cur, a[starts + j], xi)
        if j + 1 in KS:
            e = wm.spec.error_vector(cur, s[starts + j + 1])
            tot[j + 1] = (e * e).mean(-1).numpy()
            comp[j + 1] = {c: (e[:, sl] ** 2).mean(-1).numpy() for c, sl in COMP.items()}
    return tot, comp


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(8)
    fk = fd.G1Kinematics(fd.load_g1_model(), device="cpu")
    heads = fd.AnalyticHeads(fk, fd.TRACKED, "torso_link", "pelvis", torch.zeros(29, 2), torch.zeros(29))
    wm, meta = fd.load_wm(a.ckpt, heads)
    t = {k: v for k, v in np.load(a.traj).items()}
    rec = {k: [] for k in ("cat", "clip", "refz", "rew", "e1", "e8", "e32", "speed")}
    comp_all = {k: {c: [] for c in COMP} for k in KS}
    for i, nm in enumerate(t["names"]):
        T = int(t["executed_steps"][i])
        if T < 80:
            continue
        s = torch.as_tensor(np.concatenate([t["qpos"][i, : T + 1], t["qvel"][i, : T + 1]], -1).astype(np.float32))
        act = torch.as_tensor(t["action_executed"][i, :T].astype(np.float32))
        starts = np.arange(5, T - 33, a.stride)
        tot, comp = rollout_err(wm, s, act, starts)
        rz = t["ref_qpos"][i, : T + 1, 2]
        rw = t["reward"][i, :T]
        jv = np.abs(t["qvel"][i, : T + 1, 6:]).mean(-1)
        n = len(starts)
        rec["cat"] += [re.sub(r"\d*_subject\d+$", "", str(nm))] * n
        rec["clip"] += [str(nm)] * n
        rec["refz"] += [float(rz[st : st + 32].min()) for st in starts]
        rec["rew"] += [float(rw[st : st + 32].mean()) for st in starts]
        rec["speed"] += [float(jv[st : st + 32].mean()) for st in starts]
        for k in KS:
            rec[f"e{k}"] += tot[k].tolist()
            for c in COMP:
                comp_all[k][c] += comp[k][c].tolist()
        print(f"{str(nm):28s} n={n:5d} nse1 {tot[1].mean():.4f} nse8 {tot[8].mean():.3f} nse32 {tot[32].mean():.3f} (median {np.median(tot[32]):.3f})", flush=True)
    R = {k: np.asarray(v) for k, v in rec.items()}
    def stat(m):
        return {"n": int(m.sum()), **{f"nse{k}_mean": float(R[f"e{k}"][m].mean()) for k in KS}, **{f"nse{k}_median": float(np.median(R[f"e{k}"][m])) for k in KS}, "reward_mean": float(R["rew"][m].mean())}
    out = {"checkpoint": str(a.ckpt), "meta": meta, "stride": a.stride, "all": stat(np.ones(len(R["e1"]), bool)),
           "by_category": {c: stat(R["cat"] == c) for c in sorted(set(R["cat"]))},
           "by_ref_root_height": {"standing (>0.6 m)": stat(R["refz"] > 0.6), "low (0.35-0.6 m)": stat((R["refz"] <= 0.6) & (R["refz"] > 0.35)), "on the ground (<0.35 m)": stat(R["refz"] <= 0.35)},
           "by_joint_speed_quartile": {}, "components_share": {}, "corr_log_nse32_reward": float(np.corrcoef(np.log(R["e32"] + 1e-9), R["rew"])[0, 1])}
    qs = np.quantile(R["speed"], [0.25, 0.5, 0.75])
    for j, nm in enumerate(("q1 slow", "q2", "q3", "q4 fast")):
        m = (R["speed"] > (qs[j - 1] if j else -1)) & (R["speed"] <= (qs[j] if j < 3 else 1e9))
        out["by_joint_speed_quartile"][nm] = stat(m)
    for k in KS:
        tot = {c: float(np.mean(comp_all[k][c]) * (COMP[c].stop - COMP[c].start)) for c in COMP}
        z = sum(tot.values())
        out["components_share"][f"k{k}"] = {c: round(v / z, 3) for c, v in tot.items()}
    grd = R["refz"] <= 0.35
    for k in KS:
        tot = {c: float(np.asarray(comp_all[k][c])[grd].mean() * (COMP[c].stop - COMP[c].start)) for c in COMP}
        z = sum(tot.values())
        out["components_share"][f"k{k}_on_ground"] = {c: round(v / z, 3) for c, v in tot.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("checkpoint",)}, indent=1))


if __name__ == "__main__":
    main()
