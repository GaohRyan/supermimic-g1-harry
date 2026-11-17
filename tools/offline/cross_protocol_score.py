"""Score closed-loop trajectories from different systems under ONE definition (Cuboi / paper Table A.2 tracking terms).

  source tools/offline/env.sh
  python3 tools/offline/cross_protocol_score.py --out <json> --cuboi tag=<trajectories.npz> ... --instanttrack tag=<traj_final.npz> ...

For every clip: the six tracking terms (body pos / ori / lin vel / ang vel, root pos / ori; weights 1,1,1,1,0.5,0.5,
max 0.1 per 50 Hz step) evaluated on the logged robot states against the SAME reference (our LAFAN40 conversion),
normalised by 0.1 x planned steps (fixed denominator), plus the paper termination (root height 0.25 m, tilt 0.8)
re-evaluated on the states. Regularisers (action rate, joint limit) are excluded because action parametrisations differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sm_offline as smo  # noqa: E402
from supermimic.envs.mdp.rewards import tracking_reward_terms  # noqa: E402
from supermimic.kinematics.rotations import quat_apply_inverse  # noqa: E402
from supermimic.world_model.heads import ReferenceFrame  # noqa: E402

TRACK = ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")
MAX_R = 0.1


def rotm_to_quat(R: torch.Tensor) -> torch.Tensor:
    m = R
    w = torch.sqrt(torch.clamp(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], min=1e-12)) / 2
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], -1)


@torch.no_grad()
def score(heads, lib, s: torch.Tensor, frames: torch.Tensor, planned: int, chunk: int = 4000) -> dict:
    """s (T+1,71) robot states; frames (T+1,) library frame index of the same instant. Scores steps 1..T."""
    T = s.shape[0] - 1
    tot = {k: 0.0 for k in TRACK}
    first_term = None
    for a in range(1, T + 1, chunk):
        b = min(a + chunk, T + 1)
        fr = frames[a:b]
        r, bd = lib.ref(fr), lib.bodies(fr)
        ref = ReferenceFrame(bd.pos_w, bd.quat_w, bd.lin_vel_w, bd.ang_vel_w, r.root_pos_w, r.root_quat_w)
        z = torch.zeros(b - a, 29)
        terms = heads.reward(s[a:b], z, z, ref)
        for k in TRACK:
            tot[k] += float(terms[k].sum())
        term = heads.terminated(s[a:b], ref)
        if first_term is None and bool(term.any()):
            first_term = a + int(term.nonzero()[0])
    denom = MAX_R * planned
    out = {f"{k}_norm": tot[k] / denom for k in TRACK}
    out["tracking_return_norm"] = sum(tot.values()) / denom
    out["first_paper_termination_step"] = first_term
    out["executed_steps"] = T
    out["planned_steps"] = planned
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuboi", nargs="*", default=[])
    ap.add_argument("--instanttrack", nargs="*", default=[])
    ap.add_argument("--npz-dir", default=os.environ.get("NPZ40_LOCAL"))
    a = ap.parse_args()
    lib, heads, names = smo.build(Path(a.npz_dir), np.zeros(29))
    # sanity: the analytic reward dict must expose the six tracking terms
    res = {"definition": "six tracking terms of paper Table A.2 on logged states vs our LAFAN40 reference; fixed denominator 0.1 x planned steps; regularisers excluded", "systems": {}}
    for spec in a.cuboi:
        tag, path = spec.split("=", 1)
        t = np.load(path)
        rows = []
        for i, nm in enumerate([str(x) for x in t["names"]]):
            T = int(t["executed_steps"][i])
            s = torch.as_tensor(np.concatenate([t["qpos"][i, : T + 1], t["qvel"][i, : T + 1]], -1)).float()
            # Cuboi log: frame[k] is the target of step k (= frame of state k+1); state k is at frame[k]-1
            fr = torch.as_tensor(t["frame"][i, : T + 1]).long() - 1
            fr[1:] = torch.as_tensor(t["frame"][i, :T]).long()
            rows.append({"clip": nm, **score(heads, lib, s, fr, int(t["planned_steps"][i]))})
        res["systems"][tag] = {"source": path, "clips": rows}
    for spec in a.instanttrack:
        tag, path = spec.split("=", 1)
        t = np.load(path)
        rows = []
        ref_q_err = []
        for i, nm in enumerate([str(x) for x in t["names"]]):
            T = int(t["alive"][:, i].sum()) - 1
            if T < 1:
                continue
            quat = torch.as_tensor(t["root_quat"][: T + 1, i]).float()
            w_b = quat_apply_inverse(quat, torch.as_tensor(t["root_w"][: T + 1, i]).float())
            s = torch.cat([torch.as_tensor(t["root_p"][: T + 1, i]).float(), quat, torch.as_tensor(t["q"][: T + 1, i]).float(),
                           torch.as_tensor(t["root_v"][: T + 1, i]).float(), w_b, torch.as_tensor(t["qd"][: T + 1, i]).float()], -1)
            local = torch.as_tensor(t["f"][: T + 1, i]).long() - int(t["clip_start"][i])
            ci = names.index(nm)
            fr = lib.clip_start[ci] + torch.clamp(local, max=int(lib.clip_len[ci]) - 1)
            ref_q_err.append(float((lib.ref(fr).joint_pos - torch.as_tensor(t["ref_q"][t["f"][: T + 1, i]]).float()).abs().max()))
            rows.append({"clip": nm, **score(heads, lib, s, fr, int(t["clip_len"][i]) - 1), "native_return_norm": float(t["ret"][i] / (MAX_R * (int(t["clip_len"][i]) - 1))), "native_failed": bool(t["failed_eval"][i])})
        res["systems"][tag] = {"source": path, "reference_joint_maxabs_vs_ours": max(ref_q_err), "clips": rows}
    for tag, sysres in res["systems"].items():
        c = sysres["clips"]
        sysres["summary"] = {"clips": len(c), "macro_tracking_return": float(np.mean([r["tracking_return_norm"] for r in c])),
                             "completed_by_paper_termination": sum(1 for r in c if r["first_paper_termination_step"] is None and r["executed_steps"] >= r["planned_steps"] - 1),
                             "terminated_by_paper_rule": sum(1 for r in c if r["first_paper_termination_step"] is not None),
                             **{k + "_macro": float(np.mean([r[k + "_norm"] for r in c])) for k in TRACK}}
        print(tag, json.dumps(sysres["summary"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
