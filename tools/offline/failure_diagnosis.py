"""Why do full clips fail? CPU-only diagnosis on logged closed-loop trajectories + the checkpoint's world model.

python3 tools/offline/failure_diagnosis.py --ckpt <ckpt.pt> --traj <full-clip trajectories.npz> --out <json> [--pre-s 3.0]

PYTHONPATH = <variant>/src + stub mjlab (utils/lab_api/math.py + asset_zoo), as for anchor_offline_check.py.

Per clip: termination, margin to the termination thresholds, tracking error in the last `pre_s` seconds before
failure vs the clip's own earlier part, what the reference was doing there (root height, joint speed), and the
world model's open-loop 1- and 8-step prediction error on those states vs random control windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import importlib.util as _ilu
import sys as _sys
import types as _types

_src = Path(_ilu.find_spec("supermimic").submodule_search_locations[0])
for _name, _sub in (("supermimic.envs", "envs"), ("supermimic.envs.mdp", "envs/mdp")):
    if _name not in _sys.modules:
        _m = _types.ModuleType(_name)
        _m.__path__ = [str(_src / _sub)]
        _sys.modules[_name] = _m

from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
           "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
DT = 0.02
GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tilt_err(q_robot, q_ref):
    g = torch.tensor([0.0, 0.0, -1.0]).expand(q_robot.shape[0], 3)
    return (quat_apply_inverse(q_ref, g)[:, 2] - quat_apply_inverse(q_robot, g)[:, 2]).abs().numpy()


def load_wm(ckpt: Path, heads):
    import pathlib

    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    wc = dict(ck["config"]["world_model"])
    ss = wc.pop("state_spec")
    cfg = WorldModelCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in wc.items()}, state_spec=StateSpecCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ss.items()}))
    wm = WorldModel(cfg, heads, "cpu")
    wm.load_state_dict(ck["wm"])
    wm.eval()
    return wm, {"iteration": ck.get("iteration"), "num_updates": ck.get("num_updates")}


@torch.no_grad()
def wm_errors(wm, s, a, starts, k=8):
    """Open-loop k-step normalised squared error from each start index. s (T+1,71), a (T,29). Returns (n,k)."""
    starts = np.asarray([t for t in starts if t + k < s.shape[0]])
    if starts.size == 0:
        return np.zeros((0, k))
    xi = torch.zeros(len(starts), 33)
    xi[:, 0] = 1.0  # nominal friction, zero encoder bias / com offset
    cur = s[starts]
    out = []
    for j in range(k):
        cur = wm.step(cur, a[starts + j], xi)
        out.append(wm.spec.normalized_sq_error(cur, s[starts + j + 1]).numpy())
    return np.stack(out, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pre-s", type=float, default=3.0)
    ap.add_argument("--n-control", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(8)
    model = load_g1_model()
    fk = G1Kinematics(model, device="cpu")
    heads = AnalyticHeads(fk, TRACKED, "torso_link", "pelvis", torch.zeros(29, 2), torch.zeros(29))
    wm, meta = load_wm(a.ckpt, heads)
    t = np.load(a.traj)
    names = list(t["names"])
    pre = int(a.pre_s / DT)
    rng = np.random.default_rng(0)
    rows = []
    for i, nm in enumerate(names):
        T = int(t["executed_steps"][i])
        planned = int(t["planned_steps"][i])
        q = t["qpos"][i, : T + 1].astype(np.float32)
        qv = t["qvel"][i, : T + 1].astype(np.float32)
        rq = t["ref_qpos"][i, : T + 1].astype(np.float32)
        ae = t["action_executed"][i, :T].astype(np.float32)
        terms = [str(x) for x in t["term_names"]]
        flags = t["term_flags"][i, :T]
        fired = [terms[j] for j in range(len(terms)) if flags[T - 1, j]] if T > 0 else []
        failed = any(f in ("root_height", "root_tilt") for f in fired)
        hz = np.abs(q[:, 2] - rq[:, 2])
        tl = tilt_err(torch.as_tensor(q[:, 3:7]), torch.as_tensor(rq[:, 3:7]))
        je = np.abs(wrap(q[:, 7:] - rq[:, 7:]))
        ref_speed = np.abs(np.diff(rq[:, 7:], axis=0)).mean(-1) / DT  # rad/s, mean over joints
        s = torch.as_tensor(np.concatenate([q, qv], -1))
        at = torch.as_tensor(ae)
        # windows: last pre steps before the end (failure for failed clips), and random control windows away from the end
        tail = np.arange(max(0, T - pre), T - 8)
        ctrl_starts = rng.integers(10, max(11, T - pre - 10), size=a.n_control) if T > pre + 30 else np.array([], int)
        e_tail = wm_errors(wm, s, at, tail[::5]) if tail.size else np.zeros((0, 8))
        e_ctrl = wm_errors(wm, s, at, ctrl_starts) if ctrl_starts.size else np.zeros((0, 8))
        early = slice(0, max(1, T - pre))
        late = slice(max(0, T - pre), T + 1)
        row = {
            "clip": nm, "executed": T, "planned": planned, "completed": T == planned and not failed, "termination": "+".join(fired) if fired else ("clip_end" if T == planned else "?"),
            "fail_time_s": round(T * DT, 2) if failed else None, "fail_frac": round(T / planned, 3) if failed else None,
            "margin_height_max_frac": float((hz[early] / 0.25).max()) if T > pre else None,
            "margin_tilt_max_frac": float((tl[early] / 0.8).max()) if T > pre else None,
            "joint_err_early": {g: float(je[early][:, sl].mean()) for g, sl in GROUPS.items()},
            "joint_err_last3s": {g: float(je[late][:, sl].mean()) for g, sl in GROUPS.items()},
            "height_err_last3s_mean": float(hz[late].mean()), "tilt_err_last3s_mean": float(tl[late].mean()),
            "ref_root_z_last3s_min": float(rq[late, 2].min()), "ref_root_z_clip_min": float(rq[:, 2].min()),
            "ref_joint_speed_last3s": float(ref_speed[max(0, T - pre):T].mean()) if T > 1 else None, "ref_joint_speed_clip": float(ref_speed.mean()) if T > 1 else None,
            "ref_root_z_at_fail": float(rq[T, 2]) if failed else None, "robot_root_z_at_fail": float(q[T, 2]) if failed else None,
            "wm_nse1_last3s": float(e_tail[:, 0].mean()) if e_tail.size else None, "wm_nse8_last3s": float(e_tail[:, -1].mean()) if e_tail.size else None,
            "wm_nse1_control": float(e_ctrl[:, 0].mean()) if e_ctrl.size else None, "wm_nse8_control": float(e_ctrl[:, -1].mean()) if e_ctrl.size else None,
        }
        rows.append(row)
        print(f"{nm:28s} {row['termination']:12s} T={T:6d}/{planned:6d} ref_z_min={row['ref_root_z_clip_min']:.2f} "
              f"jerr early/late {row['joint_err_early']['legs']:.3f}/{row['joint_err_last3s']['legs']:.3f} legs, {row['joint_err_early']['arms']:.3f}/{row['joint_err_last3s']['arms']:.3f} arms "
              f"nse8 late/ctrl {row['wm_nse8_last3s'] if row['wm_nse8_last3s'] is None else round(row['wm_nse8_last3s'],3)}/{row['wm_nse8_control'] if row['wm_nse8_control'] is None else round(row['wm_nse8_control'],3)} "
              f"margin h/t {row['margin_height_max_frac'] and round(row['margin_height_max_frac'],2)}/{row['margin_tilt_max_frac'] and round(row['margin_tilt_max_frac'],2)}")
    failed = [r for r in rows if not r["completed"]]
    passed = [r for r in rows if r["completed"]]
    summ = {
        "n_failed": len(failed), "n_passed": len(passed),
        "failed_wm_nse8_last3s_mean": float(np.mean([r["wm_nse8_last3s"] for r in failed if r["wm_nse8_last3s"] is not None])) if failed else None,
        "failed_wm_nse8_control_mean": float(np.mean([r["wm_nse8_control"] for r in failed if r["wm_nse8_control"] is not None])) if failed else None,
        "passed_wm_nse8_control_mean": float(np.mean([r["wm_nse8_control"] for r in passed if r["wm_nse8_control"] is not None])) if passed else None,
        "failed_ref_root_z_at_fail_mean": float(np.mean([r["ref_root_z_at_fail"] for r in failed])) if failed else None,
        "failed_with_ref_root_below_0p5": sum(1 for r in failed if r["ref_root_z_at_fail"] is not None and r["ref_root_z_at_fail"] < 0.5),
        "passed_margin_height_max": sorted([(round(r["margin_height_max_frac"], 2), r["clip"]) for r in passed if r["margin_height_max_frac"] is not None], reverse=True)[:8],
        "passed_margin_tilt_max": sorted([(round(r["margin_tilt_max_frac"], 2), r["clip"]) for r in passed if r["margin_tilt_max_frac"] is not None], reverse=True)[:8],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"checkpoint": str(a.ckpt), "meta": meta, "pre_s": a.pre_s, "summary": summ, "clips": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
