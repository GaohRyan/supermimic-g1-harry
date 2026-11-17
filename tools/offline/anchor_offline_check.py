"""CPU-only consistency + sensitivity check of the anchor observation path (no mjlab runtime needed).

python3 tools/offline/anchor_offline_check.py --ckpt <ckpt.pt> --traj <trajectories.npz> --npz-dir <npz of the 4 clips> \
    --g1-constants <mjlab g1_constants.py> --out <json> [--clip walk1_subject1] [--max-steps 4000]

Requires PYTHONPATH = <variant>/src plus a stub `mjlab` package exposing utils/lab_api/math.py and asset_zoo (see report).

1. Reproduction: rebuild the actor input analytically (93-d obs history via analytic_observation, goal via the motion
   library + anchor_relative_obs on torch-FK torso pose) from the logged closed-loop states and compare the policy's
   action with the logged clean action, which mjlab's own observations produced. Near-zero difference = the
   imagination-side observation (training) and the mjlab observation (collection / eval) agree.
2. Sensitivity: zero the 9 anchor dims; inject +/-15 deg yaw error into the anchor orientation. Reports the induced
   action change and the sign of the waist / hip yaw response.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

# Bypass supermimic.envs.__init__ (it imports the mjlab runtime); load only the pure-torch submodules we need.
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
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_yaw, quat_mul
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, anchor_relative_obs, flatten_history_term_major

TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
           "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
YAW_JOINTS = {"waist_yaw": 12, "left_hip_yaw": 2, "right_hip_yaw": 8}


def default_joint_pos(constants_py: Path, joint_names: list[str]) -> torch.Tensor:
    """KNEES_BENT_KEYFRAME joint_pos dict (regex keys) -> 29 values in model joint order."""
    txt = constants_py.read_text()
    m = re.search(r"KNEES_BENT_KEYFRAME = EntityCfg\.InitialStateCfg\((.*?)\n\)", txt, re.S)
    block = m.group(1)
    jp = re.search(r"joint_pos=\{(.*?)\}", block, re.S).group(1)
    pairs = re.findall(r"\"([^\"]+)\"\s*:\s*([-0-9.eE]+)", jp)
    out = []
    for n in joint_names:
        v = 0.0
        for k, val in pairs:
            if re.fullmatch(k, n):
                v = float(val)
        out.append(v)
    return torch.tensor(out, dtype=torch.float32)


def load_policy(ckpt: Path):
    import pathlib

    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    pc = ck["config"]["policy"]
    cfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    pol = Policy(cfg)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, ck["config"], {"iteration": ck.get("iteration"), "num_updates": ck.get("num_updates")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--g1-constants", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--max-steps", type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(8)
    model = load_g1_model()
    import mujoco

    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    q0 = default_joint_pos(a.g1_constants, jn)
    pol, cfg, meta = load_policy(a.ckpt)
    anchor = bool(cfg["env"].get("anchor_obs", False))
    files = sorted(str(p) for p in a.npz_dir.glob("*.npz"))
    lib = MotionLibrary(files, TRACKED, q0, device="cpu")
    fk = G1Kinematics(model, device="cpu")
    heads = AnalyticHeads(fk, TRACKED, "torso_link", "pelvis", torch.zeros(29, 2), q0)
    ai = heads.anchor_index
    t = np.load(a.traj)
    names = list(t["names"])
    i = names.index(a.clip)
    T = min(int(t["executed_steps"][i]), a.max_steps)
    qpos = torch.as_tensor(t["qpos"][i, : T + 1]).float()
    qvel = torch.as_tensor(t["qvel"][i, : T + 1]).float()
    act = torch.as_tensor(t["action_clean"][i, :T]).float()
    frame = torch.as_tensor(t["frame"][i, : T + 1]).long()
    # the library here holds only the 4 clips: map the logged global frame index onto this library's index
    clip_i = [Path(f).stem for f in files].index(a.clip)
    # logged frame = clip_start_pod + local; recover local offset from the first logged frame (clip start +1 after reset)
    local = frame - frame[0] + 1
    lib_frame = lib.clip_start[clip_i] + local
    s = torch.cat([qpos, qvel], dim=-1)  # (T+1, 71) env-origin relative, same as StateIO
    prev = torch.cat([torch.zeros(1, 29), act[:-1]], 0)  # action observed at step t = a_{t-1}
    xi = torch.zeros(T + 1, 33)
    obs = analytic_observation(s[:T], prev, xi[:T], q0)  # (T, 93) noise-free, encoder bias 0
    b = heads.robot_bodies(s[:T])
    ref = lib.bodies(lib_frame[:T])
    rel = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], ref.pos_w[:, ai], ref.quat_w[:, ai])
    goal = lib.goal_window(lib_frame[:T], cfg["env"]["future_frames"])
    if anchor:
        goal = torch.cat([goal, rel], -1)
    H = cfg["policy"]["history"]
    # history t-9..t, oldest first; before t=9 pad with the first obs (mjlab fills the buffer with the reset observation)
    idx = torch.arange(T)[:, None] + torch.arange(-H + 1, 1)[None, :]
    idx = idx.clamp(min=0)
    hist = obs[idx]  # (T, H, 93)
    flat = flatten_history_term_major(hist.permute(1, 0, 2))  # expects (H, N, 93)
    pred = pol(flat, goal)
    d = (pred - act).abs()
    valid = slice(H, T)
    rep = {"clip": a.clip, "steps_compared": T - H, "action_maxabs_diff": float(d[valid].max()), "action_meanabs_diff": float(d[valid].mean()),
           "action_meanabs_value": float(act[valid].abs().mean()), "first_20_steps_maxabs_diff": float(d[:H].max())}
    out = {"checkpoint": str(a.ckpt), "meta": meta, "anchor_obs": anchor, "reproduction": rep}
    if anchor:
        # variant hypotheses, to localise a mismatch if the reproduction is not exact
        g_zero = goal.clone(); g_zero[:, -9:] = 0
        p_zero = pol(flat, g_zero)
        resp = {"zero_anchor_meanabs_delta": float((p_zero - pred).abs()[valid].mean())}
        for deg in (15.0, -15.0):
            yaw = torch.full((T,), np.deg2rad(deg))
            q_ref_rot = quat_mul(quat_from_yaw(yaw), ref.quat_w[:, ai])  # rotate the reference anchor about world z
            p_ref_rot = ref.pos_w[:, ai]
            rel2 = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], p_ref_rot, q_ref_rot)
            g2 = torch.cat([goal[:, :-9], rel2], -1)
            p2 = pol(flat, g2)
            dd = (p2 - pred)[valid]
            resp[f"yaw_err_{int(deg):+d}deg"] = {"meanabs_delta_all": float(dd.abs().mean()),
                                                   **{k: float(dd[:, j].mean()) for k, j in YAW_JOINTS.items()}}
        # orientation-only variant: rot6d built from columns instead of rows (what rot6d_from_quat does for the goal)
        out["sensitivity"] = resp
        out["anchor_rel_stats"] = {"pos_norm_mean_m": float(rel[:, :3].norm(dim=-1).mean()), "pos_norm_max_m": float(rel[:, :3].norm(dim=-1).max())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
