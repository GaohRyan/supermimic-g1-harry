"""Numeric check: mjlab anchor observation (what the actor sees in collection and evaluation) vs the
analytic anchor_relative_obs + torch FK used inside imagination (what the actor is trained on).

Run on the Pod (needs mjlab + CUDA):
  env -u PYTHONPATH uv run --frozen python check_anchor_obs.py --npz-dir <npz40> --out <json> [--steps 60] [--num-envs 64]
Exit code 0 = max abs difference below tolerance, 3 = mismatch (training/eval anchor observation disagree).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES, SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import ANCHOR_OBS_DIM, AnalyticHeads, analytic_observation, anchor_relative_obs

TOL_POS_M = 2e-3
TOL_ROT6D = 2e-3
TOL_OBS = 2e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--anchor-obs-mode", choices=["full", "ori", "ori_window"], default="full", help="ori = rot6d only (variant branch opt/anchor-ori)")
    a = ap.parse_args()
    official = a.anchor_obs_mode == "ori_window"  # official-release goal layout (variant official-contract)
    extra = {"full": ANCHOR_OBS_DIM, "ori": 6, "ori_window": 60}[a.anchor_obs_mode]
    base = 580 if official else 680
    mode_kw = {} if a.anchor_obs_mode == "full" else {"anchor_obs_mode": a.anchor_obs_mode}  # older code has no such field
    if official:
        mode_kw["goal_contract"] = "official"
    device = "cuda:0"
    files = [str(f) for f in discover_npz(a.npz_dir, names=["walk1_subject1", "dance1_subject1", "jumps1_subject1", "fight1_subject2"])]
    args = SuperMimicEnvCfgArgs(npz_files=files, num_envs=a.num_envs, future_frames=10, episode_length_s=100.0, action_noise_std=0.0,
                                obs_noise=False, reset_noise=True, domain_randomization=False, sampling_mode="return",
                                record_transitions=False, seed=1, anchor_obs=True, **mode_kw)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env = ManagerBasedRlEnv(make_env_cfg(args), device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, "pelvis" if official else ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        ai = heads.anchor_index
        q0 = lib.default_joint_pos
        obs, _ = env.reset()
        d_pos, d_rot, d_obs, d_goal, rows = [], [], [], [], []
        prev_a = torch.zeros(a.num_envs, 29, device=device)
        for t in range(a.steps):
            s = io.read()
            # --- analytic (imagination-side) ---
            b = heads.robot_bodies(s)
            ref = lib.bodies(cmd.frame)
            rel = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], ref.pos_w[:, ai], ref.quat_w[:, ai])
            if official:
                from supermimic.world_model.heads import anchor_ori_window_obs

                win = anchor_ori_window_obs(b["quat"][:, ai], lib.root_quat_window_official(cmd.frame))
                goal_a = torch.cat([lib.goal_window_official(cmd.frame), win], dim=-1)
            else:
                goal_a = torch.cat([lib.goal_window(cmd.frame, 10), rel], dim=-1)
            xi = torch.zeros(a.num_envs, 33, device=device)
            xi[:, 0] = 1.0  # friction (unused by analytic_observation)
            obs_a = analytic_observation(s, prev_a, xi, q0)
            # --- mjlab (collection/eval-side) ---
            goal_m = obs["goal"]
            obs_m = obs["policy"]
            # mjlab policy group is term-major history (10 frames); the newest frame of each term is the last slice
            H = 10
            terms = [("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29)]
            newest = []
            start = 0
            for _, dim in terms:
                blk = obs_m[:, start : start + H * dim].reshape(a.num_envs, H, dim)
                newest.append(blk[:, -1])
                start += H * dim
            obs_m_newest = torch.cat(newest, dim=-1)
            assert goal_m.shape[1] == base + extra, f"goal width {goal_m.shape[1]} != {base + extra} for mode {a.anchor_obs_mode}"
            dp = (goal_m[:, -ANCHOR_OBS_DIM:-6] - rel[:, :3]).abs().max().item() if a.anchor_obs_mode == "full" else 0.0  # ori modes: no position term
            if official:  # whole 60-D orientation window and the 580-D joint-target window, env vs analytic
                dr = (goal_m[:, 580:] - goal_a[:, 580:]).abs().max().item()
                dg = (goal_m[:, :580] - goal_a[:, :580]).abs().max().item()
            else:
                dr = (goal_m[:, -6:] - rel[:, 3:]).abs().max().item()
                dg = (goal_m[:, :-extra] - goal_a[:, :-ANCHOR_OBS_DIM]).abs().max().item()
            do = (obs_m_newest - obs_a).abs().max().item()
            # also compare FK anchor pose against the simulator's body pose directly
            fk_pos = (b["pos"][:, ai] + io.origins - cmd.robot_anchor_pos_w).abs().max().item()
            qd = (b["quat"][:, ai] - cmd.robot_anchor_quat_w).abs().min(dim=-1).values  # sign ambiguity handled below
            qsum = (b["quat"][:, ai] + cmd.robot_anchor_quat_w).abs().max(dim=-1).values
            qdiff = (b["quat"][:, ai] - cmd.robot_anchor_quat_w).abs().max(dim=-1).values
            fk_quat = torch.minimum(qsum, qdiff).max().item()
            d_pos.append(dp); d_rot.append(dr); d_obs.append(do); d_goal.append(dg)
            rows.append({"t": t, "anchor_pos_maxabs": dp, "anchor_rot6d_maxabs": dr, "goal_ref_part_maxabs": dg, "obs_newest_maxabs": do,
                         "fk_vs_sim_anchor_pos": fk_pos, "fk_vs_sim_anchor_quat": fk_quat,
                         "anchor_rel_pos_mean_norm": rel[:, :3].norm(dim=-1).mean().item()})
            act = torch.zeros(a.num_envs, 29, device=device) if t < 5 else (torch.rand(a.num_envs, 29, device=device) - 0.5)
            obs, _, _, _, _ = env.step(act)
            prev_a = act
        summ = {"anchor_pos_maxabs": max(d_pos), "anchor_rot6d_maxabs": max(d_rot), "goal_ref_part_maxabs": max(d_goal), "obs_newest_maxabs": max(d_obs),
                "fk_vs_sim_anchor_pos_max": max(r["fk_vs_sim_anchor_pos"] for r in rows), "fk_vs_sim_anchor_quat_max": max(r["fk_vs_sim_anchor_quat"] for r in rows),
                "anchor_obs_mode": a.anchor_obs_mode, "goal_width": base + extra, "tolerances": {"pos_m": TOL_POS_M, "rot6d": TOL_ROT6D, "obs": TOL_OBS}}
        ok = summ["anchor_pos_maxabs"] < TOL_POS_M and summ["anchor_rot6d_maxabs"] < TOL_ROT6D and summ["goal_ref_part_maxabs"] < TOL_OBS
        summ["ok"] = bool(ok)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summ, "steps": rows, "note": "obs_newest compares the noise-free analytic 93-d observation with the newest history frame of mjlab's policy group; obs noise off, DR off, random actions after 5 zero steps"}, indent=1))
        print(json.dumps(summ))
        return 0 if ok else 3
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
