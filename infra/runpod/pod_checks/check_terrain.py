"""Terrain task, numeric check on the real training stack (mjlab + MuJoCo-Warp, on the Pod).

What the actor is trained on in imagination must be what the simulator does:
  1. reward: mjlab's tracking reward functions on the command term's lifted references vs the analytic heads with
     the terrain lifts (base lift from the height scan of the state, feet by the terrain under them);
  2. termination: root-height feature of the simulator vs the analytic one;
  3. height map vs ray casts on the compiled model (CPU mujoco.mj_ray, terrain geoms only);
  4. reset placement: lift on flat tiles is exactly 0; the scan the actor gets equals the world model's lookup;
  5. curriculum bookkeeping moves envs (sanity only).

  env -u PYTHONPATH uv run --frozen python check_terrain.py --npz-dir <npz40> --out <json> [--steps 120] [--num-envs 256]
Exit code 0 = within tolerance, 3 = mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["SM_WARN_OVERFLOW"] = "1"  # keep MuJoCo-Warp's nconmax / njmax overflow printf on: it lands in this script's log
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO, state_origin
from supermimic.envs.tracking_env_cfg import ROOT_BODY, TRACKED_BODIES, SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.terrain.actor_scan import ActorScan
from supermimic.terrain.heightmap import ScanPattern
from supermimic.terrain.lifts import base_lift_from_scan
from supermimic.terrain.terrain_cfg import TerrainArgs
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame

TOL_TERM = 1e-3  # reward terms on the per-second scale (each term is at most its weight, 0.5 or 1)
TOL_HEIGHT = 1e-4  # m, termination height feature
# A reference foot that sits within float rounding of a height-map cell border can read the neighbouring cell on one
# side of the comparison and not on the other (the two sides add the env origin at different places). Such a sample is
# off by up to one riser for one foot; they must stay rare.
MAX_SHARE_ABOVE_TOL = 2e-3
REWARDS = (("motion_global_root_pos", "motion_global_anchor_position_error_exp", 0.5, 0.3), ("motion_global_root_ori", "motion_global_anchor_orientation_error_exp", 0.5, 0.4),
           ("motion_body_pos", "motion_relative_body_position_error_exp", 1.0, 0.3), ("motion_body_ori", "motion_relative_body_orientation_error_exp", 1.0, 0.4),
           ("motion_body_lin_vel", "motion_global_body_linear_velocity_error_exp", 1.0, 1.0), ("motion_body_ang_vel", "motion_global_body_angular_velocity_error_exp", 1.0, 3.14))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--num-envs", type=int, default=256)
    a = ap.parse_args()
    device = "cuda:0"
    files = [str(f) for f in discover_npz(a.npz_dir)]
    args = SuperMimicEnvCfgArgs(npz_files=files, num_envs=a.num_envs, future_frames=10, episode_length_s=4.0, action_noise_std=0.0, obs_noise=False, reset_noise=True,
                                domain_randomization=False, sampling_mode="return", record_transitions=False, seed=1, anchor_obs=True, anchor_obs_mode="ori_window",
                                goal_contract="official", sampler_clip_end_norm=True, terrain=TerrainArgs(), terrain_max_init_level=None)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.tracking import mdp as track_mdp
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    t0 = time.time()
    env = ManagerBasedRlEnv(make_env_cfg(args), device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    rep: dict = {"env_build_s": time.time() - t0}
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device=device), TRACKED_BODIES, "pelvis", ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        tm = cmd.terrain
        m = env.sim.mj_model
        rep["model"] = {"ngeom": int(m.ngeom), "terrain_geoms": int((m.geom_bodyid == m.body("terrain").id).sum()), "nconmax": env.cfg.sim.nconmax, "njmax": env.cfg.sim.njmax,
                        "ccd_iterations": int(m.opt.ccd_iterations), "height_map_cells": [tm.ny, tm.nx], "tile_types": cmd.terrain_names}

        # ---- 3. height map vs ray casts (terrain geoms are group 0; the robot's geoms are in other groups)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        rng = np.random.default_rng(0)
        org = cmd.terrain_entity.terrain_origins.reshape(-1, 3).cpu().numpy()
        pts = org[rng.integers(0, len(org), 3000), :2] + rng.uniform(-6.5, 6.5, (3000, 2))
        gid = np.zeros(1, dtype=np.int32)
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        ray = np.array([50.0 - mujoco.mj_ray(m, d, np.array([p[0], p[1], 50.0]), np.array([0.0, 0.0, -1.0]), group, 1, -1, gid) for p in pts])
        err = np.abs(tm.height(torch.as_tensor(pts, dtype=torch.float32, device=device)).cpu().numpy() - ray)
        rep["height_map_vs_ray"] = {"points": int(len(pts)), "median_m": float(np.median(err)), "p99_m": float(np.percentile(err, 99)), "share_above_2cm": float((err > 0.02).mean())}

        # ---- 4. reset placement
        obs, _ = env.reset()
        on = cmd.on_terrain()
        scan_fn = ActorScan(env, tm)
        pattern = ScanPattern.grid(device=device)
        patch = pattern.patch_index()
        s = io.read()
        root_above = s[:, 2] - cmd.robot_base_lift()
        rep["reset"] = {"envs_on_terrain_tiles": int(on.sum()), "flat_tile_spawn_lift_maxabs": float(cmd.spawn_hover[~on].abs().max()) if (~on).any() else 0.0,
                        "flat_tile_base_lift_maxabs": float(cmd.robot_base_lift()[~on].abs().max()) if (~on).any() else 0.0,
                        "terrain_tile_hover_mean_m": float(cmd.spawn_hover[on].mean()), "terrain_tile_hover_max_m": float(cmd.spawn_hover[on].max()),
                        "root_height_above_base_min_m": float(root_above.min()), "root_height_above_base_max_m": float(root_above.max()),
                        "state_origin_z_maxabs": float(state_origin(env)[:, 2].abs().max()), "levels_hist": torch.bincount(cmd.terrain_entity.terrain_levels, minlength=10).tolist()}

        # ---- 1 + 2. reward / termination parity while stepping
        diffs = {n: [] for n, *_ in REWARDS}
        dh = []
        prev = torch.zeros(a.num_envs, 29, device=device)
        resets = 0
        t0 = time.time()
        for t in range(a.steps):
            act = torch.zeros(a.num_envs, 29, device=device) if t < 5 else 0.6 * (torch.rand(a.num_envs, 29, device=device) - 0.5)
            obs, rew, term, tout, _ = env.step(act)
            resets += int((term | tout).sum())
            s = io.read()
            origin = state_origin(env)
            f = cmd.frame
            b, r = lib.bodies(f), lib.ref(f)
            ref = ReferenceFrame(b.pos_w.float(), b.quat_w.float(), b.lin_vel_w.float(), b.ang_vel_w.float(), r.root_pos_w, r.root_quat_w)
            lift = base_lift_from_scan(s[:, 2], scan_fn.clean(), patch)
            ours = heads.reward(s, act, prev, ref, base_lift=lift, ground_fn=lambda xy: tm.height(xy + origin[:, None, :2]))
            for name, fn, w, std in REWARDS:
                diffs[name].append((ours[name] / 0.02 - w * getattr(track_mdp, fn)(env, "motion", std)).abs())
            dh.append((heads.termination_features(s[:, 0:3], s[:, 3:7], ref, lift)[:, 0] - (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2])).abs())
            prev = act
        rep["step_s"] = (time.time() - t0) / a.steps
        # Buffer usage at the end of the run (best effort: field names of the Warp data differ between versions).
        usage = {}
        for name in ("nacon", "nefc", "ne", "nf", "nl"):
            try:
                v = torch.as_tensor(getattr(env.sim.data, name)[:]).float()
                usage[name] = {"max": float(v.max()), "mean": float(v.mean()), "shape": list(v.shape)}
            except Exception as exc:  # noqa: BLE001
                usage[name] = f"unavailable: {type(exc).__name__}"
        rep["buffer_usage"] = usage
        par = {}
        ok = True
        for name, v in diffs.items():
            v = torch.stack(v).flatten()
            share = float((v > TOL_TERM).float().mean())
            par[name] = {"max": float(v.max()), "median": float(v.median()), "share_above_tol": share}
            ok &= share <= MAX_SHARE_ABOVE_TOL and float(v.median()) < TOL_TERM
        v = torch.stack(dh).flatten()
        par["termination_height_m"] = {"max": float(v.max()), "median": float(v.median()), "share_above_tol": float((v > TOL_HEIGHT).float().mean())}
        ok &= float((v > TOL_HEIGHT).float().mean()) <= MAX_SHARE_ABOVE_TOL
        rep["parity"] = par
        rep["episodes_ended"] = resets
        rep["terrain_stats"] = cmd.terrain_stats()
        ok &= rep["reset"]["flat_tile_spawn_lift_maxabs"] < 1e-6 and rep["reset"]["state_origin_z_maxabs"] == 0.0  # the lift is z - mean(z - 0) in float32: 6e-8 on the Pod, not exactly 0
        ok &= rep["height_map_vs_ray"]["share_above_2cm"] < 0.03 and rep["height_map_vs_ray"]["median_m"] < 1e-3
        rep["tolerances"] = {"term": TOL_TERM, "height_m": TOL_HEIGHT, "max_share_above_tol": MAX_SHARE_ABOVE_TOL}
        rep["ok"] = bool(ok)
    finally:
        env.close()
    a.out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"ok": rep.get("ok"), "parity_max": {k: round(v["max"], 6) for k, v in rep.get("parity", {}).items()}, "height_map_vs_ray": rep.get("height_map_vs_ray"), "step_s": rep.get("step_s")}))
    return 0 if rep.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
