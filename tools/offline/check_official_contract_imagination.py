"""CPU check of the imagination path for goal_contract="official" (variant branch repro/official-contract).

  source tools/offline/env.sh && python3 tools/offline/check_official_contract_imagination.py

Asserts: goal width 640 = 10 x 58 joint targets (stride 5, clamped at the clip end) + 10 x 6 relative orientation;
window frame 0 of the orientation block equals the single-frame rot6d of the ori mode with a pelvis anchor;
gradients reach the policy through the rollout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sm_offline as smo  # noqa: E402
from supermimic.config import TrainConfig  # noqa: E402
from supermimic.data.motion_convert import load_g1_model  # noqa: E402
from supermimic.data.replay_buffer import StartBatch  # noqa: E402
from supermimic.imagination.rollout import imagine  # noqa: E402
from supermimic.kinematics.g1_fk import G1Kinematics  # noqa: E402
from supermimic.policy.nets import Policy  # noqa: E402
from supermimic.world_model.heads import AnalyticHeads, anchor_relative_obs  # noqa: E402
from supermimic.world_model.world_model import WorldModel  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = TrainConfig()
    lib, _, _ = smo.build(Path(os.environ["NPZ40_LOCAL"]), np.zeros(29), clips=["walk1_subject1"])
    heads = AnalyticHeads(G1Kinematics(load_g1_model(), device="cpu"), smo.TRACKED, "pelvis", "pelvis", torch.zeros(29, 2), torch.zeros(29))
    wm = WorldModel(cfg.world_model, heads, "cpu")
    n, hist, H = 6, cfg.policy.history, 3
    last = int(lib.clip_last[0])
    frame = torch.tensor([10, 500, 2000, 4000, last - 20, last - 3])
    q, v = lib.ref_qpos_qvel(frame)
    s = torch.cat([q, v], -1).float()
    s[:, 3:7] = torch.nn.functional.normalize(s[:, 3:7] + 0.2 * torch.randn(n, 4), dim=-1)  # rotate the robot away from the reference
    start = StartBatch(state=s, xi=torch.zeros(n, 33), frame=frame, prev_action=torch.zeros(n, 29), hist_states=s[None].repeat(hist, 1, 1), hist_actions=torch.zeros(hist, n, 29))
    pc = cfg.policy
    pc.goal_dim, pc.goal_layout = 640, "official"
    pol = Policy(pc)
    ro = imagine(wm, pol, lib, start, H, cfg.env.future_frames, obs_noise=False, anchor_obs=True, anchor_obs_mode="ori_window", goal_contract="official")
    g = ro.goals[0].detach()
    assert ro.goals.shape == (H + 1, n, 640), ro.goals.shape
    ro.rewards.sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in pol.parameters() if p.grad is not None) > 0
    # joint-target block: frame k of the window = reference at frame + 5k, clamped
    idx = torch.minimum(frame[:, None] + 5 * torch.arange(10)[None], torch.full((n, 1), last))
    r = lib.ref(idx.reshape(-1))
    want = torch.cat([r.joint_pos, r.joint_vel], -1).reshape(n, 580)
    d_joint = (g[:, :580] - want).abs().max().item()
    # orientation block, window frame 0 = current reference root vs robot pelvis
    b = heads.robot_bodies(s)
    rb = lib.bodies(frame)
    ai = heads.anchor_index
    one = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], rb.pos_w[:, ai], rb.quat_w[:, ai])[:, 3:]
    d_ori0 = (g[:, 580:586] - one).abs().max().item()
    clamp_ok = bool((idx[-1] == last).sum() >= 9)
    assert d_joint == 0.0 and d_ori0 < 1e-5 and clamp_ok, (d_joint, d_ori0, clamp_ok)
    print(f"OK: goal 640 = 580 joint targets (max diff {d_joint}) + 60 rel-ori (frame-0 diff vs single-frame pelvis rot6d {d_ori0:.1e}); clip-end clamp ok; gradients flow")


if __name__ == "__main__":
    main()
