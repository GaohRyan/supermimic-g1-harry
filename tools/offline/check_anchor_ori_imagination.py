"""CPU check of the imagination path for anchor_obs_mode="ori" (variant branch opt/anchor-ori).

  source tools/offline/env.sh && python3 tools/offline/check_anchor_ori_imagination.py

Random world model and policies, a few real reference states. Asserts: goal width 689 (full) / 686 (ori); the ori goal's
anchor part equals the rot6d part of the full goal for the same state; gradients reach the policy through the rollout.
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
from supermimic.data.replay_buffer import StartBatch  # noqa: E402
from supermimic.imagination.rollout import imagine  # noqa: E402
from supermimic.policy.nets import Policy  # noqa: E402
from supermimic.world_model.world_model import WorldModel  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = TrainConfig()
    lib, heads, names = smo.build(Path(os.environ["NPZ40_LOCAL"]), np.zeros(29), clips=["walk1_subject1"])
    wm = WorldModel(cfg.world_model, heads, "cpu")
    n, hist, H = 6, cfg.policy.history, 3
    frame = lib.clip_start[0] + torch.tensor([10, 500, 2000, 4000, 8000, 12000])
    q, v = lib.ref_qpos_qvel(frame)
    s = torch.cat([q, v], -1).float()
    s[:, 0:2] += 0.3 * torch.randn(n, 2)  # move the robot away from the reference so the anchor term is non-trivial
    start = StartBatch(state=s, xi=torch.zeros(n, 33), frame=frame + 1, prev_action=torch.zeros(n, 29), hist_states=s[None].repeat(hist, 1, 1), hist_actions=torch.zeros(hist, n, 29))
    out = {}
    for mode, extra in (("full", 9), ("ori", 6)):
        pc = cfg.policy
        pc.goal_dim = cfg.env.future_frames * 68 + extra
        pol = Policy(pc)
        ro = imagine(wm, pol, lib, start, H, cfg.env.future_frames, obs_noise=False, anchor_obs=True, anchor_obs_mode=mode)
        assert ro.goals.shape == (H + 1, n, 680 + extra), ro.goals.shape
        ro.rewards.sum().backward()
        g = sum(float(p.grad.abs().sum()) for p in pol.parameters() if p.grad is not None)
        assert g > 0, "no gradient reached the policy"
        out[mode] = ro.goals[0].detach()
    d = (out["ori"][:, 680:] - out["full"][:, 683:]).abs().max().item()
    d_ref = (out["ori"][:, :680] - out["full"][:, :680]).abs().max().item()
    pos_norm = out["full"][:, 680:683].norm(dim=-1).mean().item()
    assert d == 0.0 and d_ref == 0.0, (d, d_ref)
    print(f"OK: goal dims 689/686, ori part identical to full[683:689] (max diff {d}), reference part identical, mean |anchor pos| in full mode {pos_norm:.3f} m, gradients flow")


if __name__ == "__main__":
    main()
