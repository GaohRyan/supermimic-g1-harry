"""Shared helpers for the CPU-only tools: import the pure-torch parts of the variant without the mjlab runtime.

Needs `source tools/offline/env.sh` (PYTHONPATH = variant src + stub mjlab). Import this module before `supermimic.*`.
"""
from __future__ import annotations

import importlib.util as _ilu
import pathlib
import sys as _sys
import types as _types
from pathlib import Path

import torch

_src = Path(_ilu.find_spec("supermimic").submodule_search_locations[0])
for _name, _sub in (("supermimic.envs", "envs"), ("supermimic.envs.mdp", "envs/mdp")):
    if _name not in _sys.modules:  # bypass supermimic.envs.__init__ (imports the mjlab runtime)
        _m = _types.ModuleType(_name)
        _m.__path__ = [str(_src / _sub)]
        _sys.modules[_name] = _m

from supermimic.data.motion_convert import load_g1_model  # noqa: E402
from supermimic.envs.mdp.motion_library import MotionLibrary  # noqa: E402
from supermimic.kinematics.g1_fk import G1Kinematics  # noqa: E402
from supermimic.policy.nets import Policy, PolicyCfg  # noqa: E402
from supermimic.world_model.heads import AnalyticHeads  # noqa: E402

TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
           "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")


def load_ckpt(path: Path) -> dict:
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    return torch.load(path, map_location="cpu", weights_only=True)


def load_policy(path: Path):
    ck = load_ckpt(path)
    pc = ck["config"]["policy"]
    pol = Policy(PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()}))
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, ck["config"], {"iteration": ck.get("iteration"), "num_updates": ck.get("num_updates")}


def build(npz_dir: Path, default_joint_pos, soft_limits=None, clips: list[str] | None = None, anchor: str = "torso_link"):
    """-> (MotionLibrary on CPU with all npz in name order, AnalyticHeads, clip names)."""
    files = sorted(str(p) for p in Path(npz_dir).glob("*.npz"))
    if clips:
        files = [f for f in files if Path(f).stem in clips]
    q0 = torch.as_tensor(default_joint_pos, dtype=torch.float32)
    lib = MotionLibrary(files, TRACKED, q0, device="cpu")
    fk = G1Kinematics(load_g1_model(), device="cpu")
    sl = torch.as_tensor(soft_limits, dtype=torch.float32) if soft_limits is not None else torch.zeros(29, 2)
    heads = AnalyticHeads(fk, TRACKED, anchor, "pelvis", sl, q0)  # official-contract variant anchors on the pelvis
    return lib, heads, [Path(f).stem for f in files]
