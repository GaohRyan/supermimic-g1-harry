"""Pre-flight for the terrain training glue: run the variant's real Trainer (supermimic/training/loop.py) for a few
iterations on a laptop, on the plain-MuJoCo stand-in environment (tools/terrain/local_env.py).

What it exercises that the unit tests do not: the Trainer's construction for the terrain task (config asserts, env
creation arguments, height map hand-over to the world model, actor scan), --init-from (expanded networks and optimizer
moments), the collect / update / log loop, checkpoints with the terrain levels, and --resume.
What it cannot tell: anything about mjlab / MuJoCo-Warp, speed, or learning (tiny batches, a handful of updates).

  source tools/offline/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/tools/terrain:$PWD/tools/offline"
  .venv-sim/bin/python tools/terrain/local_trainer_run.py --ckpt <flat checkpoint> --out <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
import types
from pathlib import Path

import torch

import local_env as LE  # stand-ins for mjlab.managers, supermimic.envs registered bare
import sm_offline as smo

CONTRACT = Path("artifacts/projects/cuboi-g1/sim2sim/contract-bf237d7")
SCALARS: dict[str, list] = {}


def _install() -> None:
    """Stand-ins for what the Trainer imports and a laptop does not have."""
    tb = types.ModuleType("torch.utils.tensorboard")

    class SummaryWriter:
        def __init__(self, *a, **k):
            pass

        def add_scalar(self, key, value, step):
            SCALARS.setdefault(key, []).append((int(step), float(value)))

        def flush(self):
            pass

    tb.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = tb
    torch.cuda.synchronize = lambda *a, **k: None  # CPU run

    cfgmod = types.ModuleType("supermimic.envs.tracking_env_cfg")
    cfgmod.ANCHOR_BODY, cfgmod.ROOT_BODY, cfgmod.TRACKED_BODIES = "torso_link", "pelvis", smo.TRACKED
    cfgmod.anchor_body_for = lambda goal_contract: "pelvis" if goal_contract == "official" else "torso_link"
    sys.modules["supermimic.envs.tracking_env_cfg"] = cfgmod

    class TrainerEnv(LE.LocalEnv):
        """LocalEnv with the return conventions and manager handles the Trainer uses."""

        def __init__(self, args):
            files = [Path(f) for f in args.npz_files]
            super().__init__(CONTRACT, files[0].parent, num_envs=args.num_envs, terrain=args.terrain, clips=[f.stem for f in files], max_init_level=args.terrain_max_init_level,
                             episode_length_s=args.episode_length_s, reset_noise=args.reset_noise, sampling_mode=args.sampling_mode, record=True, capacity=args.replay_capacity,
                             seed=args.seed or 0, terrain_clip_regex=args.terrain_clip_regex)
            self.recorder_manager = types.SimpleNamespace(get_term=lambda name: self.recorder)

        def reset(self):
            return super().reset(), {}

        def step(self, action):
            obs, rew, term, tout = super().step(action)
            return obs, rew, term, tout, {}

    envs = sys.modules["supermimic.envs"]
    envs.SuperMimicEnvCfgArgs = lambda **kw: types.SimpleNamespace(**kw)
    envs.make_env = lambda args, device="cpu": TrainerEnv(args)


_install()

from supermimic.config import DataCfg, EnvCfg, ScheduleCfg, TrainConfig  # noqa: E402
from supermimic.imagination.policy_trainer import PolicyTrainerCfg  # noqa: E402
from supermimic.policy.nets import CriticCfg, PolicyCfg  # noqa: E402
from supermimic.training.loop import Trainer  # noqa: E402
from supermimic.world_model.state_spec import StateSpecCfg  # noqa: E402
from supermimic.world_model.wm_trainer import WMTrainerCfg  # noqa: E402
from supermimic.world_model.world_model import WorldModelCfg  # noqa: E402


def from_dict(cls, d: dict, **over):
    names = {f.name for f in dataclasses.fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    kw.update(over)
    return cls(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz-dir", default="data/processed/cuboi-g1/lafan40-local")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=70)
    ap.add_argument("--scan-mode", default="raw")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]
    wm = dict(c["world_model"])
    ss = wm.pop("state_spec")

    def config(**top) -> TrainConfig:
        return TrainConfig(
            run_name="local", log_root=Path(a.out), seed=0, device="cpu", data=DataCfg(npz_dir=Path(a.npz_dir)),
            env=from_dict(EnvCfg, c["env"], terrain_task=True, num_envs=a.envs, replay_capacity=200),
            world_model=from_dict(WorldModelCfg, wm, state_spec=from_dict(StateSpecCfg, ss)),
            wm_trainer=from_dict(WMTrainerCfg, c["wm_trainer"], batch_size=64), policy=from_dict(PolicyCfg, c["policy"], scan_mode=a.scan_mode), critic=from_dict(CriticCfg, c["critic"]),
            policy_trainer=from_dict(PolicyTrainerCfg, c["policy_trainer"], batch_size=64, horizon=8, compile=None, critic_minibatches=2, critic_epochs=2),
            schedule=ScheduleCfg(total_iterations=top.pop("iterations"), warmup_iterations=40, log_interval=10, checkpoint_interval=30), **top)

    rep: dict = {}
    t0 = time.time()
    tr = Trainer(config(iterations=a.iterations, init_from=Path(a.ckpt)))
    rep["first_layer_inputs"] = {"world_model": int(next(p for p in tr.wm.dynamics.parameters() if p.ndim == 2).shape[1]), "policy_decoder": int(next(p for p in tr.policy.decoder.parameters() if p.ndim == 2).shape[1]),
                                 "critic": int(next(p for p in tr.critic.parameters() if p.ndim == 2).shape[1])}
    tr.run()
    levels = tr.command.terrain_entity.terrain_levels.clone()
    rep["run"] = {"iterations": tr.iteration, "updates": tr.num_updates, "wall_s": time.time() - t0, "levels_hist": torch.bincount(levels, minlength=10).tolist(),
                  "scalars": {k: v[-1][1] for k, v in SCALARS.items()}}
    ck = torch.load(Path(a.out) / "local" / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    rep["checkpoint_keys"] = sorted(ck.keys())
    assert "terrain" in ck and torch.equal(ck["terrain"]["levels"], levels)
    assert all(torch.isfinite(p).all() for m in (tr.wm, tr.policy, tr.critic) for p in m.parameters())
    # ---- resume: the curriculum state must come back
    SCALARS.clear()
    tr2 = Trainer(config(iterations=a.iterations + 20, resume=Path(a.out) / "local" / "ckpt_latest.pt"))
    assert torch.equal(tr2.command.terrain_entity.terrain_levels, levels), "terrain levels not restored on resume"
    assert tr2.iteration == a.iterations and tr2.num_updates == tr.num_updates
    tr2.run()
    rep["resume"] = {"iterations": tr2.iteration, "updates": tr2.num_updates, "levels_restored": True}
    (Path(a.out) / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "run"}, default=float))
    print("scalars at the end of the first run:", {k: round(v, 5) for k, v in rep["run"]["scalars"].items() if k.split("/")[0] in ("terrain", "train", "wm") or "clearance" in k or "imag_return" in k})


if __name__ == "__main__":
    main()
