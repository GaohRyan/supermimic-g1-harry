"""Closed-loop runs on the generated terrain in plain MuJoCo on a laptop (tools/terrain/local_env.py).

  baseline   put robots on chosen tiles (terrain type x level), one 10 s window each from locomotion clips, and report
             per tile: share that did not fall, tracking return, progress along the reference, climbed height.
             With a flat-ground checkpoint this is the "before terrain training" number.

  source tools/offline/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/tools/terrain:$PWD/tools/offline"
  .venv-sim/bin/python tools/terrain/local_terrain_run.py baseline --ckpt <pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import local_env as LE
import sm_offline as smo
from supermimic.policy.nets import Policy, PolicyCfg

CONTRACT = Path("artifacts/projects/cuboi-g1/sim2sim/contract-bf237d7")


def load_policy(path: Path, expand_scan: str | None = None):
    """Policy of a checkpoint. ``expand_scan`` ("raw"): load a flat-ground policy into the terrain policy with a
    zero-initialised scan input, as the warm start of a terrain run does (it then acts exactly like the flat one)."""
    from supermimic.terrain.expand import load_state_dict_expand

    ck = smo.load_ckpt(path)
    pc = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"]["policy"].items()}
    if expand_scan:
        pc["scan_mode"] = expand_scan
    pol = Policy(PolicyCfg(**pc))
    load_state_dict_expand(pol, ck["policy"], allow_missing_prefixes=("scan_encoder.",))
    pol.eval()
    return pol, ck["config"]


@torch.no_grad()
def baseline(a) -> None:
    from supermimic.eval.terrain_eval import format_rows, run_tile_windows, tile_list
    from supermimic.terrain.actor_scan import actor_scan_for, relief_scaled

    torch.set_num_threads(a.threads)
    pol, cfg = load_policy(Path(a.ckpt), a.expand_scan)
    per = a.envs_per_tile
    n = per * len(a.types) * len(a.levels)
    from supermimic.terrain.terrain_cfg import TerrainArgs

    targs = TerrainArgs(p_slope=0.05, p_rough=0.05) if a.heightfields else TerrainArgs()
    env = LE.LocalEnv(CONTRACT, Path(a.npz_dir), num_envs=n, terrain=targs, max_init_level=None, seed=a.seed, reset_noise=a.reset_noise, episode_length_s=a.seconds,
                      command_kwargs={"terrain_curriculum": False})
    cmd = env.command
    tiles = tile_list(cmd.terrain_names, a.types, a.levels)
    tile_of = torch.arange(n) // per
    env.terrain.place(torch.tensor([tiles[i][1] for i in tile_of.tolist()]), torch.tensor([tiles[i][0] for i in tile_of.tolist()]))
    scan_fn = actor_scan_for(pol, env, None if a.scan == "flat" else cmd.terrain)
    if scan_fn is not None and a.scan_relief != 1.0:
        scan_fn = relief_scaled(scan_fn, a.scan_relief)
    obs = env.reset()
    t0 = time.time()
    res = run_tile_windows(cmd, env.step, obs, pol, scan_fn, tile_of, tiles, env.max_episode_length, env.terrain_args)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"protocol_id": "cuboi-g1/terrain-window-nominal-v1 (local MuJoCo C)", "checkpoint": str(a.ckpt), "policy_scan_mode": pol.cfg.scan_mode, "scan_fed": a.scan, "scan_relief": a.scan_relief,
           "seconds": a.seconds, "reset_noise": a.reset_noise, "per_tile": per, "seed": a.seed,
           "simulator": "mujoco C (CPU), tools/terrain/local_env.py: nominal context, no action noise / delay", "wall_seconds": time.time() - t0, "tiles": res["tiles"]}
    (out / "results.json").write_text(json.dumps(rep, indent=1))
    print(format_rows(res["tiles"]))
    print(f"wall {time.time() - t0:.0f}s -> {out / 'results.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("baseline")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--npz-dir", default="data/processed/cuboi-g1/lafan40-local")
    p.add_argument("--types", nargs="+", default=["flat", "stairs_up", "stairs_down", "stairs_random"])
    p.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9])
    p.add_argument("--envs-per-tile", type=int, default=16)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--reset-noise", action="store_true")
    p.add_argument("--scan", choices=["terrain", "flat"], default="terrain", help="what a scan policy is fed: the real terrain, or the flat-ground scan (blanking test)")
    p.add_argument("--scan-relief", type=float, default=1.0, help="scale every height difference in the actor's scan (0.9: a 20 cm riser looks like 18 cm)")
    p.add_argument("--expand-scan", default=None, choices=["raw"], help="load a flat-ground checkpoint into a scan policy (zero-initialised scan input)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--heightfields", action="store_true", help="add the slope and rough height-field tiles to the terrain (off in training by default)")
    a = ap.parse_args()
    {"baseline": baseline}[a.cmd](a)


if __name__ == "__main__":
    main()
