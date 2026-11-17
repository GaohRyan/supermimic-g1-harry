"""Videos of a terrain checkpoint on the stairs, rendered on a laptop (plain MuJoCo C, tools/terrain/local_env.py).

  source tools/offline/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/tools/terrain:$PWD/tools/offline"
  MUJOCO_GL=glfw .venv-sim/bin/python tools/terrain/render_stairs.py --ckpt <pt> --out <dir> [--level 9] [--per-tile 8]

Per terrain type it runs ``--per-tile`` 10 s windows (same loop and numbers as local_terrain_run.py baseline) and renders
``--videos`` of them. Which windows are rendered is fixed before looking at the outcome: those whose REFERENCE path
changes height the most (most risers on the path). A rendered window that falls is rendered as it is; the video stops
at the fall. Writes <out>/results.json (per-tile rows, and for every rendered window: clip, fell, steps, height change)
<out>/trajectories.npz (the closed-loop qpos of every window; the videos are drawn from it) and <out>/mp4/<type>-<k>.mp4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch

import local_env as LE
from local_terrain_run import CONTRACT, load_policy


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz-dir", default="data/processed/cuboi-g1/lafan40-local")
    ap.add_argument("--types", nargs="+", default=["stairs_up", "stairs_down", "stairs_random"])
    ap.add_argument("--level", type=int, default=9)
    ap.add_argument("--per-tile", type=int, default=8)
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, nargs=2, default=[960, 540])
    a = ap.parse_args()
    from supermimic.eval.terrain_eval import format_rows, run_tile_windows, tile_list
    from supermimic.terrain.actor_scan import actor_scan_for
    from supermimic.terrain.terrain_cfg import TerrainArgs

    torch.set_num_threads(4)
    pol, _ = load_policy(Path(a.ckpt))
    per = a.per_tile
    n = per * len(a.types)
    env = LE.LocalEnv(CONTRACT, Path(a.npz_dir), num_envs=n, terrain=TerrainArgs(), max_init_level=None, seed=a.seed, reset_noise=False, episode_length_s=a.seconds,
                      command_kwargs={"terrain_curriculum": False})
    cmd = env.command
    tiles = tile_list(cmd.terrain_names, a.types, [a.level])
    tile_of = torch.arange(n) // per
    env.terrain.place(torch.tensor([tiles[i][1] for i in tile_of.tolist()]), torch.tensor([tiles[i][0] for i in tile_of.tolist()]))
    scan_fn = actor_scan_for(pol, env, cmd.terrain)
    obs = env.reset()
    clip_of = [cmd.library.clips[i].name for i in cmd.library.frame_clip[cmd.frame].tolist()]
    traj = [[d.qpos.copy()] for d in env.datas]
    alive = np.ones(n, dtype=bool)

    def step(act):
        out = env.step(act)
        done = (out[2] | out[3]).numpy()
        for e in range(n):
            if alive[e] and not done[e]:
                traj[e].append(env.datas[e].qpos.copy())
        alive[done] = False
        return out

    res = run_tile_windows(cmd, step, obs, pol, scan_fn, tile_of, tiles, env.max_episode_length, env.terrain_args)
    print(format_rows(res["tiles"]))
    pe = res["per_env"]
    out = Path(a.out)
    (out / "mp4").mkdir(parents=True, exist_ok=True)
    # Real closed-loop states of every window (NaN after the window ended): the videos are drawn from this file only.
    T = max(len(t) for t in traj)
    qpos = np.full((T, n, env.model.nq), np.nan)
    for e, t in enumerate(traj):
        qpos[: len(t), e] = np.stack(t)
    np.savez_compressed(out / "trajectories.npz", qpos=qpos, control_dt=env.step_dt, clips=np.array(clip_of), tile_type=np.array([a.types[i] for i in tile_of.tolist()]), fell=pe["fell"].numpy())
    model = env.model
    w, h = a.size
    model.vis.global_.offwidth, model.vis.global_.offheight = max(model.vis.global_.offwidth, w), max(model.vis.global_.offheight, h)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    fps = int(round(1.0 / env.step_dt))
    rendered = []
    for i, t in enumerate(a.types):
        ids = (tile_of == i).nonzero().squeeze(-1)
        order = ids[torch.argsort(pe["ref_height_change"][ids], descending=True)][: a.videos].tolist()
        for k, e in enumerate(order):
            q = np.stack(traj[e])
            path = out / "mp4" / f"{t}-{k}.mp4"
            d_xy = q[-1, :2] - q[0, :2]
            az = np.degrees(np.arctan2(d_xy[1], d_xy[0])) + 90.0 + 25.0  # from the side, slightly ahead
            ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "22", str(path)], stdin=subprocess.PIPE)
            look = q[0, :3].copy()
            for s in range(len(q)):
                data.qpos[:] = q[s]
                mujoco.mj_forward(model, data)
                look = 0.9 * look + 0.1 * q[s, :3]
                cam.lookat[:] = look
                cam.distance, cam.elevation, cam.azimuth = 4.0, -12.0, az
                r.update_scene(data, cam)
                ff.stdin.write(r.render().tobytes())
            ff.stdin.close()
            ff.wait()
            row = {"video": str(path), "type": t, "riser_cm": res["tiles"][i]["riser_cm"], "env": int(e), "clip": clip_of[e], "fell": bool(pe["fell"][e]), "steps": int(pe["steps"][e]),
                   "planned_steps": int(pe["planned_steps"][e]), "ref_height_change_m": float(pe["ref_height_change"][e]), "robot_height_change_m": float(pe["height_change"][e])}
            rendered.append(row)
            print(row)
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    for row in rendered:
        row["sha256"] = sha(row["video"])
    (out / "results.json").write_text(json.dumps({"checkpoint": str(a.ckpt), "checkpoint_sha256": sha(a.ckpt), "trajectories_sha256": sha(out / "trajectories.npz"), "level": a.level, "simulator": "mujoco C (CPU), tools/terrain/local_env.py, nominal, no reset noise", "seed": a.seed, "per_tile": per,
                                                  "selection": "windows with the largest reference height change, chosen before looking at the outcome", "tiles": res["tiles"], "rendered": rendered}, indent=1))


if __name__ == "__main__":
    main()
