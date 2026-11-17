"""End-to-end smoke test of the terrain training path on a laptop (no GPU, no mjlab runtime).

  1. collect replay data on the generated terrain with the real command term and recorder (tools/terrain/local_env.py),
     acting with a flat-ground checkpoint warm-started into the terrain networks (scan inputs, zero-initialised);
  2. check the replay rows (origins, scans of stored states), and measure the warm-started world model's error on
     flat tiles and on terrain tiles;
  3. run a few world-model and policy updates on that data (small batches) and check that everything is finite and
     that the new scan columns start to learn.

  source tools/offline/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/tools/terrain:$PWD/tools/offline"
  .venv-sim/bin/python tools/terrain/local_train_smoke.py --ckpt <flat checkpoint> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import local_env as LE
import sm_offline as smo
from supermimic.data.motion_convert import load_g1_model
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import heading_yaw
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.terrain.actor_scan import ActorScan
from supermimic.terrain.expand import expand_optimizer_state, load_state_dict_expand
from supermimic.terrain.heightmap import ScanPattern
from supermimic.training.optim import load_optimizer_state
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

CONTRACT = Path("artifacts/projects/cuboi-g1/sim2sim/contract-bf237d7")


def tup(d):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz-dir", default="data/processed/cuboi-g1/lafan40-local")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--scan-mode", default="raw", choices=["raw", "latent", "none"])
    ap.add_argument("--scan-encoder", default=None)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-init-level", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"checkpoint": a.ckpt, "scan_mode": a.scan_mode}

    # ---------------------------------------------------------------- environment and warm-started networks
    env = LE.LocalEnv(CONTRACT, Path(a.npz_dir), num_envs=a.envs, max_init_level=a.max_init_level, seed=0, record=True, capacity=a.steps + 50)
    cmd, lib, c = env.command, env.command.library, env.contract
    import pathlib

    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    heads = AnalyticHeads(G1Kinematics(load_g1_model(), device="cpu"), smo.TRACKED, "pelvis", "pelvis", torch.tensor(c["soft_joint_pos_limits"]), torch.tensor(c["default_joint_pos"]))
    wc = dict(cfg["world_model"])
    ss = wc.pop("state_spec")
    wc.pop("scan_dim", None)
    wm = WorldModel(WorldModelCfg(**tup(wc), state_spec=StateSpecCfg(**tup(ss)), scan_dim=187), heads, "cpu")
    pc = tup(dict(cfg["policy"]))
    pc["scan_mode"] = a.scan_mode
    policy = Policy(PolicyCfg(**pc))
    critic = Critic(CriticCfg(**tup(dict(cfg["critic"]))), xi_dim=38, pcfg=PolicyCfg(**pc), scan_dim=187)
    exp = {"wm": load_state_dict_expand(wm, ck["wm"])["expanded"], "policy": load_state_dict_expand(policy, ck["policy"], allow_missing_prefixes=("scan_encoder.",))["expanded"],
           "critic": load_state_dict_expand(critic, ck["critic"])["expanded"]}
    if a.scan_mode == "latent":
        from supermimic.terrain.scan_ae import load_frozen_encoder

        policy.scan_encoder.load_from(load_frozen_encoder(a.scan_encoder)[0])
    wm.set_terrain(cmd.terrain)
    heads.reward_cfg.w_foot_clearance = 30.0
    rep["expanded_inputs"] = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in exp.items()}
    print("expanded inputs:", rep["expanded_inputs"])

    # ---------------------------------------------------------------- 1. collect
    scan_fn = ActorScan(env, cmd.terrain, noise=0.03) if a.scan_mode != "none" else None
    obs = env.reset()
    t0 = time.time()
    episodes = fails = 0
    scan_check = 0.0
    with torch.no_grad():
        for k in range(a.steps):
            if k % 100 == 0:  # the world model's lookup of the current state equals the scan the actor is given (before noise)
                s = env.state()
                scan_check = max(scan_check, float((wm.scan(s, cmd.origin) - ActorScan(env, cmd.terrain).clean()).abs().max()))
            act = policy(obs["policy"], obs["goal"], scan=scan_fn() if scan_fn is not None else None)
            obs, rew, term, tout = env.step(act)
            episodes += int((term | tout).sum())
            fails += int(term.sum())
    buf = env.recorder.buffer
    rep["collect"] = {"envs": a.envs, "steps": a.steps, "episodes": episodes, "failures": fails, "wall_s": time.time() - t0, "terrain_stats": cmd.terrain_stats(),
                      "wm_scan_vs_actor_scan_maxabs": scan_check}
    print(f"collected {a.envs} x {a.steps} steps in {time.time() - t0:.0f}s: {episodes} episodes, {fails} failures; world-model scan vs actor scan max |diff| {scan_check:.1e}")

    # ---------------------------------------------------------------- 2. replay checks and world-model error by tile type
    pat = ScanPattern.grid()
    patch = pat.patch_index()
    w = buf.sample_windows(512, 8)
    above = []
    for j in range(9):
        s = w.states[j]
        scan = cmd.terrain.scan(s[:, 0:3] + w.origin, heading_yaw(s[:, 3:7]), pat)
        above.append(scan[:, patch].mean(-1))
    above = torch.stack(above)
    # tile type of a window: column of the tile whose origin the window carries
    tile_xy = cmd.terrain_entity.terrain_origins[..., :2].reshape(-1, 2)
    col = (torch.cdist(w.origin[:, :2], tile_xy).argmin(1) % cmd.terrain_entity.terrain_origins.shape[1])
    flat = col == cmd.flat_col
    with torch.no_grad():
        s_pred = w.states[0]
        e1 = None
        errs = []
        for j in range(8):
            s_pred = wm.step(s_pred, w.a_exec[j], w.xi, origin=w.origin)
            errs.append(wm.spec.normalized_sq_error(s_pred, w.states[j + 1]))
        errs = torch.stack(errs)  # (8, n, ...) -> mean over state dims if needed
        errs = errs.reshape(8, errs.shape[1], -1).mean(-1)
    rep["replay"] = {"rows": int(buf.valid.sum()), "origin_z_all_zero": bool((buf.origin[..., 2] == 0).all()),
                     "root_height_above_mean_terrain": {"min": float(above.min()), "median": float(above.median()), "max": float(above.max())},
                     "windows_flat_tiles": int(flat.sum()), "windows_terrain_tiles": int((~flat).sum()),
                     "wm_nse_1step": {"flat_tiles": float(errs[0, flat].mean()), "terrain_tiles": float(errs[0, ~flat].mean())},
                     "wm_nse_8step": {"flat_tiles": float(errs[7, flat].mean()), "terrain_tiles": float(errs[7, ~flat].mean())}}
    print("replay:", json.dumps(rep["replay"]))

    # ---------------------------------------------------------------- 3. a few updates
    wt = WMTrainerCfg(**{**tup(dict(cfg["wm_trainer"])), "batch_size": a.batch})
    pt = PolicyTrainerCfg(**{**tup(dict(cfg["policy_trainer"])), "batch_size": a.batch, "horizon": a.horizon, "compile": None, "scan_noise": 0.03, "critic_minibatches": 2, "critic_epochs": 2})
    wm_tr = WorldModelTrainer(wm, lib, wt)
    pi_tr = PolicyTrainer(wm, policy, critic, lib, pt)
    fresh = []
    for key, (opt, module, kw) in {"wm_opt": (wm_tr.opt, wm, {"weight_decay": wt.weight_decay}), "policy_opt": (pi_tr.opt, policy, {"betas": pt.betas, "weight_decay": pt.weight_decay}),
                                   "critic_opt": (pi_tr.critic_opt, critic, {"weight_decay": pt.weight_decay})}.items():
        sd = expand_optimizer_state(ck[key], list(module.parameters()))
        if sd is None or not load_optimizer_state(opt, sd, **kw):
            fresh.append(key)
    rep["optimizer_state_fresh_for"] = fresh
    first_w = lambda mod: next(p for n, p in mod.named_parameters() if p.ndim == 2)  # noqa: E731
    cols = {"wm": (first_w(wm.dynamics), 187), "critic": (first_w(critic), 187)}
    if a.scan_mode != "none":
        cols["policy_decoder"] = (first_w(policy.decoder), 187 if a.scan_mode == "raw" else policy.cfg.scan_latent_dim)
    before = {k: float(p[:, -n:].abs().mean()) for k, (p, n) in cols.items()}
    logs = []
    t0 = time.time()
    for u in range(a.updates):
        m_wm = wm_tr.update(buf)
        m_pi = pi_tr.update(buf)
        logs.append({"wm": m_wm, "policy": m_pi})
        print(f"update {u}: wm loss {m_wm['loss']:.4f} nse1 {m_wm['nse_1step']:.5f} | imag return {m_pi['imag_return_undisc']:.4f} survival {m_pi['imag_survival_H']:.3f} "
              f"clearance {m_pi.get('imag_r/foot_clearance', float('nan')):.5f} | {time.time() - t0:.0f}s")
    after = {k: float(p[:, -n:].abs().mean()) for k, (p, n) in cols.items()}
    finite = all(torch.isfinite(p).all() for mod in (wm, policy, critic) for p in mod.parameters())
    rep["updates"] = {"n": a.updates, "batch": a.batch, "horizon": a.horizon, "logs": logs, "new_scan_columns_mean_abs": {"before": before, "after": after}, "all_parameters_finite": bool(finite)}
    print("new scan columns, mean |weight| before -> after:", {k: (before[k], round(after[k], 7)) for k in before}, "| all finite:", finite)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("->", out / "report.json")


if __name__ == "__main__":
    main()
