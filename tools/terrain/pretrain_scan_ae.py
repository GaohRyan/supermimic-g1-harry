"""Step 0 of the terrain plan: pre-train the height-scan autoencoder on the level-1 terrain and report reconstruction
error, in particular at step edges. CPU only, about two minutes.

  source tools/offline/env.sh
  PYTHONPATH=<variant>/src:tools/offline/.mjlab_stub:tools/terrain .venv-sim/bin/python tools/terrain/pretrain_scan_ae.py \
      --out artifacts/projects/cuboi-g1/terrain/scan_ae-v1 [--steps 20000] [--noise 0.03]

Writes <out>/scan_ae.pt (encoder, decoder, config, metadata incl. the terrain signature), report.json, reconstruction.png.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import mjlab_terrain_local

mjlab_terrain_local.install()

from supermimic.terrain.terrain_cfg import TILE_SIZE, TerrainArgs, make_terrain_generator_cfg  # noqa: E402
from supermimic.terrain.heightmap import ScanPattern, TerrainMap, terrain_signature  # noqa: E402
from supermimic.terrain.scan_ae import ScanAE, ScanAECfg  # noqa: E402

TYPES = ("flat", "stairs_up", "stairs_down", "stairs_random", "slope", "rough")


def sample_scans(tm: TerrainMap, origins: torch.Tensor, pat: ScanPattern, n: int, gen: torch.Generator, level=None, col=None):
    """Scans at random poses: uniform over tiles (or one tile), position inside the tile, yaw, and root height above ground."""
    L, C, _ = origins.shape
    lev = torch.randint(0, L, (n,), generator=gen) if level is None else torch.full((n,), level)
    cc = torch.randint(0, C, (n,), generator=gen) if col is None else torch.full((n,), col)
    o = origins[lev, cc]
    xy = o[:, :2] + (torch.rand(n, 2, generator=gen) * 2 - 1) * torch.tensor([TILE_SIZE[0] / 2 - 0.5, TILE_SIZE[1] / 2 - 0.5])
    yaw = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
    standing = torch.rand(n, generator=gen) < 0.8
    hgt = torch.where(standing, 0.6 + 0.35 * torch.rand(n, generator=gen), 0.15 + 0.45 * torch.rand(n, generator=gen))
    root = torch.cat([xy, (tm.height(xy) + hgt)[:, None]], -1)
    return tm.scan(root, yaw, pat), lev, cc


def edge_mask(scan: torch.Tensor, shape, thr: float = 0.03) -> torch.Tensor:
    """Cells that differ from a 4-neighbour by more than thr (a step edge passes next to them)."""
    g = scan.reshape(-1, *shape)
    m = torch.zeros_like(g, dtype=torch.bool)
    dx = (g[:, 1:, :] - g[:, :-1, :]).abs() > thr
    dy = (g[:, :, 1:] - g[:, :, :-1]).abs() > thr
    m[:, 1:, :] |= dx
    m[:, :-1, :] |= dx
    m[:, :, 1:] |= dy
    m[:, :, :-1] |= dy
    return m.reshape(scan.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--noise", type=float, default=0.03, help="half width of the uniform input noise (denoising target = clean scan)")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resolution", type=float, default=0.025)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    targs = TerrainArgs()
    model, origins_np = mjlab_terrain_local.build_terrain_model(make_terrain_generator_cfg(targs))
    tm = TerrainMap.from_model(model, a.resolution)
    sig = terrain_signature(model, a.resolution)
    origins = torch.as_tensor(origins_np, dtype=torch.float32)
    pat = ScanPattern.grid()
    gen = torch.Generator().manual_seed(1)
    ae = ScanAE(ScanAECfg(latent_dim=a.latent, hidden=tuple(a.hidden)))
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)
    t0 = time.time()
    for it in range(a.steps):
        clean, _, _ = sample_scans(tm, origins, pat, a.batch, gen)
        noisy = clean + (torch.rand(clean.shape, generator=gen) * 2 - 1) * a.noise
        rec, _ = ae(noisy)
        loss = ((rec - clean) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 2000 == 0 or it == a.steps - 1:
            print(f"step {it:6d} rmse {loss.item() ** 0.5 * 100:.2f} cm  ({time.time() - t0:.0f}s)", flush=True)
    ae.eval()
    vgen = torch.Generator().manual_seed(12345)
    report = {"terrain_signature": sig, "terrain_args": targs.__dict__, "noise": a.noise, "steps": a.steps, "latent_dim": a.latent, "hidden": list(a.hidden), "by_type": {}, "by_input_noise": {}}
    with torch.no_grad():
        for c, name in enumerate(TYPES):
            row = {}
            for lev in (0, 4, 9):
                clean, _, _ = sample_scans(tm, origins, pat, 8192, vgen, level=lev, col=c)
                rec, _ = ae(clean + (torch.rand(clean.shape, generator=vgen) * 2 - 1) * a.noise)
                e = rec - clean
                em = edge_mask(clean, pat.shape)
                row[f"level{lev}"] = {"rmse_cm": float(e.pow(2).mean().sqrt() * 100), "edge_rmse_cm": float(e[em].pow(2).mean().sqrt() * 100) if em.any() else None,
                                     "edge_share": float(em.float().mean()), "p99_abs_cm": float(e.abs().quantile(0.99) * 100)}
            report["by_type"][name] = row
        clean, _, _ = sample_scans(tm, origins, pat, 32768, vgen)
        for nz in (0.0, a.noise, 0.05, 0.10):
            rec, _ = ae(clean + (torch.rand(clean.shape, generator=vgen) * 2 - 1) * nz)
            report["by_input_noise"][f"{nz:.2f}"] = float((rec - clean).pow(2).mean().sqrt() * 100)
        z = ae.encoder(clean)
        report["latent_std_mean"] = float(z.std(0).mean())
        report["latent_dead_dims"] = int((z.std(0) < 1e-3).sum())
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": ae.encoder.state_dict(), "decoder": ae.decoder.state_dict(), "cfg": {"scan_dim": 187, "latent_dim": a.latent, "hidden": list(ae.cfg.hidden)},
                "meta": {"terrain_signature": sig, "pattern": {"x_range": [-0.8, 0.8], "y_range": [-0.5, 0.5], "resolution": 0.1, "order": "x-major"}, "noise": a.noise,
                         "scan_convention": "z_root - terrain height; network input (scan - 0.8) / 0.25"}}, a.out / "scan_ae.pt")
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["by_type"], indent=1))
    print("by input noise (cm):", report["by_input_noise"], "latent std", report["latent_std_mean"], "dead", report["latent_dead_dims"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 6, figsize=(15, 6.2))
    cases = [(1, 9, "stairs up, 20 cm"), (1, 4, "stairs up, 12 cm"), (1, 0, "stairs up, 5 cm"), (3, 9, "uneven stairs"), (4, 9, "slope"), (5, 9, "rough")]
    g2 = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for k, (c, lev, title) in enumerate(cases):
            clean, _, _ = sample_scans(tm, origins, pat, 64, g2, level=lev, col=c)
            em = edge_mask(clean, pat.shape).float().mean(1)
            i = int(em.argmax()) if c in (1, 3) else 0  # the sample with the most step edges
            x = clean[i : i + 1]
            rec, _ = ae(x + (torch.rand(x.shape, generator=g2) * 2 - 1) * a.noise)
            gt, rc = -(x.reshape(pat.shape) - x.mean()), -(rec.reshape(pat.shape) - x.mean())  # terrain height relative to the mean, up = positive
            vmin, vmax = float(min(gt.min(), rc.min())), float(max(gt.max(), rc.max()))
            for r, (img, lab) in enumerate(((gt, "terrain (truth)"), (rc, "reconstruction"), ((rc - gt).abs(), "|error|"))):
                ax = axes[r, k]
                im = ax.imshow(img.T.numpy(), origin="lower", cmap="viridis" if r < 2 else "magma", vmin=vmin if r < 2 else 0, vmax=vmax if r < 2 else max(0.03, float(img.max())), extent=(-0.85, 0.85, -0.55, 0.55))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=10)
                if k == 0:
                    ax.set_ylabel(lab, fontsize=10)
                if r == 2:
                    ax.set_xlabel(f"max {float(img.max()) * 100:.1f} cm, rmse {float(img.pow(2).mean().sqrt()) * 100:.1f} cm", fontsize=9)
    fig.suptitle(f"Height-scan autoencoder (187 -> {a.latent} -> 187, hidden {a.hidden}): 17 x 11 cells of 10 cm around the pelvis, forward = right", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / "reconstruction.png", dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
