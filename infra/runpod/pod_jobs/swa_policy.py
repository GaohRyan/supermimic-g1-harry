"""Uniform average of the policy weights of several checkpoints (SWA-style) -> one policy-only checkpoint.

  python swa_policy.py --out <swa.pt> [--last K] ckpt_0002000.pt ... ckpt_latest.pt

Checkpoints are given in training order; duplicates (same iteration) are dropped; with --last K only the newest K are
averaged. Meta (config, iteration, num_updates) is copied from the newest one. Only floating-point policy tensors are
averaged; everything else is copied. Pure torch, runs in the Pod and on a laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import torch


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--last", type=int, default=0, help="average only the newest K checkpoints (0 = all)")
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    seen, items = set(), []
    for p in a.ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=True)
        it = ck.get("iteration")
        if it in seen:
            continue
        seen.add(it)
        items.append((p, ck))
    if a.last:
        items = items[-a.last:]
    assert len(items) >= 2, f"need at least two distinct checkpoints, got {len(items)}"
    last = items[-1][1]
    keys = list(last["policy"].keys())
    avg = {}
    for k in keys:
        ts = [ck["policy"][k] for _, ck in items]
        assert all(t.shape == ts[0].shape for t in ts), k
        avg[k] = torch.stack([t.double() for t in ts]).mean(0).to(ts[0].dtype) if ts[0].is_floating_point() else ts[-1].clone()
    src = [{"file": pathlib.Path(p).name, "iteration": ck.get("iteration"), "num_updates": ck.get("num_updates"), "sha256": sha256(p)} for p, ck in items]
    out = {"config": last["config"], "policy": avg, "iteration": last.get("iteration"), "num_updates": last.get("num_updates")}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    meta = {"kind": "uniform policy-weight average (SWA-style); meta copied from the newest checkpoint", "sources": src, "sha256": sha256(a.out)}
    pathlib.Path(a.out + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"out": a.out, "n": len(items), "iterations": [s["iteration"] for s in src], "sha256": meta["sha256"][:16]}))


if __name__ == "__main__":
    main()
