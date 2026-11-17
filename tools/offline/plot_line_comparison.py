"""Compare training lines on LAFAN40: training return and whole-clip evaluation return against policy updates.

Reads, for every session of every line, the session summary written by the session driver
(`artifacts/infra_sessions/<session>/analysis/session_summary.json`) and the TensorBoard file recovered from the Pod.
TensorBoard steps are iterations; they are mapped to policy updates linearly within each session, between the parent's
update count at the first iteration and the session's own count at the last one.

Usage (from the repository root, with the offline environment):
    .venv-sim/bin/python tools/offline/plot_line_comparison.py --out docs/assets/opt-lines-lafan40.png
"""
import argparse, glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from train_dashboard import read_tb  # noqa: E402

ROOT = "artifacts/infra_sessions"
# label, colour, line style, sessions in order (the accepted session of each index; relaunches excluded)
LINES = [
    ("Official contract (no randomization)", "#888888", "--", [
        "cuboi-g1-g1f-lafan40-official-seed0-s1-20261111b", "cuboi-g1-g1f-lafan40-official-seed0-s2-20261111b",
        "cuboi-g1-g1f-lafan40-official-seed0-s3-20261111b"]),
    ("Opt 1: + sim2real randomization", "#c9a227", "-", [
        "cuboi-g1-o1d-lafan40-sim2realdr-seed0-s1-20261112", "cuboi-g1-o1d-lafan40-sim2realdr-seed0-s2-20261112",
        "cuboi-g1-o1d-lafan40-sim2realdr-seed0-s3-20261112-r1"]),
    ("  + reset-sampler fix", "#8fb8de", "-", [
        "cuboi-g1-o2e-lafan40-samplernorm-seed0-s1-20261112b", "cuboi-g1-o2e-lafan40-samplernorm-seed0-s2-20261112b-r2",
        "cuboi-g1-o2e-lafan40-samplernorm-seed0-s3-20261112b"]),
    ("Opt 2: + world-model hard mining", "#2a6fb0", "-", [
        "cuboi-g1-o3g-lafan40-wmhard-seed0-s1-20261113b", "cuboi-g1-o3g-lafan40-wmhard-seed0-s2-20261113b-r1",
        "cuboi-g1-o3g-lafan40-wmhard-seed0-s3-20261113b"]),
    ("Opt 3: + Muon", "#c0392b", "-", [
        "cuboi-g1-o4m-lafan40-muon-seed0-s1-20261115b", "cuboi-g1-o4m-lafan40-muon-seed0-s2-20261115c",
        "cuboi-g1-o4m-lafan40-muon-seed0-s3-20261115c"]),
]
TAG = "train/episode_norm_return"


def smooth(y, k=25):
    if len(y) < k:
        return np.asarray(y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


def line_data(sessions):
    train_x, train_y, evals = [], [], []
    prev_updates = 0
    for s in sessions:
        summ = json.load(open(f"{ROOT}/{s}/analysis/session_summary.json"))
        tr = summ["train"]
        tb = glob.glob(f"{ROOT}/{s}/recovered/out/train_logs/*/tb/events.out.tfevents.*")
        if tb:
            pts = sorted(read_tb(tb[0]).get(TAG, []))
            it0, it1, u1 = tr["first_iteration"], tr["last_iteration"], tr["updates"]
            for it, v in pts:
                train_x.append(prev_updates + (u1 - prev_updates) * (it - it0) / max(it1 - it0, 1))
                train_y.append(v)
        for e in summ.get("fullclip") or []:
            evals.append((e["updates"], e["macro"], e["completed"], e["kind"]))
        prev_updates = tr["updates"]
    return np.array(train_x), np.array(train_y), evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/opt-lines-lafan40.png")
    args = ap.parse_args()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)
    for label, colour, ls, sessions in LINES:
        x, y, ev = line_data(sessions)
        if len(x):
            ys = smooth(y)
            a.plot(x[len(x) - len(ys):], ys, color=colour, ls=ls, lw=1.4, label=label)
        swa = sorted((u, m) for u, m, c, k in ev if k == "swa")
        ck = sorted((u, m) for u, m, c, k in ev if k == "ckpt")
        if swa:
            b.plot(*zip(*swa), color=colour, ls=ls, lw=1.6, marker="o", ms=4, label=label)
        if ck:
            b.scatter(*zip(*ck), color=colour, s=10, alpha=0.45)
    a.set(title="Training return (episodes in the simulator, smoothed)", xlabel="policy updates",
          ylabel="normalized return")
    b.set(title="Whole-clip evaluation, 40 clips (line: weight average; dots: single checkpoints)",
          xlabel="policy updates", ylabel="normalized return", ylim=(0.2, 0.9))
    b.title.set_fontsize(9); a.title.set_fontsize(9)
    for ax in (a, b):
        ax.grid(alpha=0.25)
    b.legend(fontsize=7, loc="lower right")
    fig.text(0.01, 0.005, "LAFAN1, seed 0. Opt lines train under randomization; the official-contract line does not, "
             "and from the sampler fix on, resets favour harder segments; compare training return within similar setups only.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
