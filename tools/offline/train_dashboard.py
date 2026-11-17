"""Training dashboard without W&B: TensorBoard scalars + per-checkpoint eval results -> one PNG + one JSON.

python3 -I tools/offline/train_dashboard.py <out.png> <session_out_dir> [<session_out_dir> ...]

Each session_out_dir is a recovered (or live-synced) `out/` directory containing
  train_logs/<run>/tb/events.out.tfevents.*        TensorBoard scalars (read with a tiny built-in parser, no tensorflow)
  eval/win_logs_<run>_ckpt_*/results.json           window10s-nominal-v2 per checkpoint
  eval/full_logs_<run>_ckpt_*/results.json          fullclip-nominal-v2 per checkpoint
Several sessions (warm restarts) are concatenated on the iteration axis. Falls back to train.log if no TB file.
"""
import glob, json, os, re, struct, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i = 0
    while i < len(b):
        k, i = _varint(b, i); f, t = k >> 3, k & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 1:
            v = b[i:i + 8]; i += 8
        elif t == 2:
            n, i = _varint(b, i); v = b[i:i + n]; i += n
        elif t == 5:
            v = b[i:i + 4]; i += 4
        else:
            raise ValueError(t)
        yield f, t, v


def read_tb(path):
    """{tag: [(step, value)]} from a TensorBoard event file (simple_value or float tensor scalars)."""
    out = defaultdict(list)
    data = open(path, "rb").read(); i = 0
    while i + 12 <= len(data):
        n = struct.unpack("<Q", data[i:i + 8])[0]; rec = data[i + 12:i + 12 + n]; i += 12 + n + 4
        step = 0; summ = None
        for f, t, v in _fields(rec):
            if f == 2 and t == 0:
                step = v
            elif f == 5 and t == 2:
                summ = v
        if summ is None:
            continue
        for f, t, v in _fields(summ):
            if f != 1:
                continue
            tag = None; val = None
            for f2, t2, v2 in _fields(v):
                if f2 == 1 and t2 == 2:
                    tag = v2.decode(errors="replace")
                elif f2 == 2 and t2 == 5:
                    val = struct.unpack("<f", v2)[0]
                elif f2 == 8 and t2 == 2:  # TensorProto: float_val (5) or tensor_content (4)
                    for f3, t3, v3 in _fields(v2):
                        if f3 == 5 and t3 == 5:
                            val = struct.unpack("<f", v3)[0]
                        elif f3 == 5 and t3 == 2 and len(v3) >= 4:
                            val = struct.unpack("<f", v3[:4])[0]
                        elif f3 == 4 and t3 == 2 and len(v3) == 4:
                            val = struct.unpack("<f", v3)[0]
            if tag is not None and val is not None:
                out[tag].append((step, val))
    return out


LOG = re.compile(r"^it (\d+) \| ([\d.]+) h \| ([\d.]+) it/s \| updates (\d+) \| train_ret ([-\d.]+).*?nse1 ([\d.]+) nse8 ([\d.]+) \| imag_ret ([-\d.]+) surv ([\d.]+)")


def read_log(path):
    out = defaultdict(list)
    for line in open(path, errors="replace"):
        m = LOG.match(line)
        if m:
            it = int(m[1])
            for k, j in (("log/train_ret", 5), ("log/wm_nse1", 6), ("log/wm_nse8", 7), ("log/imag_ret", 8), ("log/imag_survival", 9), ("log/updates", 4), ("log/it_per_s", 3)):
                out[k].append((it, float(m[j])))
    return out


def read_evals(out_dir, kind):
    rows = []
    for f in glob.glob(os.path.join(out_dir, "eval", f"{kind}_logs_*", "results.json")):
        r = json.load(open(f)); it = r["checkpoint_meta"]["iteration"]
        if "full40" in f or it is None:
            continue
        per = defaultdict(list)
        for c in r["clips"]:
            per[c["clip"].split("@")[0]].append(c)
        rows.append({"iteration": it, "updates": r["checkpoint_meta"]["num_updates"], "macro": r["summary"]["macro_norm_return_fixed_denominator"],
                     "completed": r["summary"]["completed"], "n": r["summary"]["clips"], "phys": r["summary"]["physical_failures"],
                     "per_clip": {k: float(np.mean([x["norm_return_fixed_denominator"] for x in v])) for k, v in per.items()},
                     "per_clip_done": {k: sum(x["completed"] for x in v) / len(v) for k, v in per.items()}})
    return sorted(rows, key=lambda r: r["iteration"])


def smooth(y, k=9):
    if len(y) < k:
        return np.asarray(y)
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def main(out_png, *dirs):
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    scal = defaultdict(list); win, full = [], []; bounds = []
    for d in dirs:
        tbs = glob.glob(os.path.join(d, "train_logs", "*", "tb", "events.out.tfevents.*"))
        tbs = [t for t in tbs if not os.path.basename(os.path.dirname(os.path.dirname(t))).startswith("init")]
        s = defaultdict(list)
        for t in tbs:
            for k, v in read_tb(t).items():
                s[k] += v
        for k, v in read_log(os.path.join(d, "train.log")).items():
            s[k] += v
        for k, v in s.items():
            scal[k] += sorted(v)
        if s.get("log/train_ret"):
            bounds.append(s["log/train_ret"][0][0])
        win += read_evals(d, "win"); full += read_evals(d, "full")
    win = sorted({r["iteration"]: r for r in win}.values(), key=lambda r: r["iteration"])
    full = sorted({r["iteration"]: r for r in full}.values(), key=lambda r: r["iteration"])

    tb_tags = sorted(k for k in scal if not k.startswith("log/"))
    rew_tags = [k for k in tb_tags if "imag_r/" in k]
    loss_tags = [k for k in ("wm/loss", "policy/critic_loss", "policy/policy_grad_norm", "wm/grad_norm") if k in scal]
    clips = sorted({c for r in win for c in r["per_clip"]})

    fig, ax = plt.subplots(3, 3, figsize=(18, 13))
    a = ax[0, 0]
    x, y = zip(*scal["log/train_ret"]) if scal["log/train_ret"] else ([], [])
    a.plot(x, y, color="#999", lw=0.6, alpha=0.6); a.plot(x, smooth(np.array(y)), color="#222", lw=1.5, label="train_ret (normalized training-episode return)")
    if scal["log/imag_ret"]:
        xi, yi = zip(*scal["log/imag_ret"]); a2 = a.twinx(); a2.plot(xi, smooth(np.array(yi)), color="#00798c", lw=1, label="imag_ret (imagined return, right axis)"); a2.set_ylabel("imag_ret")
    a.set(title="Training return", xlabel="iteration"); a.legend(loc="upper left", fontsize=8)
    a = ax[0, 1]
    for k, c in (("log/wm_nse1", "#edae49"), ("log/wm_nse8", "#d1495b")):
        if scal[k]:
            xx, yy = zip(*scal[k]); a.semilogy(xx, smooth(np.array(yy)), color=c, label=k[4:])
    a.set(title="world-model error (nse, log scale)", xlabel="iteration"); a.legend(fontsize=8)
    a = ax[0, 2]
    for k in rew_tags:
        xx, yy = zip(*sorted(scal[k])); a.plot(xx, smooth(np.array(yy)), lw=1, label=k.split("/")[-1])
    a.set(title="reward terms (per-step mean in imagined rollouts, TensorBoard imag_r/*)", xlabel="iteration"); a.legend(fontsize=6, ncol=2)
    a = ax[2, 0]
    for k in loss_tags:
        xx, yy = zip(*sorted(scal[k])); yy = np.array(yy); a.semilogy(xx, smooth(np.abs(yy) + 1e-12), lw=1, label=k)
    a.set(title="loss / gradient norm (log scale)", xlabel="iteration"); a.legend(fontsize=7)
    a = ax[2, 1]
    for k, c in (("train/episode_len", "#222"),):
        if scal[k]:
            xx, yy = zip(*sorted(scal[k])); a.plot(xx, smooth(np.array(yy)), color=c, lw=1, label="training episode length (steps, max 500)")
    if scal["policy/imag_survival_H"]:
        xx, yy = zip(*sorted(scal["policy/imag_survival_H"])); a2 = a.twinx(); a2.plot(xx, smooth(np.array(yy)), color="#00798c", lw=1, label="imagined survival (right axis)")
    a.set(title="Survival", xlabel="iteration"); a.legend(loc="lower right", fontsize=8)
    a = ax[2, 2]
    for k, c in (("sampler/entropy", "#edae49"), ("sampler/bin_return_mean", "#d1495b"), ("policy/action_abs_mean", "#555")):
        if scal[k]:
            xx, yy = zip(*sorted(scal[k])); a.plot(xx, smooth(np.array(yy)), color=c, lw=1, label=k)
    a.set(title="sampler / action magnitude", xlabel="iteration"); a.legend(fontsize=7)
    a = ax[1, 0]
    if win:
        it = [r["iteration"] for r in win]
        a.plot(it, [r["macro"] for r in win], "o-", color="#222", lw=2, label="macro")
        for c in clips:
            a.plot(it, [r["per_clip"].get(c, np.nan) for r in win], ".--", lw=1, label=c)
        a.axhline(0.80, color="gray", ls=":", lw=1); a.text(it[0], 0.805, "threshold 0.80", fontsize=8, color="gray")
    a.set(title="eval: 10 s window return (window10s-nominal-v2)", xlabel="iteration", ylim=(0, 1)); a.legend(fontsize=7)
    a = ax[1, 1]
    if win:
        a.plot(it, [r["completed"] / r["n"] for r in win], "o-", color="#00798c", label="window completion rate")
    if full:
        fi = [r["iteration"] for r in full]
        a.plot(fi, [r["completed"] / r["n"] for r in full], "s-", color="#d1495b", label="whole-clip completion rate")
        a.plot(fi, [r["macro"] for r in full], "^--", color="#edae49", label="whole-clip macro return")
    a.set(title="eval: completion / whole-clip return (fullclip-nominal-v2)", xlabel="iteration", ylim=(0, 1.05)); a.legend(fontsize=8)
    a = ax[1, 2]
    if scal["log/it_per_s"]:
        xx, yy = zip(*scal["log/it_per_s"]); a.plot(xx, yy, color="#555", lw=0.8, label="it/s")
    if scal["log/updates"]:
        xu, yu = zip(*scal["log/updates"]); a2 = a.twinx(); a2.plot(xu, yu, color="#d1495b", lw=1, label="updates (right axis)"); a2.set_ylabel("updates")
    a.set(title="throughput and cumulative updates", xlabel="iteration"); a.legend(loc="upper left", fontsize=8)
    for a in ax.flat:
        for b in bounds[1:]:
            a.axvline(b, color="#aaa", ls="--", lw=0.8)
    fig.suptitle("Training monitor: " + " + ".join(os.path.basename(os.path.dirname(os.path.dirname(d.rstrip("/")))) for d in dirs) + " (dashed = session boundary / warm restart)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    json.dump({"window10s": win, "fullclip": full, "tb_tags": tb_tags}, open(os.path.splitext(out_png)[0] + ".json", "w"), indent=1)
    print(out_png, "tb tags:", len(tb_tags), "reward tags:", rew_tags[:4])


if __name__ == "__main__":
    main(*sys.argv[1:])
