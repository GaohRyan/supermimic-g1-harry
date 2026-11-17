"""Summarise one recovered session directory (no GPU): training stats, window10s and full-clip evals.

python3 -I tools/offline/session_summary.py <session_dir>   -> <session_dir>/analysis/session_summary.json + a short table on stdout
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

LOG = re.compile(r"^it (\d+) \| ([\d.]+) h \| ([\d.]+) it/s \| updates (\d+) \| train_ret ([-\d.]+).*?nse1 ([\d.]+) nse8 ([\d.]+)")


def quat_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def window_geom(traj_path):
    t = np.load(traj_path)
    q, rq, ex = t["qpos"], t["ref_qpos"], t["executed_steps"]
    ye, xe, je = [], [], []
    for i in range(q.shape[0]):
        T = int(ex[i])
        ye.append(np.degrees(abs(wrap(quat_yaw(q[i, T, 3:7]) - quat_yaw(rq[i, T, 3:7])))))
        xe.append(np.linalg.norm(q[i, T, :2] - rq[i, T, :2]))
        je.append(np.abs(wrap(q[i, : T + 1, 7:] - rq[i, : T + 1, 7:])).mean())
    return {"end_yaw_deg": float(np.mean(ye)), "end_xy_m": float(np.mean(xe)), "joint_err_rad": float(np.mean(je))}


def geom(eval_dir: str) -> dict:
    """Yaw / position / joint error summary: from geom.json (written in the Pod) or from trajectories.npz when it was kept."""
    g = os.path.join(eval_dir, "geom.json")
    if os.path.exists(g):
        d = json.load(open(g))
        return {k: d[k] for k in ("end_yaw_deg", "end_xy_m", "joint_err_rad")}
    tp = os.path.join(eval_dir, "trajectories.npz")
    return window_geom(tp) if os.path.exists(tp) else {}


def up_style(r: dict):
    """Mean over clips of return / (0.1 x survived steps): the normalisation of the upstream benchmark (early termination barely penalised)."""
    v = [c.get("norm_return_survived_steps_upstream_style") for c in r["clips"]]
    return float(np.mean(v)) if v and all(x is not None for x in v) else None


def kind(results_path: str) -> str:
    """ckpt = one checkpoint; swa = weight average of several (same iteration meta as the newest source)."""
    return "swa" if "swa" in os.path.basename(os.path.dirname(results_path)) else "ckpt"


def main(sd: str) -> None:
    out = os.path.join(sd, "recovered", "out")
    res: dict = {"session": os.path.basename(sd.rstrip("/"))}
    if os.path.exists(os.path.join(sd, "bundle.json")):
        b = json.load(open(os.path.join(sd, "bundle.json")))
        res["bundle"] = {k: b.get(k) for k in ("config_id", "config_sha256", "run_name", "session_index", "anchor_obs", "seed", "parent_ckpt_sha256", "deviation_from_frozen_argv")}
    tl = os.path.join(out, "train.log")
    if os.path.exists(tl):
        rows = [LOG.match(line) for line in open(tl, errors="replace")]
        rows = [m for m in rows if m]
        if rows:
            its = [float(m[3]) for m in rows[3:]] or [float(rows[-1][3])]
            res["train"] = {"first_iteration": int(rows[0][1]), "last_iteration": int(rows[-1][1]), "updates": int(rows[-1][4]), "hours": float(rows[-1][2]),
                            "it_per_s_median": statistics.median(its), "train_ret_last": float(rows[-1][5]), "wm_nse8_last": float(rows[-1][7])}
    for name in ("host-info.txt", "exits.txt", "tests.txt", "runtime-versions.txt"):
        p = os.path.join(out, name)
        if os.path.exists(p):
            res[name.split(".")[0].replace("-", "_")] = open(p).read().strip().replace("\n", "; ")[:400]
    p = os.path.join(out, "check_anchor_obs.json")
    if os.path.exists(p):
        res["check_anchor_obs"] = json.load(open(p))["summary"]
    wins, fulls = [], []
    for f in sorted(glob.glob(os.path.join(out, "eval", "win_*", "results.json"))):
        r = json.load(open(f))
        by = collections.defaultdict(list)
        for c in r["clips"]:
            by[c["clip"].split("@")[0]].append(c)
        row = {"dir": os.path.basename(os.path.dirname(f)), "kind": kind(f), "protocol_id": r.get("protocol_id"), "iteration": r["checkpoint_meta"]["iteration"], "updates": r["checkpoint_meta"]["num_updates"], "checkpoint_sha256": r["checkpoint_sha256"],
               "completed": r["summary"]["completed"], "windows": r["summary"]["clips"], "macro": r["summary"]["macro_norm_return_fixed_denominator"], "macro_upstream_style": up_style(r),
               "clips_all_windows_done": sum(all(x["completed"] for x in v) for v in by.values()), "n_clips": len(by), "anchor_obs": r["env_args"].get("anchor_obs"),
               "per_clip": {k: {"mean": float(np.mean([x["norm_return_fixed_denominator"] for x in v])), "failed": sum(1 for x in v if not x["completed"])} for k, v in by.items()}}
        row.update(geom(os.path.dirname(f)))
        wins.append(row)
    for f in sorted(glob.glob(os.path.join(out, "eval", "full_*", "results.json"))):
        r = json.load(open(f))
        s = r["summary"]
        fulls.append({"dir": os.path.basename(os.path.dirname(f)), "kind": kind(f), "protocol_id": r.get("protocol_id"), "iteration": r["checkpoint_meta"]["iteration"], "updates": r["checkpoint_meta"]["num_updates"], "checkpoint_sha256": r["checkpoint_sha256"],
                      "completed": s["completed"], "clips": s["clips"], "physical_failures": s["physical_failures"], "macro": s["macro_norm_return_fixed_denominator"], "macro_upstream_style": up_style(r),
                      "time_weighted": s["time_weighted_norm_return_fixed_denominator"], "data_label": r.get("data_label"),
                      "failed": [[c["clip"], c["executed_steps"], c["planned_steps"], c["termination_reason"]] for c in r["clips"] if not c["completed"]],
                      "per_clip": {c["clip"]: c["norm_return_fixed_denominator"] for c in r["clips"]}})
    res["window10s"] = sorted(wins, key=lambda x: (x["iteration"] or 0, x["kind"] == "swa"))
    res["fullclip"] = sorted(fulls, key=lambda x: (x["iteration"] or 0, x["kind"] == "swa"))
    os.makedirs(os.path.join(sd, "analysis"), exist_ok=True)
    json.dump(res, open(os.path.join(sd, "analysis", "session_summary.json"), "w"), indent=1)
    t = res.get("train", {})
    print(f"{res['session']}: updates {t.get('updates')} it/s {t.get('it_per_s_median')} host {res.get('host_info', '')[:70]}")
    for w in res["window10s"]:
        g = f" yaw {w['end_yaw_deg']:.1f} xy {w['end_xy_m']:.2f} joint {w['joint_err_rad']:.3f}" if "end_yaw_deg" in w else ""
        print(f"  win{'*' if w['kind'] == 'swa' else ' '} it {w['iteration']:>6} upd {w['updates']:>6}  {w['completed']}/{w['windows']}  macro {w['macro']:.3f}  clips {w['clips_all_windows_done']}/{w['n_clips']}{g}")
    for f in res["fullclip"]:
        print(f"  full{'*' if f['kind'] == 'swa' else ''} it {f['iteration']:>6} upd {f['updates']:>6}  {f['completed']}/{f['clips']}  phys {f['physical_failures']}  macro {f['macro']:.3f}  [{f['data_label']}] failed {[x[0] for x in f['failed']]}")


if __name__ == "__main__":
    main(sys.argv[1])
