"""Summarise the terrain part of one recovered session (no GPU): per-tile evaluation tables, the two sensor tests,
curriculum levels over the session.

python3 -I tools/offline/terrain_summary.py <session_dir>   -> a table per evaluated checkpoint on stdout, analysis/terrain_summary.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

LEVEL = re.compile(r"^it (\d+) .*?updates (\d+) .*?train_ret ([-\d.na]+).*?\| level ([-\d.na]+) up ([-\d.na]+) down ([-\d.na]+)")
SPEED = re.compile(r"^it (\d+) \| [\d.]+ h \| ([\d.]+) it/s .*?t sim/wm/pi ([\d.]+)/([\d.]+)/([\d.]+)s")


def table(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["type"], {})[r["riser_cm"]] = r[key]
    return out


def fmt(name: str, rep: dict) -> str:
    rows = rep["tiles"]
    risers = sorted({r["riser_cm"] for r in rows})
    lines = [f"  {name}: scan {rep.get('scan_fed', '?')}" + (f", relief x{rep['scan_relief']}" if rep.get("scan_relief", 1.0) != 1.0 else "") + f", {rows[0]['windows']} windows per tile"]
    lines.append("    " + f"{'':16s}" + "".join(f"{r:>7.1f}cm" for r in risers) + "    (not fallen / completed with the climb done)")
    nf, cc = table(rows, "not_fallen"), table(rows, "completed_with_climb")
    for t in nf:
        lines.append("    " + f"{t:16s}" + "".join(f"  {nf[t].get(r, float('nan')):.2f}/{cc[t].get(r, float('nan')):.2f}" for r in risers))
    return "\n".join(lines)


def main() -> None:
    sd = sys.argv[1].rstrip("/")
    out_dir = os.path.join(sd, "recovered", "out")
    res: dict = {"session": os.path.basename(sd), "evals": {}}
    print(f"{os.path.basename(sd)}: terrain")
    for p in sorted(glob.glob(os.path.join(out_dir, "eval", "terrain_*", "results.json"))):
        name = os.path.basename(os.path.dirname(p))
        rep = json.load(open(p))
        res["evals"][name] = {"meta": rep.get("checkpoint_meta"), "scan_fed": rep.get("scan_fed"), "scan_relief": rep.get("scan_relief"), "tiles": rep["tiles"]}
        stairs = [r for r in rep["tiles"] if "stairs" in r["type"]]
        if stairs:
            res["evals"][name]["stairs_mean"] = {k: sum(r[k] for r in stairs) / len(stairs) for k in ("not_fallen", "completed_with_climb", "norm_return_fixed_denominator")}
        print(fmt(name, rep))
    log = os.path.join(out_dir, "train.log")
    if os.path.exists(log):
        lv, sp = [], []
        for line in open(log, errors="replace"):
            m = LEVEL.match(line)
            if m:
                lv.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6)))
            m = SPEED.match(line)
            if m:
                sp.append((float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
        if lv:
            pick = lv[:: max(len(lv) // 8, 1)] + [lv[-1]]
            print("  curriculum (iteration, updates, train return, mean level, up share, down share):")
            for r in pick:
                print("    it %6d upd %6d ret %s level %s up %s down %s" % r)
            res["curriculum"] = [dict(zip(("iteration", "updates", "train_return", "level_mean", "up_share", "down_share"), r)) for r in pick]
        if len(sp) > 10:
            tail = sp[len(sp) // 2 :]
            mean = lambda i: sum(x[i] for x in tail) / len(tail)  # noqa: E731
            res["speed"] = {"its": mean(0), "sim_s_per_50": mean(1), "wm_s_per_50": mean(2), "policy_s_per_50": mean(3)}
            print("  speed (second half): %.2f it/s; per 50 iterations: sim %.1f s, world model %.1f s, policy %.1f s" % (mean(0), mean(1), mean(2), mean(3)))
    os.makedirs(os.path.join(sd, "analysis"), exist_ok=True)
    json.dump(res, open(os.path.join(sd, "analysis", "terrain_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
