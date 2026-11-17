"""Side-by-side table of training lines at their session ends (no GPU).

  python3 -I tools/offline/compare_lines.py --out <json> NAME=sessdir1,sessdir2,... [NAME=...]

For every session of every line: iteration, updates, single final checkpoint and (when present) the 4-checkpoint weight
average, on window10s and full clip, with both normalisations (planned steps = main metric; survived steps = upstream).
Reads <session>/analysis/session_summary.json (rebuilt on every call).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def last(rows, kind):
    rows = [r for r in rows if r.get("kind", "ckpt") == kind and "init" not in r["dir"] and "robust" not in (r.get("protocol_id") or "")]
    return rows[-1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("lines", nargs="+")
    a = ap.parse_args()
    out = {}
    for spec in a.lines:
        name, dirs = spec.split("=", 1)
        rows = []
        for d in dirs.split(","):
            sd = Path(d)
            subprocess.run([sys.executable, "-I", str(ROOT / "tools/offline/session_summary.py"), str(sd)], capture_output=True)
            s = json.loads((sd / "analysis/session_summary.json").read_text())
            t = s.get("train", {})
            row = {"session": s["session"], "iteration": t.get("last_iteration"), "updates": t.get("updates"), "it_per_s": t.get("it_per_s_median"), "wm_nse8": t.get("wm_nse8_last")}
            for proto, key, n in (("window10s", "win", "windows"), ("fullclip", "full", "clips")):
                for kind in ("ckpt", "swa"):
                    r = last(s.get(proto, []), kind)
                    if r:
                        row[f"{key}_{kind}"] = {"done": f"{r['completed']}/{r[n]}", "macro": round(r["macro"], 3), "macro_upstream_style": None if r.get("macro_upstream_style") is None else round(r["macro_upstream_style"], 3),
                                                **({"yaw_deg": round(r["end_yaw_deg"], 1), "xy_m": round(r["end_xy_m"], 2)} if "end_yaw_deg" in r else {}),
                                                **({"failed": [x[0] for x in r["failed"]]} if "failed" in r else {})}
            rows.append(row)
        out[name] = rows
        print(f"\n== {name}")
        print("| it | upd | win final | win avg4 | full final | full avg4 | full final (upstream) | yaw° |")
        print("|---:|---:|---|---|---|---|---:|---:|")
        f = lambda x: f"{x['done']} · {x['macro']}" if x else "—"
        for r in rows:
            print(f"| {r['iteration']} | {r['updates']} | {f(r.get('win_ckpt'))} | {f(r.get('win_swa'))} | {f(r.get('full_ckpt'))} | {f(r.get('full_swa'))} | {(r.get('full_ckpt') or {}).get('macro_upstream_style', '—')} | {(r.get('win_ckpt') or {}).get('yaw_deg', '—')} |")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
