"""Go / no-go check of a recovered terrain smoke session (infra/runpod/pod_jobs/terrain_smoke.sh) before any terrain training.

  python3 -I tools/terrain/smoke_gate.py <smoke session dir> [--source-session <session the warm-start checkpoint came from>]

Prints one line per check (PASS / FAIL / INFO with the observed value and the threshold), writes analysis/smoke_gate.json and
exits 0 only if every blocking check passes. Thresholds are written here, before the run they judge.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

MIN_ITS = 1.5  # it/s with the terrain; the flat-ground lines run at 2.7-3.1. Below this the cost per update needs a decision by a person
FLAT_RETURN_TOL = 0.03  # warm start on the flat plane vs the source checkpoint's own 10 s window score
FLAT_COMPLETED_MIN = 0.97
PARITY_SHARE_MAX = 2e-3
IT = re.compile(r"^it (\d+) \| [\d.]+ h \| ([\d.]+) it/s \| updates (\d+) \| train_ret ([-\d.na]+).*?nse1 ([-\d.na]+).*?imag_ret ([-\d.na]+) surv ([-\d.na]+) \| t sim/wm/pi ([\d.]+)/([\d.]+)/([\d.]+)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--source-session", default=None)
    a = ap.parse_args()
    out = os.path.join(a.session.rstrip("/"), "recovered", "out")
    checks: list[dict] = []

    def add(name, ok, value, need, blocking=True):
        checks.append({"check": name, "ok": None if ok is None else bool(ok), "value": value, "need": need, "blocking": blocking})

    def read(path):
        try:
            return open(os.path.join(out, path), errors="replace").read()
        except OSError:
            return None

    # ---- exit codes of every step
    ex = read("exits.txt") or ""
    codes = dict(re.findall(r"(\w+_exit)=(\d+)", ex))
    for k in ("check_anchor_obs_exit", "check_terrain_exit", "init_exit", "train_exit", "resume_exit"):
        add(k, codes.get(k) == "0", codes.get(k), "0")
    evals = re.findall(r"eval (\S+) exit=(\d+)", ex)
    add("evaluation exit codes", bool(evals) and all(c == "0" for _, c in evals), {n: c for n, c in evals}, "all 0, at least one")
    t = read("tests.txt") or ""
    add("unit tests", " passed" in t and "failed" not in t, t.strip()[:80], "passed, none failed")

    # ---- numeric check on the real stack
    try:
        ct = json.load(open(os.path.join(out, "check_terrain.json")))
    except (OSError, ValueError):
        ct = None
    if ct is None:
        add("check_terrain.json", False, "missing", "present")
    else:
        par = ct.get("parity", {})
        worst = max((v.get("share_above_tol", 1.0) for v in par.values()), default=1.0)
        add("reward / termination parity (share of samples above tolerance)", worst <= PARITY_SHARE_MAX, {k: [round(v["max"], 6), v["share_above_tol"]] for k, v in par.items()}, f"<= {PARITY_SHARE_MAX}")
        hm = ct.get("height_map_vs_ray", {})
        add("height map vs ray casts", hm.get("share_above_2cm", 1.0) < 0.03 and hm.get("median_m", 1.0) < 1e-3, hm, "median < 1 mm, < 3 % of points off by > 2 cm")
        rs = ct.get("reset", {})
        add("flat tiles: reset lift and state origin z", rs.get("flat_tile_spawn_lift_maxabs", 1.0) < 1e-6 and rs.get("state_origin_z_maxabs") == 0.0, {k: rs.get(k) for k in ("flat_tile_spawn_lift_maxabs", "state_origin_z_maxabs", "terrain_tile_hover_mean_m")}, "< 1e-6 m and 0")
        add("buffer usage (information)", None, ct.get("buffer_usage"), "nefc max well below njmax 600", blocking=False)
        add("simulation step of the check env (information)", None, {"step_s": ct.get("step_s"), "model": ct.get("model")}, "-", blocking=False)
    log = read("check_terrain.log")
    n_over = None if log is None else len(re.findall(r"overflow", log, flags=re.I))
    add("nconmax / njmax overflow messages in the terrain check", n_over == 0, n_over, "0")

    # ---- the Trainer on the terrain
    tl = read("train.log") or ""
    its = [IT.match(line) for line in tl.splitlines()]
    its = [m for m in its if m]
    if not its:
        add("training log", False, "no iteration lines", "iterations logged")
    else:
        upd = [m for m in its if int(m.group(3)) > 0]
        speed = sorted(float(m.group(2)) for m in upd[1:]) if len(upd) > 2 else []
        med = speed[len(speed) // 2] if speed else None
        add("training speed with updates running (median it/s)", med is not None and med >= MIN_ITS, med, f">= {MIN_ITS}")
        last = its[-1]
        vals = {"iteration": int(last.group(1)), "updates": int(last.group(3)), "wm_nse1": last.group(5), "imag_ret": last.group(6), "surv": last.group(7), "sim/wm/pi s per 50": [last.group(8), last.group(9), last.group(10)]}
        finite = all("nan" not in str(vals[k]) for k in ("wm_nse1", "imag_ret", "surv")) and int(last.group(3)) > 0
        add("last training line finite, updates ran", finite, vals, "no nan, updates > 0")
    add("no traceback in the training / resume / init logs", not any("Traceback" in (read(f) or "") for f in ("train.log", "resume.log", "init_train.log")), None, "none")
    rl = read("resume.log") or ""
    add("resume restored the run and the terrain levels", "resumed from" in rl and "terrain levels not restored" not in rl, None, "resumed, levels restored")

    # ---- the warm start itself
    try:
        win = json.load(open(os.path.join(out, "eval", "win_init", "results.json")))["summary"]
    except (OSError, ValueError, KeyError):
        win = None
    ref = None
    if a.source_session:
        cands = sorted(glob.glob(os.path.join(a.source_session.rstrip("/"), "recovered", "out", "eval", "win_*ckpt_latest", "results.json")))
        if cands:
            ref = json.load(open(cands[-1]))["summary"]
    if win is None:
        add("flat-ground windows of the warm start", False, "missing", "present")
    else:
        share = win["completed"] / max(win["clips"], 1)
        ok = share >= FLAT_COMPLETED_MIN and (ref is None or win["macro_norm_return_fixed_denominator"] >= ref["macro_norm_return_fixed_denominator"] - FLAT_RETURN_TOL)
        add("flat-ground windows of the warm start vs its source checkpoint", ok, {"warm_start": [win["completed"], win["clips"], round(win["macro_norm_return_fixed_denominator"], 4)],
                                                                                      "source": None if ref is None else [ref["completed"], ref["clips"], round(ref["macro_norm_return_fixed_denominator"], 4)]},
            f"completed >= {FLAT_COMPLETED_MIN}, return within {FLAT_RETURN_TOL} of the source")
    for name, blocking in (("terrain_init", True), ("terrain_init_heightfields", False), ("terrain_smoke", False)):
        try:
            tiles = json.load(open(os.path.join(out, "eval", name, "results.json")))["tiles"]
        except (OSError, ValueError, KeyError):
            add(f"terrain evaluation {name}", False if blocking else None, "missing", "present", blocking=blocking)
            continue
        flat = [r["not_fallen"] for r in tiles if r["type"] == "flat"]
        easy = [r["not_fallen"] for r in tiles if "stairs" in r["type"] and r["level"] == 0]
        summ = {r["type"] + "@" + str(r["riser_cm"]): r["not_fallen"] for r in tiles}
        if name == "terrain_init":
            add("warm start on terrain: flat tiles and the 5 cm stairs", bool(flat) and min(flat) >= 0.95 and bool(easy) and sum(easy) / len(easy) >= 0.5, summ, "flat tiles >= 0.95 not fallen, 5 cm stairs >= 0.5 on average")
        else:
            add(f"terrain evaluation {name} (information)", None, summ, "-", blocking=False)

    ok = all(c["ok"] for c in checks if c["blocking"])
    for c in checks:
        tag = "INFO" if c["ok"] is None else ("PASS" if c["ok"] else "FAIL")
        print(f"{tag:4s} {c['check']}: {json.dumps(c['value'], default=str)[:420]}   [need: {c['need']}]")
    print("GATE", "PASS" if ok else "FAIL")
    os.makedirs(os.path.join(a.session, "analysis"), exist_ok=True)
    json.dump({"ok": ok, "checks": checks}, open(os.path.join(a.session, "analysis", "smoke_gate.json"), "w"), indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
