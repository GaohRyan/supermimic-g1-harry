#!/usr/bin/env python3
"""Unattended driver for one training line: bundle -> Pod -> train/eval -> recover -> stop -> ledger -> next session.

  python3 infra/runpod/session_driver.py line --line-id P --base cuboi-g1-n1p-lafan40-proprio-s0 --date 20261110 \
      --config configs/cuboi-g1/<frozen>.json --sessions 4 [--start-index 1] [--parent-ckpt <pt>] \
      [--extra-script <py> ...] [--min-its 2.8] [--tier N1] [--train-extra ...] [--render-clips ...] [--post-cmd ...]

Safety: every Pod is created by session_pod.py (fuse <= 120 min, rate cap), gets a local and an in-Pod watchdog, and is
stopped by this driver as soon as recovery finishes or on any error. The tier ledger is locked while updated; a session is
not created if the tier's remaining budget is below the per-session reserve. Nothing is ever deleted on the Pod or locally.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _site(name: str, default: str = "") -> str:
    """Account-specific setting: environment variable RUNPOD_<NAME>, else .secrets/runpod-site.json in the repository."""
    v = os.environ.get("RUNPOD_" + name.upper())
    if v:
        return v
    try:
        p = Path(__file__).resolve().parents[2] / ".secrets/runpod-site.json"
        return str(json.loads(p.read_text()).get(name, default))
    except (OSError, ValueError, IndexError):
        return default

ROOT = Path(__file__).resolve().parents[2]
SESS = ROOT / "artifacts/infra_sessions"
LEDGER = ROOT / "artifacts/p0/budget-ledger.json"
KEY = Path(_site("ssh_key", "~/.ssh/id_ed25519")).expanduser()
RESERVE = 2.5  # USD held per session until it is settled (--reserve-usd)
MINUTES = 115  # Pod fuse (--minutes); rate cap x minutes must fit in the reservation
GPUS = {"5090": ("NVIDIA GeForce RTX 5090", 1.30), "L40S": ("NVIDIA L40S", 1.20)}


def now() -> datetime:
    return datetime.now(timezone.utc)


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **kw) -> None:
        rec = {"utc": now().isoformat(timespec="seconds"), "event": event, **kw}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)


def run(cmd, timeout=None, check=True, **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
    return p


def ledger_update(fn):
    with open(str(LEDGER) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.loads(LEDGER.read_text())
        out = fn(d)
        d["updated_utc"] = now().isoformat()
        tmp = LEDGER.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        os.replace(tmp, LEDGER)
        return out


def reserve(tier: str, session: str, plan: str) -> bool:
    def fn(d):
        t = d["tiers"][tier]
        spent = sum(x.get("estimate_usd", 0.0) for x in t["sessions"] if "estimate_usd" in x)
        open_res = sum(x.get("reserve_usd", 0.0) for x in t["sessions"] if x.get("open"))
        if spent + open_res + RESERVE > t["cap_usd"]:
            return False
        t["sessions"].append({"session": session, "reserved_utc": now().isoformat(), "reserve_usd": RESERVE, "open": True, "plan": plan, "approved_by": t.get("approved_by")})
        return True
    return ledger_update(fn)


def settle(tier: str, session: str, rec: dict) -> float:
    def fn(d):
        t = d["tiers"][tier]
        for x in t["sessions"]:
            if x.get("session") == session and x.get("open"):
                x["open"] = False
        t["sessions"].append({"session": session, **rec})
        t["estimate_spent_usd"] = round(sum(x.get("estimate_usd", 0.0) for x in t["sessions"] if "estimate_usd" in x), 4)
        return t["estimate_spent_usd"]
    return ledger_update(fn)


class Session:
    def __init__(self, sd: Path, log: Log):
        self.sd, self.log = sd, log
        self.ip = self.port = None

    def pod(self, action: str, *extra, check=True):
        p = run([sys.executable, "-I", str(ROOT / "infra/runpod/session_pod.py"), action, str(self.sd), *extra], timeout=120, check=check)
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"raw": (p.stdout + p.stderr)[-400:]}

    def create(self, prefer: list[str], tries: int = 6) -> dict | None:
        for g in prefer:
            name, rate = GPUS[g]
            for i in range(tries if g == prefer[0] else 2):
                r = self.pod("create", "--gpu", name, "--max-rate", str(rate), "--minutes", str(MINUTES), "--tier-cap", str(RESERVE), check=False)
                if r.get("created"):
                    self.log("pod_created", gpu=g, **{k: r[k] for k in ("pod_id", "costPerHr", "deadline")})
                    return r
                if (self.sd / "session.json").exists():
                    raise RuntimeError(f"session.json exists but create did not report success: {r}")
                self.log("pod_create_retry", gpu=g, attempt=i + 1, detail=str(r)[:160])
                time.sleep(45)
        return None

    def ssh(self, remote: str, timeout=120, check=True) -> subprocess.CompletedProcess:
        return run(["ssh", "-i", str(KEY), "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={self.sd / 'known_hosts'}", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", "-p", str(self.port), f"root@{self.ip}", remote], timeout=timeout, check=check)

    def wait_ssh(self, max_s=300) -> None:  # reachable Pods answered within 14-116 s; three that never did cost 12 min each at 720
        t0 = time.time()
        while time.time() - t0 < max_s:
            st = self.pod("status", check=False)
            ports = (st.get("runtime") or {}).get("ports") or []
            p22 = [p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")]
            if p22:
                self.ip, self.port = p22[0]["ip"], p22[0]["publicPort"]
                try:
                    if self.ssh("echo ok", timeout=40, check=False).stdout.strip().endswith("ok"):
                        (self.sd / "ssh.sh").write_text("#!/usr/bin/env bash\nD=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nexec ssh -i " + shlex.quote(str(KEY)) + " -o IdentitiesOnly=yes -o UserKnownHostsFile=\"$D/known_hosts\" "
                                                        f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -p {self.port} root@{self.ip} \"$@\"\n")
                        os.chmod(self.sd / "ssh.sh", 0o755)
                        self.log("ssh_ready", ip=self.ip, port=self.port, waited_s=int(time.time() - t0))
                        return
                except Exception:
                    pass
            time.sleep(12)
        raise RuntimeError("ssh not ready in time")

    def start_local_watchdog(self) -> None:
        wd = [sys.executable, "-I", str(ROOT / "infra/runpod/budget_watchdog.py"), "--session", str(self.sd / "session.json"), "--mode", "local"]
        run(wd + ["--check"], timeout=60)
        with open(self.sd / "watchdog-local.jsonl", "a") as f:
            subprocess.Popen(wd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        self.log("local_watchdog_started")

    def upload_and_start(self) -> None:
        self.ssh("mkdir -p /workspace/job/out")
        cmd = (f"tar cf - -C {shlex.quote(str(self.sd))} bundle session.json | ssh -i {KEY} -o IdentitiesOnly=yes -o UserKnownHostsFile={shlex.quote(str(self.sd / 'known_hosts'))} "
               f"-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -p {self.port} root@{self.ip} 'tar xf - -C /workspace/job'")
        run(["bash", "-o", "pipefail", "-c", cmd], timeout=900)
        r = self.ssh("cd /workspace/job && (cd bundle && sha256sum -c SHA256SUMS | grep -vc ': OK$'; true) && python3 bundle/budget_watchdog.py --session session.json --mode pod --check "
                     "&& (nohup python3 bundle/budget_watchdog.py --session session.json --mode pod >> out/pod-watchdog.log 2>&1 &) ; sleep 2; tail -1 out/pod-watchdog.log | cut -c1-160", timeout=300)
        first = r.stdout.strip().splitlines()[0] if r.stdout.strip() else "?"
        if first != "0" or "authentication_verified" not in r.stdout:
            raise RuntimeError(f"bundle verify / pod watchdog failed: {r.stdout[-300:]} {r.stderr[-300:]}")
        self.ssh("cd /workspace/job && (nohup bash bundle/run_remote.sh > out/run_remote.console 2>&1 &); sleep 3; head -1 out/workload.log")
        self.log("remote_started")

    def remote_state(self) -> dict:
        r = self.ssh("cd /workspace/job/out 2>/dev/null && { grep -E '^===' workload.log | tail -1; echo '@@'; grep -E 'FATAL|exit=' workload.log | tail -4; echo '@@'; "
                     "grep '^it ' train.log 2>/dev/null | head -8 | awk -F'|' '{print $3}'; echo '@@'; grep '^it ' train.log 2>/dev/null | tail -1 | cut -c1-110; echo '@@'; ls DONE CORE_DONE 2>/dev/null; "
                     "echo '@@'; cat host-info.txt 2>/dev/null | tr '\\n' ';'; echo '@@'; pgrep -f run_remote.sh >/dev/null && echo ALIVE || echo NOPROC; }", timeout=60, check=False)
        parts = [p.strip() for p in r.stdout.split("@@")]
        parts += [""] * (7 - len(parts))
        its = []
        for tok in parts[2].splitlines():
            try:
                its.append(float(tok.replace("it/s", "").strip()))
            except ValueError:
                pass
        return {"ok": r.returncode == 0, "step": parts[0], "exits": parts[1], "its": its, "last_it": parts[3], "done": "DONE" in parts[4].split(), "host": parts[5], "alive": "ALIVE" in parts[6]}

    def recover(self) -> dict:
        rec = self.sd / "recovered"
        rec.mkdir(exist_ok=True)
        self.ssh("cd /workspace/job && find out -type f ! -name RECOVERY_SHA256SUMS | sort | xargs sha256sum > out/RECOVERY_SHA256SUMS", timeout=600)
        cmd = (f"ssh -i {KEY} -o IdentitiesOnly=yes -o UserKnownHostsFile={shlex.quote(str(self.sd / 'known_hosts'))} -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ServerAliveInterval=30 "
               f"-p {self.port} root@{self.ip} 'cd /workspace/job && tar cf - out' | tee >(shasum -a 256 | cut -d' ' -f1 > {shlex.quote(str(rec / 'recovery.stream.sha256'))}) | tar xf - -C {shlex.quote(str(rec))}")
        run(["bash", "-o", "pipefail", "-c", cmd], timeout=1500)
        p = run(["bash", "-c", f"cd {shlex.quote(str(rec))} && shasum -a 256 -c out/RECOVERY_SHA256SUMS 2>&1 | grep -vc ': OK$'; wc -l < out/RECOVERY_SHA256SUMS"], check=False)
        bad, n = (p.stdout.split() + ["?", "?"])[:2]
        self.log("recovered", files=n.strip(), mismatches=bad.strip())
        return {"files": n.strip(), "mismatches": bad.strip()}

    def stop(self) -> dict:
        self.pod("stop", check=False)
        st = {}
        for _ in range(12):
            time.sleep(15)
            st = self.pod("status", check=False)
            if st.get("desiredStatus") == "EXITED" and st.get("runtime") is None:
                break
        with open(self.sd / "stop-evidence.json", "a") as f:
            f.write(json.dumps(st) + "\n")
        self.log("pod_stopped", desired=st.get("desiredStatus"), runtime_null=st.get("runtime") is None)
        return st


def one_session(a, idx: int, attempt: int, parent: Path | None, line_log: Log) -> tuple[str, Path | None, Path]:
    """Returns (status, final_ckpt, session_dir). status: ok | slow_host | failed | no_budget | no_capacity."""
    name = f"{a.base}-s{idx}-{a.date}" + (f"-r{attempt}" if attempt else "")
    sd = SESS / name
    if sd.exists():
        raise RuntimeError(f"{sd} exists; refusing to reuse a session id")
    sd.mkdir(parents=True)
    log = Log(sd / "driver.jsonl")
    if not reserve(a.tier, name, f"line {a.line_id} session {idx} attempt {attempt}; {a.config.name}"):
        line_log("no_budget", session=name)
        return "no_budget", None, sd
    mb = [sys.executable, "-I", str(ROOT / "infra/runpod/make_bundle.py"), str(sd), "--config", str(a.config), "--code-tar", str(a.code_tar), "--data-tar", str(a.data_tar),
          "--session-index", str(idx), "--max-hours", str(a.max_hours)]
    if parent:
        mb += ["--parent-ckpt", str(parent)]
    elif getattr(a, "init_ckpt", None):
        mb += ["--init-ckpt", str(a.init_ckpt)]  # first session of a new line: sm-train --init-from (weights of another run)
    if a.extra_script:
        mb += ["--extra-script", *map(str, a.extra_script)]
    if a.train_extra:
        mb += ["--train-extra", *a.train_extra]
    if a.render_clips and idx == a.sessions:
        mb += ["--render-clips", *a.render_clips]
    if a.post_cmd and idx == a.sessions:
        mb += ["--post-cmd", *a.post_cmd]
    if a.no_train:
        mb += ["--no-train"]
    if getattr(a, "nonfatal_checks", False):
        mb += ["--nonfatal-checks"]
    log("bundle", out=run(mb, timeout=300).stdout.strip()[-300:])
    s = Session(sd, log)
    created = stopped = None
    status, ckpt = "failed", None
    try:
        first = getattr(a, "gpu_first", "5090")
        other = "L40S" if first == "5090" else "5090"
        prefer = [first, other] if attempt == 0 else (["L40S", "5090"] if first == "5090" else [first, other])  # after a slow host: the L40S pool
        r = s.create(prefer)
        if not r:
            settle(a.tier, name, {"estimate_usd": 0.0, "note": "no capacity; no Pod created"})
            return "no_capacity", None, sd
        created = now()
        rate = float(r["costPerHr"])
        deadline = datetime.fromisoformat(r["deadline"])
        s.start_local_watchdog()
        s.wait_ssh()
        s.upload_and_start()
        gate_done = a.min_its <= 0 or a.no_train
        last = ""
        dead_polls = 0
        while True:
            time.sleep(60)
            st = s.remote_state()
            if st["ok"] and (st["step"], st["exits"]) != last:
                last = (st["step"], st["exits"])
                log("remote", step=st["step"][-60:], exits=st["exits"].replace("\n", " ")[-200:], last_it=st["last_it"][:80])
            if st["ok"] and not gate_done and len(st["its"]) >= 5:
                med = statistics.median(st["its"][2:5])
                gate_done = True
                log("speed_gate", its=st["its"][:6], median=med, min_its=a.min_its, host=st["host"][:200])
                if med < a.min_its and attempt < a.max_relaunch:
                    status = "slow_host"
                    break
            if st["ok"] and st["done"]:
                status = "ok" if "FATAL" not in st["exits"] else "failed"
                break
            dead_polls = dead_polls + 1 if (st["ok"] and not st["alive"] and not st["done"]) else 0
            if dead_polls >= 2:
                log("remote_script_exited_without_DONE", step=st["step"][-80:], exits=st["exits"][-200:])
                status = "failed"
                break
            if (deadline - now()).total_seconds() < 9 * 60:
                log("deadline_near_recovering_partial")
                status = "failed"
                break
        if status != "slow_host":
            rec = s.recover()
            c = sd / "recovered/out/train_logs"
            cands = sorted(c.glob("*/ckpt_latest.pt")) if c.exists() else []
            if status == "ok" and rec["mismatches"] == "0" and cands:
                ckpt = cands[0]
            elif status == "ok" and not (a.no_train and rec["mismatches"] == "0"):
                status = "failed"
    except Exception as e:  # noqa: BLE001
        log("error", detail=str(e)[-600:])
        status = "no_ssh" if "ssh not ready" in str(e) else "failed"  # a Pod that never comes up: another Pod is tried (line())
    finally:
        if created is not None:
            st = s.stop()
            stopped = now()
            est = round((stopped - created).total_seconds() / 3600 * rate, 4)
            total = settle(a.tier, name, {"pod_id": json.loads((sd / "session.json").read_text())["pod_id"], "gpu": json.loads((sd / "session.json").read_text())["gpu_type"], "rate": rate,
                                          "created_utc": created.isoformat(), "stopped_utc": stopped.isoformat(), "estimate_usd": est, "status": status,
                                          "note": f"line {a.line_id}; EXITED={st.get('desiredStatus')=='EXITED'} runtime_null={st.get('runtime') is None}"})
            log("settled", estimate_usd=est, tier_total=total, status=status)
    line_log("session_end", session=name, status=status, ckpt=str(ckpt) if ckpt else None)
    return status, ckpt, sd


def line(a) -> None:
    line_log = Log(SESS / f"line-{a.line_id}-{a.date}.jsonl")
    line_log("line_start", base=a.base, config=a.config.name, sessions=a.sessions, start_index=a.start_index)
    parent = a.parent_ckpt
    idx = a.start_index
    while idx <= a.sessions:
        attempt = 0
        while True:
            status, ckpt, sd = one_session(a, idx, attempt, parent, line_log)
            if status == "no_ssh" and attempt < max(a.max_relaunch, 1) + 1:  # never reachable: nothing ran, try another Pod (at most twice more)
                attempt += 1
                continue
            if status == "slow_host" and attempt < a.max_relaunch:
                attempt += 1
                continue
            break
        if status == "ok" and a.after_session:
            p = run(["bash", "-c", a.after_session.replace("{session_dir}", str(sd))], check=False, timeout=1800)
            line_log("after_session", rc=p.returncode, tail=(p.stdout + p.stderr)[-300:])
        if status != "ok":
            line_log("line_abort", at_session=idx, status=status)
            return
        parent = ckpt
        idx += 1
    line_log("line_done")


def main() -> None:
    global RESERVE, MINUTES
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("line")
    p.add_argument("--line-id", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--code-tar", type=Path, required=True)
    p.add_argument("--data-tar", type=Path, required=True)
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--parent-ckpt", type=Path)
    p.add_argument("--init-ckpt", type=Path, help="session 1 starts from another run's checkpoint (new line, counters at 0); later sessions resume as usual")
    p.add_argument("--max-hours", type=float, default=1.15)
    p.add_argument("--min-its", type=float, default=2.8)
    p.add_argument("--max-relaunch", type=int, default=1)
    p.add_argument("--tier", default="N1")
    p.add_argument("--extra-script", type=Path, nargs="*", default=[])
    p.add_argument("--train-extra", nargs="*", default=[])
    p.add_argument("--render-clips", nargs="*", default=[])
    p.add_argument("--post-cmd", nargs="*", default=[])
    p.add_argument("--after-session", default="")
    p.add_argument("--no-train", action="store_true", help="evaluation-only session: the bundle's post commands do the work")
    p.add_argument("--nonfatal-checks", action="store_true", help="smoke runs: failing unit tests / numeric checks are logged, the script goes on")
    p.add_argument("--gpu-first", choices=["5090", "L40S"], default="5090", help="which pool to try first (2026-11-10: every slow host was a 5090 on EPYC 7543; L40S hosts were 3.0-3.3 it/s)")
    p.add_argument("--reserve-usd", type=float, default=RESERVE, help="budget held per session (short eval-only sessions can hold less)")
    p.add_argument("--minutes", type=int, default=MINUTES, help="Pod fuse in minutes (never above 115)")
    a = ap.parse_args()
    if getattr(a, "reserve_usd", None) is not None:
        RESERVE = a.reserve_usd
    if getattr(a, "minutes", None) is not None:
        assert 20 <= a.minutes <= 115, "fuse must stay within 20..115 minutes"
        MINUTES = a.minutes
    line(a)


if __name__ == "__main__":
    main()
