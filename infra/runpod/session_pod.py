#!/usr/bin/env python3
"""Create / stop / inspect exactly one task-owned RunPod Pod via GraphQL. Never deletes anything.

  python3 -I infra/runpod/session_pod.py create <session_dir> --gpu "NVIDIA GeForce RTX 5090" --max-rate 1.30 --minutes 90 --tier-cap 3
  python3 -I infra/runpod/session_pod.py stop   <session_dir>
  python3 -I infra/runpod/session_pod.py status <session_dir>

Guards: account email check (RUNPOD_ACCOUNT_EMAIL or .secrets/runpod-site.json); Secure on-demand only; live quote <= --max-rate; deadline <= 120 min;
worst-case GPU cost (rate x minutes) <= --tier-cap <= 22; refuses to create if session.json already has a
pod_id; stop/status act only on the pod_id recorded in that session. Credentials never enter output/argv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

IMAGE = "runpod/base@sha256:4a59ffba7b63fea6fef6ee88a005134309e9179a74f0593cc09255905c61ca2a"
ALLOWED_GPUS = {"NVIDIA GeForce RTX 5090", "NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3"}


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


def key() -> str:
    k = Path(_site("api_key_file", "~/.runpod/api.key")).expanduser().read_text().strip()
    if not k or "\n" in k:
        raise SystemExit("credential unavailable")
    return k


def gql(q: str, variables: dict | None = None) -> dict:
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key(), "User-Agent": "supermimic-session/1"})
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.loads(r.read())
    if d.get("errors"):
        raise SystemExit(json.dumps({"ok": False, "errors": [e.get("message") for e in d["errors"]]}))
    return d["data"]


def check_account() -> None:
    account = _site("account_email").lower()
    if not account or gql("query{ myself{ email } }")["myself"]["email"].lower() != account:
        raise SystemExit("wrong account")


def create(a) -> None:
    sd = Path(a.session_dir)
    sp = sd / "session.json"
    if sp.exists() and json.loads(sp.read_text()).get("pod_id"):
        raise SystemExit("session already has a pod_id; refuse to create a second Pod")
    if a.gpu not in ALLOWED_GPUS or not 0 < a.minutes <= 120 or not 0 < a.tier_cap <= 22 or a.max_rate * a.minutes / 60 > a.tier_cap:
        raise SystemExit("guard violation: gpu/minutes/tier cap")
    check_account()
    q = "query($in: GpuLowestPriceInput){ gpuTypes(input:{id:%s}){ id memoryInGb lowestPrice(input:$in){ uninterruptablePrice stockStatus } } }" % json.dumps(a.gpu)
    quote = gql(q, {"in": {"gpuCount": 1, "secureCloud": True}})["gpuTypes"][0]
    price = quote["lowestPrice"]["uninterruptablePrice"]
    if price is None or price > a.max_rate:
        print(json.dumps({"created": False, "reason": "no quote or above max rate", "quote": quote}))
        sys.exit(2)
    name = sd.name
    m = ('mutation { podFindAndDeployOnDemand(input:{cloudType:SECURE, gpuCount:1, gpuTypeId:%s, name:%s, imageName:%s, '
         'containerDiskInGb:40, volumeInGb:0, ports:"22/tcp", startSsh:true, supportPublicIp:true, minVcpuCount:8, minMemoryInGb:30, '
         'allowedCudaVersions:["13.0","13.1","13.2"]}) { id desiredStatus costPerHr imageName gpuCount vcpuCount memoryInGb containerDiskInGb machine{ gpuDisplayName } } }'
         ) % (json.dumps(a.gpu), json.dumps(name), json.dumps(IMAGE))
    created = datetime.now(timezone.utc)
    p = gql(m)["podFindAndDeployOnDemand"]
    if not p:
        print(json.dumps({"created": False, "reason": "no capacity", "quote": quote}))
        sys.exit(3)
    s = {"schema_version": 1, "name": name, "created_for_this_task": True, "created_utc": created.isoformat(),
         "stop_deadline_utc": (created + timedelta(minutes=a.minutes)).isoformat(),
         "workload_archive_cutoff_utc": (created + timedelta(minutes=a.minutes - 15)).isoformat(),
         "authorized_total_budget_usd": a.tier_cap, "budget_field_semantics": "tier cap of the 70 USD program approved 2026-11-05T21:11Z",
         "quoted_gpu_hourly_usd": price, "max_acceptable_actual_hourly_usd": a.max_rate, "gpu_type": a.gpu, "image_digest": IMAGE,
         "container_disk_gb": 40, "persistent_volume_gb": 0, "workspace_persistent": False, "capacity_quote": quote,
         "pod_id": p["id"], "creation_fields": p}
    sd.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(s, indent=1))
    if p.get("costPerHr") and p["costPerHr"] > a.max_rate:
        stop_pod(p["id"])
        raise SystemExit("actual rate above max; stop requested")
    print(json.dumps({"created": True, "pod_id": p["id"], "costPerHr": p["costPerHr"], "deadline": s["stop_deadline_utc"]}))


def owned(a) -> str:
    s = json.loads((Path(a.session_dir) / "session.json").read_text())
    if s.get("created_for_this_task") is not True or not s.get("pod_id"):
        raise SystemExit("no task-owned pod recorded")
    return s["pod_id"]


def stop_pod(pid: str) -> dict:
    return gql("mutation { podStop(input:{podId:%s}) { id desiredStatus } }" % json.dumps(pid))["podStop"]


def status_of(pid: str) -> dict:
    return gql("query { pod(input:{podId:%s}) { id desiredStatus costPerHr runtime{ uptimeInSeconds ports{ ip isIpPublic privatePort publicPort type } } } }" % json.dumps(pid))["pod"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create", "stop", "status"])
    ap.add_argument("session_dir")
    ap.add_argument("--gpu")
    ap.add_argument("--max-rate", type=float)
    ap.add_argument("--minutes", type=int)
    ap.add_argument("--tier-cap", type=float)
    a = ap.parse_args()
    if a.action == "create":
        create(a)
        return
    check_account()
    pid = owned(a)
    out = stop_pod(pid) if a.action == "stop" else status_of(pid)
    out["observed_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
