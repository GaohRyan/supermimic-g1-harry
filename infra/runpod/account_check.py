#!/usr/bin/env python3
"""Verify the project RunPod credential without altering the global RunPod login."""
import json
import os
import subprocess
import sys
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

credential = Path(_site("api_key_file", "~/.runpod/api.key")).expanduser()
try:
    key = credential.read_text().strip()
    if not key or "\n" in key or key.startswith("-----BEGIN"):
        raise ValueError("Expected a single-line API credential")
    env = dict(os.environ, RUNPOD_API_KEY=key)

    def query(*args):
        p = subprocess.run(["runpodctl", *args], env=env, capture_output=True, text=True, timeout=30)
        if p.returncode:
            raise RuntimeError(f"RunPod command {args[0]} failed (exit {p.returncode})")
        return json.loads(p.stdout)

    account = query("user")
    match = str(account.get("email", "")).lower() == _site("account_email").lower() != ""
    if not match:
        raise ValueError("Credential is not associated with the configured account")
    volume = query("network-volume", "get", _site("volume_id"))
    pods = query("pod", "list", "--all")
    report = {"account_verified": True, "global_cli_config_modified": False,
              "volume": {k: volume.get(k) for k in ["id", "name", "size", "dataCenterId"]},
              "pods": [{k: p.get(k) for k in ["id", "name", "desiredStatus", "networkVolumeId"]} for p in pods]}
    print(json.dumps(report, indent=2))
except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
    print(json.dumps({"account_verified": False, "error": str(e)}))
    sys.exit(1)
