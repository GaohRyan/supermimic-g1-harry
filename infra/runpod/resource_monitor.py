#!/usr/bin/env python3
"""Sample system resources across a workflow; this is not a PyTorch operator profiler."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: str) -> float | None:
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def sample(disk_path: Path, timeout: float) -> dict:
    result = {"type": "sample", "utc": utc_now(), "gpus": [], "errors": []}
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True, shell=False,
        )
        for row in csv.reader(io.StringIO(process.stdout)):
            if len(row) != 8:
                raise ValueError("Unexpected GPU metric columns")
            result["gpus"].append({
                "index": row[0].strip(), "uuid": row[1].strip(), "name": row[2].strip(),
                "utilization_percent": finite_number(row[3]),
                "memory_used_mib": finite_number(row[4]), "memory_total_mib": finite_number(row[5]),
                "power_watts": finite_number(row[6]), "temperature_celsius": finite_number(row[7]),
            })
    except (OSError, subprocess.SubprocessError, ValueError, csv.Error) as error:
        result["gpus"] = []
        result["errors"].append({"source": "nvidia_smi", "error_type": type(error).__name__})
    try:
        result["cpu_load_average"] = dict(zip(["1_min", "5_min", "15_min"], os.getloadavg()))
    except (OSError, AttributeError) as error:
        result["errors"].append({"source": "cpu_load", "error_type": type(error).__name__})
    try:
        usage = shutil.disk_usage(disk_path)
        result["disk"] = {"path": str(disk_path), "free_bytes": usage.free, "total_bytes": usage.total}
    except OSError as error:
        result["errors"].append({"source": "disk", "error_type": type(error).__name__})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New JSONL evidence file; existing files are refused")
    parser.add_argument("--stop-file", required=True, type=Path, help="Stop when this file exists; its contents are not read")
    parser.add_argument("--max-seconds", type=float, default=7200, help="Maximum monitor duration, greater than zero and at most 7200")
    parser.add_argument("--interval", type=float, default=5, help="Seconds between scheduled samples, at least 2")
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 7200:
        parser.error("--max-seconds must be greater than zero and at most 7200")
    if not math.isfinite(args.interval) or args.interval < 2:
        parser.error("--interval must be at least 2 seconds")
    output = args.output.resolve()
    stop_file = args.stop_file.resolve()
    summary_path = output.with_name(output.stem + ".summary.json")
    if output == summary_path or stop_file in {output, summary_path}:
        parser.error("Output, summary and stop-file paths must be different")
    if output.exists() or summary_path.exists():
        parser.error("Evidence output or summary already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = utc_now()
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signal.Signals(signum).name

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    metadata = {
        "schema_version": 1, "kind": "system_resource_sampling", "started_utc": started_utc,
        "interval_seconds": args.interval, "max_seconds": args.max_seconds,
        "stop_file": str(stop_file), "disk_path": str(output.parent),
        "scope": "Observed monitor window may span tests and training; no per-run attribution.",
        "limitations": "Sampled system metrics, not a PyTorch operator profiler. Peaks between samples and activity before monitoring are not observed.",
    }
    count = 0
    first_elapsed = last_elapsed = None
    error_count = 0
    gpu_stats = {}
    next_sample = started
    reason = "max_seconds"
    with output.open("x") as stream:
        stream.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        stream.flush()
        while True:
            now = time.monotonic()
            remaining = args.max_seconds - (now - started)
            if stop_signal:
                reason = stop_signal
                break
            if stop_file.exists():
                reason = "stop_file_present"
                break
            if remaining <= 0:
                break
            if now < next_sample:
                time.sleep(min(1, next_sample - now, remaining))
                continue
            value = sample(output.parent, timeout=min(5, remaining))
            elapsed = now - started
            value["elapsed_seconds"] = round(elapsed, 3)
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            count += 1
            error_count += len(value["errors"])
            first_elapsed = elapsed if first_elapsed is None else first_elapsed
            last_elapsed = elapsed
            for gpu in value["gpus"]:
                gpu_id = gpu["uuid"] or gpu["index"]
                stats = gpu_stats.setdefault(gpu_id, {"index": gpu["index"], "uuid": gpu["uuid"], "name": gpu["name"],
                                                     "sample_count": 0, "utilization_sum": 0, "utilization_sample_count": 0,
                                                     "peak_sampled_memory_used_mib": None, "memory_total_mib": gpu["memory_total_mib"]})
                stats["sample_count"] += 1
                if gpu["utilization_percent"] is not None:
                    stats["utilization_sum"] += gpu["utilization_percent"]
                    stats["utilization_sample_count"] += 1
                if gpu["memory_used_mib"] is not None:
                    peak = stats["peak_sampled_memory_used_mib"]
                    stats["peak_sampled_memory_used_mib"] = max(peak or 0, gpu["memory_used_mib"])
            next_sample += args.interval
            # Skip missed polling slots instead of creating a catch-up burst.
            if next_sample < time.monotonic():
                next_sample = time.monotonic() + args.interval
        for stats in gpu_stats.values():
            total = stats.pop("utilization_sum")
            n = stats["utilization_sample_count"]
            stats["mean_sampled_utilization_percent"] = round(total / n, 3) if n else None
        summary = {
            **metadata, "ended_utc": utc_now(), "stop_reason": reason,
            "duration_seconds": round(time.monotonic() - started, 3), "sample_count": count,
            "sample_coverage_seconds": round(last_elapsed - first_elapsed, 3) if count else 0,
            "first_sample_elapsed_seconds": first_elapsed, "last_sample_elapsed_seconds": last_elapsed,
            "error_count": error_count, "gpus": list(gpu_stats.values()),
            "utilization_mean_method": "Arithmetic mean of valid sampled percentages per GPU; not time weighted.",
        }
        stream.write(json.dumps({"type": "summary", **summary}, allow_nan=False) + "\n")
        stream.flush()
    with summary_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "summary": str(summary_path), "sample_count": count}))


if __name__ == "__main__":
    main()
