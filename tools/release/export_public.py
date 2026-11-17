#!/usr/bin/env python3
"""Build the public release tree from a commit of this (private) repository.

The public repository is a fresh history with one commit; nothing from the private history is carried over.
Only allow-listed paths are copied; personal or account details are removed; a scan must come back clean.

    python3 -I tools/release/export_public.py <out_dir> [--rev origin/main] [--commit]

<out_dir> must not exist. With --commit the script runs `git init` there and makes one commit (no remote, no push).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INCLUDE = [
    "README.md", "LICENSE", ".gitignore", ".gitmodules", ".github/pull_request_template.md",
    "docs/NOTES.md", "docs/PIPELINE.md", "docs/decisions.md", "docs/status.md", "docs/assets/",
    "configs/protocols/", "configs/cuboi-g1/",
    "variants/instanttrack-g1/native-v13.json", "variants/instanttrack-g1/native-v13.patches/",
    "infra/runpod/README.md", "infra/runpod/site.example.json", "infra/runpod/session_driver.py",
    "infra/runpod/session_pod.py", "infra/runpod/make_bundle.py", "infra/runpod/budget_watchdog.py",
    "infra/runpod/budget_watchdog_v2.py", "infra/runpod/account_check.py", "infra/runpod/bootstrap.sh",
    "infra/runpod/system_dependencies.sh", "infra/runpod/requirements-infra.txt", "infra/runpod/resource_monitor.py",
    "infra/runpod/pod_checks/", "infra/runpod/pod_jobs/",
    "tools/offline/", "tools/sim2sim/", "tools/terrain/", "tools/release/",
    "scripts/checkout_variant.sh", "tests/infra/",
]
CUBOI_VARIANTS = ["corrected-baseline", "anchor-ori", "official-contract", "sim2real-dr", "sampler-norm",
                  "wm-hard-mining", "optim", "terrain-l1"]
INCLUDE += [f"variants/cuboi-g1/{v}.json" for v in CUBOI_VARIANTS] + [f"variants/cuboi-g1/{v}.patches/" for v in CUBOI_VARIANTS]
EXCLUDE = [
    re.compile(r"^configs/cuboi-g1/(video|terrain-l2|platform)"),   # later steps, not in this release
    re.compile(r"^infra/runpod/pod_jobs/gvhmr_job\.sh$"),           # video line
    re.compile(r"^tools/terrain/omni_terrain\.py$"),                 # level-2 data preparation
]
# Decision-log rows about personal study, other projects or account setup.
DROP_DECISIONS = {"D002", "D005", "D006", "D007", "D009", "D011", "D012", "D015", "D016", "D017", "D018", "D020", "D021",
                  "D022", "D023", "D031", "D035"}
AUTHOR = "Hang(Harry) Gao <GaohRyan@users.noreply.github.com>"
SCAN = [
    (re.compile("[\\u4e00-\\u9fff]"), "CJK text"),
    (re.compile(r"/Use[r]s/|/hom[e]/[a-z]", re.I), "local path"),
    (re.compile(r"rpa_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY"), "secret-like string"),
]
BLOCKED_SUFFIX = {".mp4", ".mov", ".webm", ".gif", ".pt", ".ckpt", ".npz", ".pkl", ".h5", ".safetensors", ".pdf", ".bvh", ".csv"}
MAX_BYTES = 2_000_000


def private_patterns() -> list:
    """Account names, e-mails, volume IDs etc. to scan for, one regex per line, kept out of git in .secrets/."""
    f = ROOT / ".secrets/release-scan.txt"
    if not f.exists():
        sys.exit(f"missing {f}: list the personal and account strings to scan for, one regex per line")
    return [(re.compile(line.strip(), re.I), "personal or account detail")
            for line in f.read_text().splitlines() if line.strip() and not line.startswith("#")]


def wanted(path: str) -> bool:
    if any(rx.search(path) for rx in EXCLUDE):
        return False
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in INCLUDE)


def fix_patch(text: str) -> str:
    return re.sub(r"^From: .*<[^>]+>$", "From: " + AUTHOR, text, flags=re.M)


def fix_decisions(text: str) -> str:
    out = []
    for line in text.splitlines():
        m = re.match(r"\| (D\d{3}) /", line)
        if m and m.group(1) in DROP_DECISIONS:
            continue
        out.append(line)
    out.insert(2, "Rows about personal study plans, other projects and account setup are omitted from the public copy.\n")
    return "\n".join(out) + "\n"


def fix_pipeline(text: str) -> str:
    text = text.replace("- **Repository**: <https://github.com/GaohRyan/supermimic-g1> (private).",
                        "- **Repository**: <https://github.com/GaohRyan/supermimic-g1-harry>, published from a private working repository.")
    return re.sub(r"\nLocal storage: .*?formal backup = local \+ S3 with read-back verification\)\.\n", "\n", text, flags=re.S)


def fix_links(md: Path, out: Path) -> None:
    text = md.read_text()

    def repl(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"[a-z]+://|#|mailto:", target):
            return m.group(0)
        dest = (md.parent / target.split("#")[0]).resolve()
        return m.group(0) if dest.exists() else label

    md.write_text(re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", repl, text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} already exists")
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", a.rev], check=True, capture_output=True).stdout
    a.out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            if m.isfile() and wanted(m.name):
                t.extract(m, a.out)
    for v in CUBOI_VARIANTS + ["../instanttrack-g1/native-v13"]:
        vj = a.out / "variants/cuboi-g1" / f"{v}.json"
        meta = json.loads(vj.read_text())
        for p in meta.get("patches", []):
            pf = vj.parent / f"{v}.patches" / p["file"]
            pf.write_text(fix_patch(pf.read_text()))
            p["sha256"] = hashlib.sha256(pf.read_bytes()).hexdigest()
        meta["note_public"] = "Patch author lines were normalised for publication; variant_tree is unchanged, commit SHAs after git am differ."
        vj.write_text(json.dumps(meta, indent=1) + "\n")
    (a.out / "docs/decisions.md").write_text(fix_decisions((a.out / "docs/decisions.md").read_text()))
    (a.out / "docs/PIPELINE.md").write_text(fix_pipeline((a.out / "docs/PIPELINE.md").read_text()))
    for md in a.out.rglob("*.md"):
        fix_links(md, a.out)

    problems = []
    scan = SCAN + private_patterns()
    for f in sorted(p for p in a.out.rglob("*") if p.is_file()):
        rel = f.relative_to(a.out).as_posix()
        if f.suffix.lower() in BLOCKED_SUFFIX:
            problems.append(f"{rel}: blocked file type")
        if f.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: larger than {MAX_BYTES} bytes")
        if f.suffix.lower() in {".png", ".jpg", ".webp"}:
            continue
        text = f.read_text(errors="replace")
        for rx, what in scan:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{rel}:{line}: {what}")
                break
    files = [p for p in a.out.rglob("*") if p.is_file()]
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    if problems:
        print("\n".join(problems))
        return 1
    if a.commit:
        g = ["git", "-C", str(a.out)]
        subprocess.run(g + ["init", "-q", "-b", "main"], check=True)
        sub = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{a.rev}:upstream/SuperMimic-Reproduced"],
                             check=True, capture_output=True, text=True).stdout.strip()
        subprocess.run(g + ["add", "-A"], check=True)
        subprocess.run(g + ["update-index", "--add", "--cacheinfo", f"160000,{sub},upstream/SuperMimic-Reproduced"], check=True)
        name, email = AUTHOR[:-1].split(" <")
        subprocess.run(g + ["-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m",
                            "SuperMimic on the Unitree G1: reproduction, three improvements, stairs (level 1)"], check=True)
        print(subprocess.run(g + ["log", "--stat", "--oneline", "-1"], capture_output=True, text=True).stdout.splitlines()[0])
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
