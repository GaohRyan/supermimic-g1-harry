# RunPod runbook

Last substantive update: 2026-11-03. Since 2026-11-10 every Pod is started through `session_driver.py`; the rules are in
[PIPELINE §8 and §14](../../docs/PIPELINE.md). This page covers access, storage and what happens on the Pod.

## Access

Account-specific settings (account email, API key file, SSH key, network volume ID) are read from environment
variables `RUNPOD_<NAME>` or from `.secrets/runpod-site.json`; copy [`site.example.json`](site.example.json) there.

Check the account with `python3 infra/runpod/account_check.py`; it prints no keys and does not change the global CLI
configuration. The default CLI identity once pointed at a different account, so the check matters.

Storage uses a dedicated S3 key for the project's network volume. The platform grants that key read/write on every
S3-compatible volume of the account; our helper only touches `<volume-id>/supermimic/`, which is a client-side limit,
not server-side isolation. The credential lives in `.secrets/runpod-s3.ini` (directory 700, file 600), excluded from
git and from release bundles.

Per the RunPod [S3 documentation](https://docs.runpod.io/storage/s3-api), S3 keys are separate from Pod API keys and the
bucket name is the network volume ID. Pods mount `/workspace` by default, but check at start-up. The service has no
bucket versioning, so versions are managed by write-once run / release paths and SHA256.

Credentials never go into the repository, run manifests or chat. The helper prefers the project credential file and
clears inherited AWS environment credentials to avoid mixing accounts; without the file it uses the standard AWS chain.
A RunPod API key cannot be used as an AWS secret.

## Storage

The AWS CLI is installed in an isolated `.venv-infra`. Rebuild it and check storage:

```bash
python3 -m venv .venv-infra
.venv-infra/bin/python -m pip install -r infra/runpod/requirements-infra.txt
PATH="$PWD/.venv-infra/bin:$PATH" bash infra/runpod/s3.sh ls
```

`ls` lists one level under the project prefix, at most 100 entries; an empty prefix is a success, and `IsTruncated`
says whether there is more. Release bundles are built by `python3 scripts/package_release.py` and uploaded with
`PATH="$PWD/.venv-infra/bin:$PATH" bash infra/runpod/s3.sh upload <bundle>`; the helper verifies local checksums and
contents, stores by file SHA, downloads again to verify, and refuses to overwrite different content. Upload receipts
go to the git-ignored `artifacts/receipts/`.

## Pod sizing (first round)

The first round used one NVIDIA L4 in US-MO-2 at $0.59/h, within a $15 total for setup, tests and smoke runs, with a
60 GB container disk and SSH transfer because the original region had no GPU. Full training sizes capacity from
measured library, replay and BPTT memory; later sessions use L40S or RTX 5090 chosen by the speed gate.

The container must run the upstream's locked Python 3.12 / CUDA 13 PyTorch stack; record the image digest. Before
creating a Pod fix the GPU, live price, maximum duration, total budget, volume mountability and what happens on exit.

`--schedule.max-hours` only limits the training loop; it excludes installation / compilation and does not stop
billing. The real limit is an external deadline plus resource recovery: `budget_watchdog.py` runs independently on the
local machine and on the Pod, both must pass `--check`, the Pod uses only its auto-injected Pod-scoped key, the
deadline is at most 2 hours after creation, and a finished session is stopped early and confirmed `EXITED`. The old
CLI `--stop-after` / `--terminate-after` are not executed by the backend and give no protection
([fix note](https://github.com/runpod/runpodctl/pull/330)).

## On the Pod

1. Record the actual persistence and capacity of `/workspace`. On a temporary container disk, recover all evidence
   before stopping.
2. Verify the bundle SHA256 and unpack to `/workspace/supermimic/code/<release-id>/`. From git, use
   `git clone --recurse-submodules` and verify both commits.
3. Run `bash infra/runpod/bootstrap.sh` in the code root. It needs `uv` on the Pod, installs no drivers, adds
   `libegl1 / libgl1 / libglib2.0-0` on Ubuntu (recording dpkg versions) and installs the locked Python dependencies.
   Preflight imports MuJoCo and runs CPU FK; a working CUDA tensor is not enough.
4. Prepare the pinned data as in `docs/02-reproduction-plan.md`, then run the upstream numeric tests and the smoke run.
5. Save the exact command, config, environment, logs, checkpoints and evaluation in a new run directory.

When cloning from git, data for the first tests can be linked like this (only if the targets do not exist):

```bash
mkdir -p upstream/SuperMimic-Reproduced/data/npz upstream/SuperMimic-Reproduced/data/raw/lafan1
ln -s /workspace/supermimic/data/processed/lafan1-v1 \
  upstream/SuperMimic-Reproduced/data/npz/lafan1
ln -s /workspace/supermimic/data/raw/lafan1/g1 \
  upstream/SuperMimic-Reproduced/data/raw/lafan1/g1
```

Release tars carry no `.git`; `release.json` holds the commit / hash evidence. Training only writes to
`runs/<run-id>/training`.

## Finishing and resuming

After training, check that checkpoints load, logs are flushed and per-clip evaluation is readable; then recover
reports, logs, checkpoints and processed data locally and finish the S3 upload with read-back verification. The
wrapper records the real final state (`process_succeeded / process_failed / timed_out / interrupted`), exit code and
cost estimate; actual billing is listed separately. Closing a terminal does not stop billing.

Resuming always creates a new run with a parent checkpoint. The upstream does not save all runtime state, so a resume is
a warm restart. Keep an independent backup of key final models, because S3 and the Pod mount are the same data.

## Pipeline entry

New training is registered, bound to a runtime and executed with time limits as described in
pipeline operations. The timeout of `run_pipeline.py` does not stop the Pod;
separate local and on-Pod stop monitors are required, and every session ends with an explicit stop and an `EXITED` check.
Code, data and run reports are published as content-addressed snapshots, and a completion marker is written only after
the S3 read-back passes.
