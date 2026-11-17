# SuperMimic pipeline and infrastructure rules (single source of truth)

Version v1.1, 2026-11-06 (adds §13, repository and branch conventions). Applies to every project (cuboi-g1, instanttrack-g1,
vibed-g1, goyoungster-g1, allmimic-g1, ...).
This file defines how **data → training → checkpoint → evaluation (including sim2sim) → archive** is managed. Every
training, inference, evaluation, rendering or sim2sim run must pass the per-run gate in §9. The older
06-pipeline-operations.md describes the legacy Cuboi execution chain and still applies;
where they conflict this file wins, and the conflict is recorded in [decisions.md](decisions.md).

---

## 1. Five invariants

1. **Everything is addressable.** Every dataset, source tree, config, checkpoint and evaluation result has a SHA256 and is
   referenced by its downstream manifest. Nothing without a hash enters training or evaluation.
2. **Projects are isolated.** Each `project_id` has its own source copy, uv environment, processed cache, run directory
   and S3 prefix; checkpoints are not compatible across projects by default.
3. **Nothing is overwritten.** Runs, datasets, checkpoints and evaluation results are written once; a retry gets a new ID.
   Upstream snapshots (`upstream/`) are never modified; patches go to `variants/<project>/<variant>/` with the base commit
   and patch hash recorded.
4. **Protocols are frozen first.** The evaluation protocol (denominator, initialization, noise / DR, termination,
   clip list) is written as a protocol JSON and hashed before the run; criteria do not change during the run, and
   failures are archived like successes.
5. **Cloud resources are bounded.** One paid Pod at a time; reserve budget before creating it and derive an absolute
   stop time from that budget (a cost fuse, not a progress deadline; see §8.1); a local and an on-Pod watchdog; after
   the session, independently confirm `EXITED / runtime=null` and record the cost.

---

## 2. Directories and namespaces

```text
supermimic/
├─ upstream/<repo>/                      # original snapshots at fixed commits, read only
├─ variants/<project_id>/<variant>/      # modifiable copy (git, own commits); e.g. variants/cuboi-g1/corrected-baseline
├─ projects/<project_id>/                # project.json / pipeline.json / source-manifest.json (registry, no large files)
├─ configs/protocols/<protocol_id>.json  # frozen evaluation protocols
├─ data/
│   ├─ raw/<dataset_id>/<source_revision>/        # raw download + manifest.json (per-file sha256, source URL@revision, license)
│   └─ processed/<project_id>/<dataset_id>/<recipe_hash>/   # converted data + manifest (raw manifest hash + converter commit)
├─ runs/<project_id>/<run_id>/           # all evidence of one run (see §4)
├─ artifacts/
│   ├─ p0/                               # workspace protection list, budget ledger, read-only checks
│   ├─ projects/<project_id>/...         # external weights (e.g. release-e8-boneseed-100k), patches, protocol outputs
│   ├─ infra_sessions/<session_id>/      # each Pod session: session.json, watchdog logs, bundle, billing, stop evidence
│   └─ reports/index.{json,md}           # index of all runs
└─ docs/PIPELINE.md                      # this file
```


S3 (the project network volume, only the `supermimic/` prefix): all new content goes to
`supermimic/projects/<project_id>/{data,runs,checkpoints,eval,sessions}/<id>/snapshots/<bundle_sha>/`.
Legacy keys (`supermimic/{runs,data,releases,infra-sessions}`) stay where they are. S3 and the network volume are the
**same data**; the local copy is the second copy. Key checkpoints must exist at least locally and on S3.

ID rules: `run_id = <UTCyyyymmddTHHMMSSZ>-<project_id>-<purpose>-<8hex>`; `session_id = <project_id>-<phase>-<purpose>-<yyyymmdd>`.

---

## 3. Data management

| Stage | Output | Must record | Status |
|---|---|---|---|
| Download | `data/raw/<id>/<rev>/` | Source URL **pinned to a revision** (not `main`), per-file sha256/bytes, license file | `raw_verified` |
| Convert | `data/processed/<project>/<id>/<recipe>/` | Converter repo + commit, command argv, input raw manifest hash, per-output sha256, frame count / FPS / dims / finiteness / quaternion order checks | `processed_verified` |
| Filter | `filter-manifest.json` | Every removed clip and why; matching counts do not mean matching sets | — |
| Clip set | `clipset-<name>.json` | Ordered clip names + processed sha256 of each; training and evaluation reference only the clipset hash | frozen |

Rules: real training and evaluation reject `raw_verified` data; when different projects convert the same raw data each
produces its own processed copy (reuse across projects needs an equivalence test); credentials for gated data
(BONES-SEED) live only in local files, never in argv, logs or bundles.

Registered: `data/raw/lafan1-g1-smoke/ce15729…` (4 clips, converted and verified on 2026-11-03);
`data/raw/instanttrack-g1/ce15729…` (40 clips + G1 description, 79 files, manifest with sha256).

---

## 4. Training run lifecycle and contents

State machine: `prepared_not_started → running → {process_succeeded | process_failed | timed_out | interrupted | launch_failed}`.
A run is never rerun; a warm restart creates a new run bound to `parent_run_id + parent_checkpoint_sha256` and is
labelled "not an exact resume" (replay / RNG / simulator state not restored).

Every `runs/<project_id>/<run_id>/` contains:

```text
manifest.json         project_id, variant, source{repo, base_commit, variant_commit, patch_sha256, lock_sha256},
                      runtime{pod_id, gpu, driver, image_digest, torch/cuda/mujoco versions, hourly_price},
                      data{processed_manifest_sha256, clipset_sha256}, seed, argv, cwd, parent, protocol_ids
config.resolved.json  full resolved config (including reductions vs the paper contract: env count, batch, H, ...)
logs/                 stdout/stderr, tensorboard
training/             checkpoints (see §5)
eval/<protocol_id>/<ckpt_sha8>/   initial / periodic / final evaluation (see §6)
metrics.json          updates, gradient norms, it/s, peak memory, phase timings, cost; unmeasured items are null (never 0)
artifacts.json        size + sha256 of every file
report.md             sources, results, limitations, comparison with acceptance thresholds; written for failures too
```

The table of differences from the paper contract (H, env count, WM windows, replay, UTD counting, state loss sum/mean,
...) goes into `deviations_from_paper_contract` in `config.resolved.json`, and the report cites it.

---

## 5. Checkpoint management

- **Registry**: `training/checkpoints.json` lists `{iteration, num_updates, path, sha256, bytes, kind: light|full,
  contains: [wm, policy, critic, optim, normalizer, sampler, rng?], wall_time, sim_steps}` for each checkpoint.
- **Retention**: initial (before the first update), fixed intervals, best-by-train-return only as a marker (**never
  reported as the result**), final; keep at least initial + latest + final as full checkpoints.
- **Recovery**: copy each retained checkpoint back from the Pod as it is produced and verify its sha256; stop producing
  new ones 15 minutes before the deadline and make one last bounded copy.
- **Safe loading**: always `torch.load(weights_only=True)`, with `add_safe_globals` allow-lists as needed (e.g.
  `pathlib.PosixPath`) written into the evaluation manifest. Statically scan pickle globals of external weights first.
- **External weights** (e.g. Cuboi `e8-boneseed-100k`) live in `artifacts/projects/<project>/release-<tag>/` with the
  GitHub digest; **always labelled external, never counted as our training result**.

---

## 6. Evaluation (inference) pipeline

All inference and evaluation is defined by a (protocol, checkpoint, clipset) triple; results go to
`eval/<protocol_id>/<ckpt_sha8>/<clipset_sha8>/`.

**Default protocol `fullclip-nominal-v1`** (file `configs/protocols/fullclip-nominal-v1.json`, implementation
`variants/cuboi-g1/corrected-baseline/scripts/eval_fullclip_traj.py`):

1. One env per clip, from the first frame, fixed initialization, `reset_noise=False`, no observation noise, action noise
   0, deterministic policy.
2. **Build a new nominal environment** (DR off, friction 1.0) and read back friction / COM / encoder bias at run time
   into the results; turning off DR sampling alone is not enough.
3. `auto_reset=False`: save the terminal state s_{T+1} before reset and the termination reason (each termination term's flag).
4. Return = `reward_sum / (0.1 × planned_steps)`, with `planned_steps` fixed to clip frames − 1; **not normalized by
   survived steps** (the upstream number is stored alongside for comparison only).
5. Every clip reports `reward_sum, planned_steps, executed_steps, termination_reason, completed, physical_failure, finite`.
   `completed` requires the termination reason to be exactly `clip_end` and executed = planned; missing, NaN or
   unexpected timeouts count as failures; passing on n_failed=0 alone is not allowed.
6. Report separately: macro average, time-weighted average, completions, physical failures; whole clip, fixed 5 s
   windows and training reset starts are separate results; training coverage and holdout are separate; **no clip picking**.
7. Outputs: `results.json` (sha256 of checkpoint / protocol / clipset / npz, nominal read-back values, device and
   versions), `trajectories.npz` (qpos / qvel / reference qpos / clean action / executed action / reward / termination
   flags / per-step alive), `render_input.npz`, `mp4/`. Videos are rendered only from the real closed-loop states in
   `trajectories.npz`, never from reference motion or training logs.

Cross-dataset evaluation (e.g. BONES-trained weights on LAFAN) must say "cross-data diagnostic" in `data_label`.

**Protocol v2 (for new runs from 2026-11-06; v1 kept unchanged)**: `fullclip-nominal-v2` / `window10s-nominal-v2` add
pelvis and torso_link xy / yaw / height errors and the mean position error of 14 tracked bodies (world frame and
root-relative). Purpose: separate "metric definition differences" (repositories use different root/anchor bodies) from
"policy differences". Alive-only averages must be listed next to executed_steps to avoid survivor bias.

---

## 7. sim2sim evaluation (P5: rules first, implementation later)

- Definition: the same checkpoint runs **closed loop** in **MuJoCo Warp (the training simulator)** and in **plain
  MuJoCo on CPU (the `mujoco` package, a different solver implementation)**. Replaying or rendering a recorded
  trajectory in the other simulator **does not count**.
- Alignment items go into `configs/protocols/sim2sim-mujoco-v1.json`: MJCF / asset sha256, timestep 0.005, decimation 4,
  solver and iterations, PD gains and actuator limits, observation construction and history layout, action scale /
  default pose, friction and contact parameters, initial state.
- First **open-loop consistency**: per-step and multi-step state error between the two simulators for the same action
  sequence (proves the interface is aligned); then closed-loop evaluation with the same metrics and denominator as §6.
- Report completion / return / physical failures on both sides and the differences; attribute differences to interface
  alignment first, then to policy robustness.

---

## 8. Pod session (infrastructure) management

Each paid session = `artifacts/infra_sessions/<session_id>/`:

| Step | Tool / file | Pass condition |
|---|---|---|
| 0 reserve budget | `artifacts/p0/budget-ledger.json` | tier remainder ≥ price cap × deadline duration |
| 1 read-only check | `infra/runpod/account_check.py` | correct account, no RUNNING Pod for this task, live price |
| 2 bundle | `bundle/` + `SHA256SUMS` | source = `git archive` of the committed variant commit; data = verified processed data; no credentials |
| 3 create | `infra/runpod/session_pod.py create` | price ≤ cap, deadline ≤ 120 min, price × duration ≤ tier cap → write `session.json` |
| 4 local watchdog | `infra/runpod/budget_watchdog.py --mode local` | continue only after `authentication_verified` is logged |
| 5 on-Pod watchdog | same, `--mode pod` (Pod-scoped key) | bootstrap only after both watchdogs pass |
| 6 execute | `bundle/run_remote.sh` | every step has a timeout; log at `out/workload.log` |
| 7 recover | scp to local → verify sha256 | done before the archive cutoff; otherwise list what was not recovered |
| 8 stop | `session_pod.py stop` + independent `status` | `desiredStatus=EXITED` and `runtime=null`, write `stop-evidence.json` |
| 9 bill | billing query → ledger | record actual charges; null if not yet billed, fill in later |
| 10 archive | local → S3 `supermimic/projects/...` → full read-back sha256 | `.complete.json` receipt |

### 8.1 What "deadline" and "artifacts" mean (clarified by Harry, 2026-11-05)

- **The project has no progress deadline.** "Deadline" here is only a **cost fuse**: the longest billable time derived
  from "tier budget remaining ÷ actual price", so that a hung script, a dropped SSH connection or an interrupted
  session of mine cannot leave a Pod burning money unattended. It is not a schedule.
  - If the work is not done: recover state before stopping and continue in a **new session** (training via warm
    restart with the parent checkpoint recorded); never extend a single Pod.
  - The existing `budget_watchdog.py` limit of ≤ 7200 s per session is a second safeguard; if long training (tier D)
    needs longer sessions, change the guard, add tests and record it in decisions.md first.
- **"Artifacts" = everything needed to reproduce a run and everything it produced, not just checkpoints:**

| Category | Contents | Source and recovery |
|---|---|---|
| Code | source (variant commit + patch), `uv.lock`, our scripts / tools, bundle `SHA256SUMS` | **The local machine is authoritative**: the Pod only runs a bundle packed from a committed commit; code is not edited on the Pod; if a temporary change is unavoidable, pull it back with `git diff` and commit it |
| Config | resolved config, protocol JSON, clipset, command argv, environment variable allow-list | written into the run directory at run time and recovered with it |
| Environment | image digest, driver / CUDA / torch / mujoco versions, `uv pip freeze`, GPU model | `runtime-versions.txt` generated at run time and recovered |
| Data | raw / processed manifests, processed file hashes (npz converted on the Pod are recovered too) | recovered and checked against local manifests |
| Training output | checkpoints (light / full), `checkpoints.json`, tensorboard, stdout / stderr, metrics, resource samples | recovered incrementally |
| Evaluation output | results.json, trajectories.npz, render_input.npz, MP4 | recovered |
| Session evidence | session.json, both watchdog logs, bootstrap log, stop / EXITED evidence, billing | kept on both sides |

Recovery priority when time is short: session / stop evidence and logs → latest full checkpoint → evaluation results and
trajectories → other checkpoints → videos and tensorboard.

At the absolute deadline the Pod stops unconditionally; uploads do not extend it. `max-hours` and training timeouts do
not stop billing and are not safeguards. REST v1 is retired on 2026-11-15: our create / stop / watchdog use GraphQL
only; every session still tests stop once.

---

## 9. Per-run gate (training / inference / evaluation / rendering / sim2sim)

Before the run (if any item fails, do not run):

- [ ] `project_id` and `variant` are explicit; source is a committed commit (base commit + patch sha256 + uv.lock sha256 recorded)
- [ ] Data references a `processed_verified` manifest and a clipset hash
- [ ] Training: resolved config and the deviation table are written; evaluation: protocol JSON frozen and hashed
- [ ] Checkpoints (if any) have registered sha256 and are loaded with `weights_only=True`
- [ ] The output directory is a fresh ID that overwrites nothing
- [ ] GPU runs: budget reserved, deadline fixed, both watchdogs verified

After the run (if any item is missing, the run is `incomplete_evidence` and may not be called a success):

- [ ] Exit code, logs, `metrics.json` (unmeasured = null)
- [ ] `artifacts.json` with every sha256
- [ ] Evaluation: per-clip results + trajectories + MP4 when needed; denominator and termination as in §6
- [ ] `report.md`: conclusion, comparison with thresholds, limitations; failures too
- [ ] Recovered locally with hashes verified; GPU session stopped and EXITED confirmed independently; ledger updated
- [ ] `artifacts/reports/index.json` and `docs/status.md` updated

"Success" only means reaching a threshold written beforehand (e.g. 4/4 completed and macro return ≥ .80; 40/40
completed, 0 physical failures, macro ≥ .82). `process_succeeded`, an existing checkpoint, a falling loss or a non-zero
gradient are not success.

---

## 10. Status and to-do (2026-11-05)

| Component | Status |
|---|---|
| Workspace protection list, budget ledger, read-only checks | done: `artifacts/p0/` |
| Cuboi done-label fix + boundary tests | done: `variants/cuboi-g1/corrected-baseline` (8e58409), CPU 4/4 pass; real mjlab retest pending GPU |
| Whole-clip trajectory evaluation script | done (223b488), first GPU run pending |
| Pod create / stop tool | done: `infra/runpod/session_pod.py` (not run, awaiting authorization) |
| Two watchdogs | done: `infra/runpod/budget_watchdog.py` (GraphQL) |
| Legacy execution chain (new_run / run_pipeline / artifact_store / session_report) | exists, but tied to Cuboi paths and legacy S3 prefixes |
| **To build 1**: `scripts/run_gate.py`, running the §9 checks automatically; every entry point goes through it | next |
| **To build 2**: generic run registry per project_id + publishing to the new S3 prefixes (replacing Cuboi hard-coding) | next |
| **To build 3**: frozen `configs/protocols/*.json` + clipset manifest generator | before tier A |
| **To build 4**: incremental checkpoint recovery and the `checkpoints.json` registrar | before tier C training |
| **To build 5**: closed-loop MuJoCo CPU sim2sim adapter | P5 |

---

## 11. Multiple reference repositories: what is authoritative, and how they help

**Authority (conflicts resolved top-down)**

1. **The paper** (arXiv 2610.28480v1 §3, Fig. 4, Appendix A–F) defines the **method contract**: k8 / H32, 93-d
   proprioceptive history × 10, K10 future reference, latent 32, default-pose absolute setpoints, Table A.4 / A.5
   hyper-parameters. Items the paper does not disclose (full state / goal encoding, normalization, UTD counting
   details, ...) are written down as our "implementation conventions"; **third-party defaults are never taken as
   paper requirements**.
2. **Our main implementation = `cuboi-g1` / `corrected-baseline`** (CuboiLeo/SuperMimic-Reproduced @ ba61053 + our
   patches). It is the only scored object for "paper-contract reproduction": the environment runs, its observation is
   closest to the paper, and released weights exist for comparison. Its pinned source defines what "this implementation
   actually does".
3. **Real run reports** define what we measured. README numbers or author CSVs from any repository are third-party
   evidence and enter conclusions only after being remeasured under our protocol.

**Two acceptance goals, scored separately**

| Line | Scored object | Completion criterion |
|---|---|---|
| Paper contract | `cuboi-g1` corrected → paper-contract H32 from scratch | 40 clips: 40/40 completed, 0 physical failures, macro ≥ .82 (our threshold) |
| Enhanced system | `instanttrack-g1` (v1.3 native, separate project) | 40-clip evaluation under the same protocol + video; recorded as an "enhanced system result", **does not replace** the paper-contract result |

**Roles of other repositories (supporting only, not merged into the baseline)**

| Repository | Role | How | When |
|---|---|---|---|
| vibed-g1 (zita-ch/supermimic_vibed) | independent second implementation (IsaacLab / Newton) | if the main line underperforms, **cross-check**: WM predictions, gradients and reward terms on identical inputs to localize bugs; first test its end-bin boundary | P3 |
| ODE Policy (rl-supermimic branch) | actor ablation on the same Cuboi base | `cuboi-g1/ablation-ode-actor` variant, only the actor changes (same information, same clamp source), compared with the MLP | P3 |
| goyoungster-g1 | diagnostics + PPO comparison | verify COM constant recomputation, PPO ratio ≈ 1 before the first update; a non-BPTT reference | P3 |
| allmimic-g1 | library of single-variable improvement hypotheses | residual, latent128, share of noise-free collection, gradient scale, cluster sampling: **one at a time**, each its own variant | P3 |
| BlinkMimic | predecessor / module reference | absorbed by InstantTrack; not run separately by default; only rerun to split InstantTrack's gains | as needed |
| Uni-SuperMimic | full BONES library and multi-backend candidate | complete the run_config of its released weights before re-evaluating; reference for BONES filtering / sharded evaluation | P4 / P5 |
| ∇Mimic | another gradient source (simulator gradients, no learned WM) | borrow gradient diagnostics and cross-implementation closed-loop evaluation; not paid for yet | when budget allows |
| duskmon-go2 | terrain research reference | read only; no source or weights; no fabricated reproduction | later extension |

**Rules for combining**

- Borrowing = a **single-variable variant** on the main line (own commit, patch hash, config hash), compared with
  corrected-baseline under the same protocol and seed budget; only effective changes join the next baseline, with
  evidence recorded in [decisions.md](decisions.md).
- Checkpoints, processed caches and venvs are never shared across repositories; cross-repository comparisons use only
  numbers remeasured under the §6 protocol, not README rankings.
- When repositories conflict (done labels, bin boundaries, COM recomputation, ...), keep both sides' evidence, decide
  with the shortest real trace or a unit test, then decide which side to fix.

---

## 12. Scope of this round (Harry, 2026-11-05)

- **LAFAN1 only (the paper's LAFAN40 / 2.45 h comparison, §4.1, Fig. 4); no BONES training, P4 removed from this
  round.** Optional: once Harry accepts the BONES-SEED license, evaluate the released weights on a few dozen BONES clips
  to check our evaluation pipeline against the authors' per-clip numbers (≈ $0.1, tool validation only).
- sim2sim (train in MuJoCo Warp → closed-loop evaluation in plain MuJoCo) stays, mostly CPU cost.
- Paper-contract training scale (4096 envs, H32, WM 2048 windows) is not reduced to save compute; any reduction is a
  deviation and does not count as a formal reproduction.
- **Confirmed (Harry, 2026-11-05 22:44 UTC)**: whole-clip and fixed 10 s window metrics are frozen and reported
  separately; paper observations and +anchor are different variants scored separately. Background: long LAFAN clips
  may saturate under paper observations because of yaw drift (Cuboi reports whole clip ≈ 0.58, 0.904 with the +9D anchor).

### 12.1 Implementation convention: root and anchor bodies (checked against Cuboi ba61053, 2026-11-06)

| Use | Body | Code |
|---|---|---|
| floating base / qpos[0:7] / drift metric | pelvis | `ROOT_BODY`, tracking_env_cfg.py:66 |
| reward `motion_global_root_pos/ori` (named root, actually the anchor) | torso_link | `ANCHOR_BODY`, tracking_env_cfg.py:65; calls mjlab `motion_global_anchor_*_error_exp` |
| terminations `root_height` (0.25) / `root_tilt` (0.8) | pelvis (reference root vs robot pelvis) | terminations.py |
| yaw alignment for relative body pose rewards | torso_link | commands.py `update_relative_body_poses` |
| optional `anchor_obs` (9D) | torso_link | tracking_env_cfg.py |

vibed's root reward explicitly uses the pelvis (different meaning). The paper only says "root" and does not name the
body. The paper-contract version follows Cuboi; evaluation records pelvis and torso xy / yaw / height errors separately;
unify bodies before cross-repository comparisons.

### 12.2 Root drift under paper observations is a property of the observation (diagnosed 2026-11-06)

Report: reports/2026-11-06-walk1-drift-diagnosis.md; tool
`tools/offline/drift_timeseries.py` (offline FK on archived closed-loop trajectories, no GPU).

- Training and evaluation / render settings match item by item (observation, anchor, 50 Hz, PD / action scale, start
  frame = reference frame F state with target F+1); no alignment bug.
- Yaw is not observable and xy / yaw are not in the termination conditions: the authors' weights (tier A) drift 2.1°/s
  on walk1, tier C 1.9°/s; InstantTrack (tier B, observation includes the root pose relative to the reference) 0.00°/s.
  Long-clip whole returns therefore saturate around 0.55–0.6 (about 40% of the reward depends on global quantities).
  BONES clips are short (median 7.9 s), so it does not show there.
- Tier C's poor joint tracking (walk1 arms raised at 0–2.46 s, arm error 0.48 rad vs ≈ 0.09 for A / B) is a matter of
  training amount and sampling: absolute-target actions + a rare pose (arms raised in 3.3% of walk1 frames).
- Rule: long clips on the paper-contract line use `window10s` as the main acceptance; whole-clip results must also report
  the xy error at 10 s and the first time yaw error exceeds 15° (diagnostic only, no termination), noting "yaw is not observable".

### 4.1 Registering multi-session warm restarts (from tiers C / D)

- Each session = a new run (`runs/cuboi-g1/<run_id>`, run_name suffix `-sN`), started with `sm-train --resume <parent
  ckpt_latest.pt>`; the manifest records `parent_run_id`, `parent_checkpoint_path`, `parent_checkpoint_sha256` and the
  parent's iteration / num_updates.
- Cuboi `TrainLoop.load` restores wm / policy / critic weights, the three optimizers, sampler `bin_return/bin_memory`,
  iteration and num_updates; it **does not restore** the replay buffer (refilled each session, updates start after 40
  warm-up iterations), RNG, simulator / DR state or TensorBoard continuity. Hence "warm restart (not an exact resume)";
  the cosine LR schedule continues by iteration (total_iterations 320000 unchanged).
- Each session loses ~7.5 min to torch.compile and replay refill, counted in that session's cost and wall time; across
  sessions report "effective training time" and "total wall time" separately. Measured (D1 session 2, 2026-11-07): after
  a restart, train_ret and WM error drop noticeably within about 1500 iterations and then recover; evaluate checkpoints
  at least 2000 iterations after a restart. The same GPU model can differ 2× in throughput (4.9–5.7 vs 2.6–3.0 it/s);
  measure speed for 2 minutes after the Pod starts before deciding whether to switch machines.
- Evaluation: the last checkpoint of every session is evaluated with the v2 protocols; the initial of a formal run must
  be iteration 1 of that run's first session (never reuse a diagnostic run).
- Budget fields: the current `budget_watchdog.py` (v1) requires `authorized_total_budget_usd ≤ 15` and ≤ 7200 s per
  session; tier D sessions use that session's reservation (≤ 15), or, once Harry approves and it is recorded in
  decisions.md, `budget_watchdog_v2.py` (up to 14400 s per session, budget field ≤ 22, checks price × duration ≤ budget;
  tests `tests/infra/test_budget_watchdog_v2.py`, 5 passed).

---

## 13. Repository and branch conventions (2026-11-06)

- **Repository**: <https://github.com/GaohRyan/supermimic-g1-harry>, published from a private working repository. This file travels with the repository and
  remains the single source of truth.
- **Update 2026-11-11 (D039)**: after the official code was published it was confirmed that the actor only receives
  the reference orientation; `anchor-ori` joins the reproduction main line, full anchor is a deviation control.
- **Two main lines**: `repro` = strict paper-contract reproduction (`cuboi-g1/corrected-baseline`, base Cuboi @
  `ba61053`); `opt` = combined optimization (starting from `instanttrack-g1/native-v13`; +anchor and others each get a
  variant). Scored separately, see §11.
- **How variants are committed**: only `variants/<project>/<variant>.json` (base_repo, base_commit, variant_commit,
  sha256 of every patch) and `<variant>.patches/` (`git format-patch base..HEAD`). The working copy is a nested git
  repository and is not committed; rebuild it with `scripts/checkout_variant.sh`, and the rebuilt tree must equal
  variant_commit. When a variant gets new commits, re-export the patches and update the json.
- **Branches**: `main` only accepts PRs; prefixes `repro/`, `opt/`, `infra/`, `docs/`; squash merge, then delete the branch.
- **PR template**: `.github/pull_request_template.md`, fixed What / Why / How / Evidence sections, plain and short.
  Reproduction PRs state in How whether a deviation is introduced; Evidence lists test commands, run_id and protocol id,
  or "not tested".
- **Never committed**: `.secrets/`; data, checkpoints and videos under `data/`, `runs/`, `artifacts/`; LAFAN1 /
  BONES-SEED raw or derived files (including rendered MP4 / GIF; their licenses forbid redistribution); paper PDFs and
  full-text extracts; third-party snapshots under `upstream/` (except Cuboi, as a submodule). For traceability, commit
  only manifests (sha256 + source URL@revision).
- **Before committing**: `git status` shows none of the paths above; no secret values (credentials only in `.secrets/`
  and local configuration).

---

## 14. Run and reporting rules from 2026-11-10 (after the night review)

Basis: retrospective, decisions D031–D035.

**Starting Pods**
- Only through `infra/runpod/session_driver.py line ...`. It generates the script from the frozen config
  (`make_bundle.py`), reserves budget, starts the Pod, gates on speed, recovers, stops and chains to the next session.
  Hand-written or `sed`-edited scripts are never used.
- Speed gate: if the median of the first 5 it/s readings after training starts is below 2.8, stop and move to another
  machine (at most once per session). The same GPU model measured 2.0–5.3 it/s depending on the host CPU.
- Changing a training parameter = a new frozen config (flags written into `argv`), never an ad hoc command-line flag.
- Default remains one Pod at a time with a 115-minute fuse each. Concurrency or longer fuses are recorded in the ledger
  with who approved them.

**Fixed outputs of every training session** (automatic; failures do not block recovery)
- 10 s window evaluation at every checkpoint; whole-clip evaluation at the middle and final checkpoints.
- `ckpt_swa4.pt`: the weight average of the policy over the last 4 checkpoints, evaluated under both protocols.
- Push recovery (paper §4.3, upstream `eval_push.py`): final and average once each.

**Reporting**
- Result tables give both "single final checkpoint" and "weight average". At a high learning rate neighbouring
  checkpoints can differ by 1–3 failures, so a single one is not a conclusion.
- The main metric is normalized by planned steps (paper Appendix F: early termination scores low). When comparing with
  upstream numbers use `norm_return_survived_steps_upstream_style` (upstream normalizes by survived steps).
- Order of checks for a release candidate: nominal → `*-robust-v1` (sensor noise, domain randomization, reset noise) →
  held-out set (BONES40) → sim2sim → push. Protocol files are in `configs/protocols/`.

**sim2sim (implementation of §7)**
- On the Pod, export the contract: `scripts/export_sim2sim_contract.py` → `g1_nominal.mjb` + `contract.json`.
- Local CPU: `tools/sim2sim/sim2sim_mujoco_cpu.py`, three steps `validate` (reward formula reconciliation) → `openloop`
  (state deviation for the same action sequence) → `closed` (closed loop, 40 clips in about 7 minutes).
- Covers only "different simulator implementation", not model error (friction, mass, delay); that is the next step.

**Local disk**
- New sessions no longer keep `recovery.tar` (unpacked and verified while streaming); parent checkpoints in the bundle
  are APFS clones.
- Artifacts are not deleted. If space runs short, report it and let Harry decide what to clean.
