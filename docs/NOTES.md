# Notes: what we learned reproducing SuperMimic on the G1

Scope: motion tracking on LAFAN1 (40 clips, 2.45 h), first on flat ground (the setting of the paper's Fig. 4), then on
stairs up to 20 cm. Each idea is written as **What** (the question or change), **How** (what we ran), **Why** (the
reasoning behind the choice) and **Evidence** (numbers, with the decision ID in [decisions.md](decisions.md)).

Unless stated otherwise: seed 0, whole-clip evaluation, nominal environment, return normalized by planned steps,
"avg" = weight average of the last 4 checkpoints of a session. Single seed is the main limitation of everything below.

## Contributions at a glance

1. **A scoring and evaluation pipeline that can be trusted**: frozen protocols, planned-step normalization, nominal
   read-back, whole clip and window reported apart, closed-loop CPU sim2sim (Sections 1, 5).
2. **A diagnosis of the long-clip plateau**: heading is unobservable, so whole-clip return saturates; orientation alone
   fixes it, which the official code later confirmed (Section 2).
3. **A verified baseline**: our runs match the base repository's curve within ±0.03 once scoring is reconciled (Section 3).
4. **A cheap training recipe**: weight averaging removes the checkpoint noise that looked like a 37/40 ceiling (Section 4).
5. **Three measured flat-ground improvements**: randomization routed through the world model (delay 2 steps: 0/40 →
   38/40), world-model hard-example mining (fall-and-get-up 0.64 → 0.72), Muon on hidden layers (0.781 → 0.827)
   (Sections 6–8).
6. **Stairs with the same algorithm and flat references**: a height scan read by the world model at predicted poses,
   references lifted onto the terrain, a curriculum to 20 cm; 0.93 / 0.92 / 1.00 no-fall up / down / random 20 cm steps,
   0.00–0.03 when the scan is flattened, flat ground kept (Section 10).
7. **A list of the mistakes we made and the rule each one produced** (Section 12).

Contents

1. [How we score, and why it matters](#1-how-we-score-and-why-it-matters)
2. [Observation: heading drift and what the actor needs to see](#2-observation-heading-drift-and-what-the-actor-needs-to-see)
3. [Is the reproduction right? Reconciling with the base repository](#3-is-the-reproduction-right)
4. [Training recipe: checkpoint noise, weight averaging, annealing](#4-training-recipe)
5. [Robustness: sim2sim and a model-error sweep](#5-robustness-sim2sim-and-a-model-error-sweep)
6. [Opt 1: randomization the world model can see](#6-opt-1-randomization-the-world-model-can-see)
7. [Opt 2: the reset sampler and world-model hard examples](#7-opt-2-the-reset-sampler-and-world-model-hard-examples)
8. [Opt 3: Muon versus Adam and AdamW](#8-opt-3-muon-versus-adam-and-adamw)
9. [Reward and curriculum: what we did not change, and why](#9-reward-and-curriculum)
10. [Stairs: terrain level 1](#10-stairs-terrain-level-1)
11. [Infrastructure that saved money](#11-infrastructure-that-saved-money)
12. [Pitfalls](#12-pitfalls)
13. [Open questions](#13-open-questions)
14. [References](#14-references)

---

## 1. How we score, and why it matters

**What.** Fix the evaluation before training anything, so that results cannot drift with the code.

**How.** Frozen protocol files in [`configs/protocols/`](../configs/protocols/): every clip once from frame 0, a freshly
built nominal environment whose friction, CoM and encoder bias are read back and stored, no auto-reset (the terminal
state and the termination reason are saved), and return = Σ reward / (0.1 × planned steps). A clip counts as completed
only if it ends with `clip_end` after exactly the planned number of steps. Whole clips and fixed 10 s windows are
reported separately. Protocol v2 also records pelvis and torso position, yaw and height error and a 14-body mean error.

**Why.**
- Paper Appendix F says early termination should score low. The base repository divides by *survived* steps, which
  rewards falling early on a hard segment. We report planned-step returns as the main number and keep the
  survived-step number only to compare with other repositories.
- Turning off domain-randomization sampling is not the same as a nominal environment; reading the values back is the
  only way to know.
- Whole clips answer "does it get through 100–300 s?" and windows answer "how well does it track?". On long clips they
  can disagree a lot (Section 2), so mixing them hides the problem.

**Evidence.** The scoring mismatch once made us think we learned 0.03–0.11 slower than the base repository; under the
same scoring the gap is ±0.03 (Section 3). D004, D025, D033; protocol notes.

## 2. Observation: heading drift and what the actor needs to see

**What.** With the paper's literal observation (proprioception only, Table A.1), whole-clip return on LAFAN1 stops at
about 0.58 no matter how long we train. Is this a bug or a property of the observation?

**How.** An offline forward-kinematics tool ([`tools/offline/drift_timeseries.py`](../tools/offline/drift_timeseries.py))
reads archived closed-loop trajectories and computes error over time, without a GPU. We compared the base repository's
released weights, our own run and InstantTrack (whose observation includes the root pose relative to the reference).
Then we trained three observation variants on LAFAN40: none, full anchor (reference position + orientation relative
to the robot, as in BeyondMimic) and orientation only.

**Why.** Yaw is not observable from proprioception and is not in the termination condition, so a policy can drift
without ever falling. About 40% of the reward depends on world-frame quantities, so once the yaw error passes ~15° those
terms are lost. BONES-SEED clips are short (median under 8 s), which is why the paper's main results do not need
heading feedback. We picked orientation only as the candidate because a real robot gets orientation from its IMU,
while position needs global localization.

**Evidence.**
- Drift rate on walk1: released weights 2.1°/s, our run 1.9°/s, InstantTrack 0.00°/s. Training and evaluation settings
  matched item by item, so it is not an alignment bug (D027, drift diagnosis).
- LAFAN40 avg: proprioception only 39/40, 0.665 (23.3k updates); full anchor 40/40, 0.926 (24.8k); orientation only
  39/40, 0.820 after three sessions. Orientation only matches full anchor on everything except the root-position term.
  On BONES40 short clips all three are the same (0.70).
- When the [official code](https://github.com/HybridRobotics/supermimic) appeared it confirmed this: the actor sees joint
  targets plus the reference root orientation (rot6d, 10 future frames, sampled every 5 steps); relative position goes
  only to the world model. Our `official-contract` variant follows that layout (640-d goal, pelvis anchor) and is now the
  main reproduction: avg 40/40, 0.818 at 14.3k updates; with sensor noise 40/40, 0.807; plain MuJoCo CPU 40/40, 0.818;
  BONES40 held-out 36/40, 0.69–0.71. It ties the 6-d orientation variant and tracks heading better (2–3° vs 6° at the
  end of a window). D036, D039, D040.
- An earlier anchor audit overstated "the paper intends full anchor"; the base repository's 0.93 on BONES-SEED with
  proprioception only was the counter-example we had not weighed (anchor audit).

## 3. Is the reproduction right?

**What.** Our only change to the base repository's training code is a fix for an off-by-one in the terminal label
(the recorder wrote `done_next` on the current row while the window sampler read it from the next row). Did that, or
our data conversion and four warm restarts, change learning?

**How.** Read the survived-step return that every one of our `results.json` files also stores, and compare it with the
base repository's own logged curve (its experiment e3, same configuration) interpolated at the same iteration.

**Why.** Before tuning anything we wanted to know whether we were standing on a correct baseline; the alternative was
to keep explaining differences that might be bugs.

**Evidence.** Seven checkpoints from 12.5k to 43.3k iterations differ by −0.009 to +0.027. A second seed is within 0.01
of the first at the same stage. The released weights score 40/40, 0.870 on BONES40 (authors' number 0.871, step
counts identical per clip). The terminal-label fix therefore has no visible effect (D024, D034); a direct ablation was
not run. Retrospective E1, E9.

## 4. Training recipe

**What.** Our best LAFAN40 run sat at 37/40 for two sessions, and the failing clips were different every time.

**How.** Evaluate five neighbouring checkpoints, then the weight average of the last 4 and last 7; separately, anneal
the learning rate over one extra session from the same starting point.

**Why.** The cosine schedule is laid out over the paper's 320k iterations. Our budget reaches 13–20% of that, so the
learning rate never leaves its peak (0.93 of the initial value). High learning rate plus averaging makes faster
progress than lowering the rate early.

**Evidence.**

| Policy | Completed | Return |
|---|---:|---:|
| five single checkpoints, it 36k–43.3k | 36–38/40 | 0.839–0.856 |
| average of last 4 | 39/40 | 0.904 |
| average of last 7 | 40/40 | 0.905 |
| one annealed session, single checkpoint | 40/40 | 0.913 |

Every training session now writes `ckpt_swa4.pt` automatically
([`infra/runpod/pod_jobs/swa_policy.py`](../infra/runpod/pod_jobs/swa_policy.py)) and every result table reports both
the final checkpoint and the average. Averaging helps from the second session on; with sensor noise, a single
high-learning-rate checkpoint can drop from 37/40 to 31/40. D033, D037.

## 5. Robustness: sim2sim and a model-error sweep

**What.** Do the numbers survive a different simulator, and how much model error can the policy take?

**How.** Export the training model (`.mjb`) and an interface contract from the Pod, then run the policy closed-loop in
plain MuJoCo on CPU ([`tools/sim2sim/sim2sim_mujoco_cpu.py`](../tools/sim2sim/sim2sim_mujoco_cpu.py)): first check that
the reward formula matches, then open-loop state error on a fixed action sequence, then all 40 clips closed-loop
(about 7 minutes). The same tool sweeps action delay, motor strength, mass, foot friction and heading error.

**Why.** Replaying a recorded trajectory in another simulator proves nothing about the policy; only a closed loop does.
The sweep is free (CPU) and tells us which randomization to add before paying for training.

**Evidence** (official-contract avg, CPU, 40 clips):

| Perturbation | Completed | Return |
|---|---:|---:|
| none | 40/40 | 0.817 |
| action delay 1 / 2 / 3 steps (20 ms each) | 36 / 0 / 0 | 0.714 / 0.026 / 0.003 |
| motor torque ×0.9 / ×0.8 | 38 / 21 | 0.778 / 0.546 |
| mass ×1.1 / ×1.2 | 40 / 34 | 0.813 / 0.722 |
| foot friction 0.3 / 0.2 | 28 / 11 | 0.654 / 0.401 |
| heading bias 30°, or drift 1°/s | 40 / 40 | 0.679 / 0.559 |

Action delay is by far the weakest point: training had no delay randomization. Heading errors do not cause falls, which
is good news for an IMU-only robot. We also found that the exported model carried foot friction 0.6 instead of the
protocol's 1.0; rerunning at 1.0 gave the same result. Push recovery (paper Section 4.3): no fall below 2 m/s.

## 6. Opt 1: randomization the world model can see

**What.** Variant `sim2real-dr`: on top of the official contract, randomize action delay (0–2 steps), armature
(×0.7–1.5), PD gains (×0.85–1.15), joint damping and friction loss.

**How.** The world model is conditioned on a context vector ξ of simulator parameters; we extend it from 33 to 38 values
with the new scalars. Delay is handled differently: it is not fed to the world model. The replay buffer stores the
action that was actually executed, and during imagination the model receives the action from d steps earlier.

**Why.** The policy is trained only through the world model, so it can only become robust to things that change inside
imagination. A randomized parameter that the model cannot see becomes noise in its targets rather than something the
policy learns to handle. Mass and payload randomization are left for manipulation tasks, where they matter more.

**Evidence** (CPU sweep, session-3 avg, sim2real-dr vs official contract): no perturbation 40/40, 0.807 vs 0.817;
delay 1 / 2 / 3 steps 40 / 38 / 32 vs 36 / 0 / 0 completed; motor ×0.8 35/40 vs 21/40; mass +20% 37/40 vs 34/40. Foot
friction and heading error, which were not randomized, did not improve. Cost: 0.01 of nominal return. D041, D042,
sweep report.

## 7. Opt 2: the reset sampler and world-model hard examples

**What.** Fall-and-get-up clips stayed the weakest class. Two hypotheses: the policy does not practise them enough, or
the world model is least accurate exactly there.

**How.**
1. *Reset sampler.* The base sampler picks reset points by return-to-go per 1 s bin. We measured where resets landed.
   Variant `sampler-norm` normalizes return-to-go by the remaining horizon so that the end of a clip no longer looks hard.
2. *World-model error map* ([`tools/offline/wm_error_map.py`](../tools/offline/wm_error_map.py)): 8- and 32-step
   open-loop error of the world model, split by posture.
3. *Hard-example mining* (variant `wm-hard-mining`): half of each world-model batch is chosen from 4× as many
   candidates by one-step error.
4. As a control, twice as many world-model batches per update, without mining.

**Why.** Return-to-go is naturally small near the end of a clip and is not bootstrapped there, so the sampler confuses
"almost over" with "hard". For the world model, contact-rich low postures are rare in the data and are where the
imagined gradients are least trustworthy. A PPO tracker (InstantTrack) scored the same on fall-and-get-up as on its
other classes under the same body-only ruler (0.735 vs 0.74–0.77), so the gap was specific to the world-model method,
not a ceiling of the reference motions. We scored candidates by one-step error because scoring all 8 steps on 4×
candidates dropped throughput from 3.3 to 2.5 it/s for the same effect.

**Evidence.**
- 40.5% of resets fell in the last two bins of each clip, which hold 0.9% of the frames. After the fix the hardest bins
  are in the middle of clips. Same-stage return did not change measurably (38/40, 0.764 at 12.2k updates, on the line of
  the unfixed run), so we keep the fix as correct but do not count it as an improvement. D043, D044.
- Near the ground (reference root below 0.35 m, 5% of frames), 8-step world-model error is 10–13× that of standing.
- Mining, session 3 avg: 40/40, 0.781, fall-and-get-up 0.721; without mining 38/40, 0.764, 0.644. Mining already reached
  40/40 in session 2. Twice the batches: no gain and 17% slower, not adopted. D045, D046.
- Not explained: on a fixed set of trajectories the mined world model's near-ground error did not get smaller (0.232 vs
  0.192). The gain is real on the task metric, but the mechanism is not confirmed by this ruler.

## 8. Opt 3: Muon versus Adam and AdamW

**What.** The base repository uses Adam everywhere with a cosine schedule. The official code does not: hidden Linear
layers (between the first and last) use `torch.optim.Muon(lr, adjust_lr_fn="match_rms_adamw")`, everything else uses
Adam without decay, all at lr 2e-4, with no schedule inside the learners.

**How.**
1. Read the official source rather than secondary descriptions. One detail is easy to miss: PyTorch's `Muon` has
   `weight_decay=0.1` by default and the official code does not override it, so hidden weights *are* decayed.
2. Switch Adam to AdamW with weight decay 0 and test that it is a no-op: from scratch for 25 steps and resumed for 10
   steps, parameters differ by exactly 0.0.
3. Add a hidden-layer switch (`--wm-trainer.hidden-optimizer muon --policy-trainer.hidden-optimizer muon`) with the
   PyTorch default decay, so first/last layers and biases stay on AdamW without decay.
4. Write the adoption threshold before running: at equal updates, whole-clip return at least 0.03 higher, or reach the
   AdamW line's session-3 result one session earlier.
5. Train from scratch on the same base as Opt 2 and compare with that line.

**Why.** Muon orthogonalizes each update matrix, so no single direction dominates the step. For a policy trained by
backpropagating through 32 model steps, gradients are badly conditioned and occasionally spike; an update rule that
is insensitive to the gradient's scale is a plausible fit. The actor's Adam betas were already (0.9, 0.95) like the
official code, and our cosine rate stays above 95% of peak within our training length, so Muon was the only
optimizer difference left.

**Evidence** (LAFAN40, avg, both 40/40; curves for all lines in [the comparison figure](assets/opt-lines-lafan40.png), made by [`tools/offline/plot_line_comparison.py`](../tools/offline/plot_line_comparison.py)):

| Session | AdamW (Opt 2 line) | Muon |
|---|---:|---:|
| 1 (≈2.4k updates) | 0.372 | 0.481 |
| 2 | 0.748 | 0.805 |
| 3 | 0.781 | 0.827 |

Muon passes both thresholds: session 2 already beats AdamW's session 3. It costs about 10% more time per iteration.
Adopted (D049).

**Caveats.** One seed. We did not separate "Muon" from "weight decay 0.1 on hidden layers"; AdamW with the same decay
could explain part of the gain. A checkpoint trained with Adam cannot carry its optimizer state into the Muon layout, so
warm starts across the switch restart the optimizer.

## 9. Reward and curriculum

**What.** On flat ground we kept the BeyondMimic-style reward (8 exponential tracking terms) and the termination rules
(height error > 0.25 m or tilt > 0.8) unchanged, and the only curriculum is the return-based reset sampler (Section 7).

**Why.**
- The tracking reference already specifies what to do; every flat-ground failure we diagnosed was explained by
  observation (Section 2), evaluation noise (Section 4) or world-model accuracy (Section 7), not by reward shape.
  Changing rewards would also break the comparison with the paper and the base repository.
- The world model's reward head is analytic (computed through differentiable forward kinematics), so reward terms must
  be smooth functions of predicted state. Event-based terms such as feet air time are not differentiable and cannot be
  added without a learned surrogate.
- We did learn how the reward reacts to drift: about 40% of it depends on world-frame quantities, which is why heading
  drift caps whole-clip return instead of causing falls.

**Evidence.** Failure diagnosis on the 12 failing clips of an early LAFAN40 run: in the 3 s before each fall the
world model's 8-step error was 26× that of passing clips, and in 10 of 12 the reference was standing when the robot
fell, i.e. it lost track first and then collapsed (failure diagnosis).

## 10. Stairs: terrain level 1

**What.** Make the flat-ground tracker walk and run up and down 5–20 cm steps by perceiving the terrain, without
changing the algorithm (backpropagation through the world model) and without new reference motions.

**How.**
- *Terrain and curriculum.* Box terrain only: 50% flat tiles, 20% pyramid stairs up, 15% down, 15% stairs with random
  step heights; risers 5–20 cm in 10 levels, treads 30 cm. A level goes up when an episode does not fall, earns at least
  80% of the reward the same motion segment earns on flat tiles, and covers at least 70% of the reference's distance; it
  goes down on a fall in the first half of the episode. Non-flat tiles only get walk / run / sprint segments.
- *References.* The flat LAFAN1 references are lifted onto the terrain inside the command term: the root and upper body by
  the mean terrain height under the body (0.6 × 0.4 m), each foot by the terrain under its reference foothold, knees in
  between. Spawn points are raised until no foot sample penetrates. One implementation is shared by the simulator and
  imagination.
- *Perception.* A pelvis-centred 17 × 11 height scan at 0.1 m (187 points). The world model and critic get the noise-free
  scan, looked up from the height map at the *predicted* pose at every imagined step; the terrain is never predicted. The
  actor gets the same scan with ±3 cm noise. The world model's root-height input becomes height above the cell under the root.
- *Reward.* Simulator-side rewards are unchanged mjlab tracking terms with the lifted reference. Two terms are added in
  imagination only: a foot-clearance term and a progress term (reference velocity · root velocity, clipped), the latter
  counted only where the scan shows more than 2 cm of relief. No feet-air-time term.
- *Warm start.* From the Opt 3 (Muon) session-3 checkpoint: new input columns start at zero and optimizer moments are
  carried over, so step 0 behaves exactly like the flat policy.
- *Local verification first.* A MuJoCo C environment built from the robot model exported by the training simulator plus
  mjlab-generated terrain drives the variant's own command term, recorder, terminations and mjlab reward functions,
  so the terrain code was checked on a laptop before any GPU time ([`tools/terrain/`](../tools/terrain/)).

**Why.**
- *Same algorithm.* Other world-model trackers had already done perceptive terrain with it, and our checkpoints and tools
  carried over. PPO stayed a fallback that was never needed.
- *Flat references.* Others had placed raw flat motions on 20 cm steps successfully; generating terrain-specific
  references would have been a project of its own. The cost is that flat walking lifts the swing foot only 7–8 cm (median),
  which is why the clearance term exists.
- *Scan for the world model, not "a patch under each foot".* 37% of contact points in a reference implementation were
  behind the pelvis; 17 × 11 covers 99.8%. The world model always uses ground truth from the map, so changing the actor's
  sensor later (e.g. to depth) does not require retraining it.
- *Raw scan for the actor.* A frozen autoencoder to 32 dims blurred step edges by about one cell (3–3.7 cm error on 20 cm
  steps); the raw scan costs under 1% extra compute per iteration.
- *Progress term instead of a lag-based termination.* A robot that marches in place while matching posture and heading
  scores 0.84–0.87 of the maximum on walking clips, more than a policy that really walks (~0.80), so tracking reward alone
  cannot stop "waiting in front of the step". Our actor sees no position and drifts ~2 m on flat ground, so terminating on
  horizontal lag would fire on normal drift.
- *Not differentiating the height lookup horizontally*, and *no feet-air-time*: neither gives a useful gradient at a step edge
  or at a contact event.
- *Climb-aware success.* An evaluation window only counts as climbed if the robot's height change reaches 70% of the
  reference path's; otherwise standing at the bottom would count as success.

**Evidence** (delivered policy = weight average of the final session; 10 s windows on walk/run segments, 128 per cell,
MuJoCo Warp, nominal; single training seed unless noted):

| 20 cm tiles | No fall | Climbed | | 5 / 10 / 15 cm, up: no fall / climbed |
|---|---:|---:|---|---|
| Up | 0.93 | 0.90 | | 0.99 / 0.98, 0.99 / 0.98, 0.98 / 0.95 |
| Down | 0.92 | 0.79 | | |
| Random step heights | 1.00 | 0.98 | | |
| Flat tiles | 1.00 | — | | |

Before training, the flat policy did not fall on 20 cm stairs up in only 0.08 of windows (down 0.27, random 0.46).

- *Is the scan used?* Giving the actor a flat scan drops every stair tile to 0.00–0.03 no-fall; flat tiles stay at 1.00.
  Scaling the scan's relief by 0.9 or 1.1 (a 20 cm step seen as 18 or 22 cm) leaves 0.88–0.98.
- *With the training perturbations* (delay, actuator and joint randomization, noise), two evaluation seeds: 20 cm up
  0.88 / 0.86 no fall, down 0.95 / 0.87, random 0.98 / 1.00; 15 cm and below 0.94–1.00.
- *Another physics engine* (MuJoCo C, 48 per cell): 20 cm up 0.96 / 0.90, down 0.96 / 0.83, random 0.98 / 0.98; within
  sampling error of Warp.
- *Flat ground kept*: LAFAN40 whole clips 40/40, 0.824 (starting point 0.827); with perturbations 40/40, 0.789.
- *How we got there*: the first line (three sessions) met the stair criteria but flat return fell to 0.780. A control
  without the progress term recovered flat return by 0.025 but lost about 0.2 of climbed rate. The final line counts
  progress only on uneven terrain and doubles the sampling weight of non-climbing clips on flat tiles; it kept both.
  The two changes were not separated. Terrain cost: $9.27. D047–D050.

**Limitations.** One training seed. Climbing down 20 cm is the weakest cell (0.75–0.89 climbed): the robot does not fall
but can stop halfway. Slopes and rough heightfield tiles are off, because foot contact on heightfields differed from boxes
in MuJoCo C (flat policy fell in 25–50% of windows on a flat heightfield vs 0–4% on boxes). No latent-versus-raw scan comparison.

**Caveat: the height scan is idealized (sim-to-real gap not closed).** In simulation the actor receives all 187 scan
points every step, including the cells directly under the body, with only ±3 cm of uniform noise. A real robot does not
get that: a head-mounted LiDAR cannot see under the torso or right next to the feet (self-occlusion and minimum range),
so those cells are either stale values remembered by a local elevation map (seen a few steps earlier, delayed, and
drifting with odometry) or have no reading at all. The scan dots under the robot in our demo videos exist only because
of this idealization. Before hardware, the scan model has to come from the robot, i.e. real-to-sim:
- measure the sensor on the real G1: per-cell noise, dropout, latency, and which cells are blind for the actual mounting;
- train with a per-cell validity mask (invalid cells filled with a constant) and with the measured noise, delay and map drift;
- add an evaluation with the under-body cells masked.

None of the stairs numbers above cover these conditions. Only the actor's input changes; the world model and critic
keep using ground truth from the height map.

**Pitfalls specific to this step.**
- Picking the most stair-like clip in a dataset (35 cm then 70 cm) as the first target, instead of a curriculum from 5 cm.
- Changing the world model's root-height input from "cell under the root" to "mean under the body" on reasoning alone; on
  recorded stair data the mean was 4× worse. Measure on recorded data before choosing.
- Taking link velocity with `mjOBJ_BODY` (centre of mass) instead of `mjOBJ_XBODY` (link origin) in the local environment.
- A test helper where `state` and `hist_states` shared memory, so the lift was applied twice and imagined trajectories
  looked broken.
- Writing an estimate ("85–90% of flat reward while standing still") into the plan before measuring it; the measured
  numbers (0.55 walking, 0.25 running, 0.84–0.87 marching in place) reversed a design default.

## 11. Infrastructure that saved money

- **Frozen configs only.** A run script is generated from a frozen config; 150 s after start the run asserts its own
  `config.json`. A changed training parameter means a new config file, never a command-line edit. (Pitfall 1.)
- **Speed gate.** Throughput is set by the host CPU, not the GPU model: the same GPU ran at 1.9–4.8 it/s, 73% of the
  time is the policy update. If the median of the first 5 readings is below 2.8 it/s, the driver stops the Pod and moves
  to another one ([`infra/runpod/session_driver.py`](../infra/runpod/session_driver.py)).
- **Bounded sessions.** One paid Pod at a time, a hard 115-minute cost fuse, a local and an on-Pod watchdog, and an
  independent check that the Pod is `EXITED` after every session. Long training is a chain of warm restarts.
- **Evidence by default.** Each session automatically writes window and whole-clip evaluations, the weight average,
  push tests and a one-line summary per session.
- Total cloud cost for everything above: about $78.

## 12. Pitfalls

| # | What went wrong | Cost | Rule now |
|---|---|---|---|
| 1 | A run labelled "anchor" had the anchor off: the script was edited with `sed` and a multi-line command was missed | $1.61 and a day of wrong conclusions | scripts are generated from frozen configs; runs assert their config |
| 2 | Reported a single final checkpoint as the result | a 40/40 policy reported as 37/40 | always report final and weight average |
| 3 | Compared numbers with another repository under a different normalization | a false "we are slower" | reconcile scoring before comparing |
| 4 | Repeated a drift diagnosis the base repository had already published | about $5 and two days | read upstream experiment logs as results, not background |
| 5 | Treated the terminal-label fix as obviously correct | an open risk for a week | indirect check done (Section 3); direct ablation still open |
| 6 | Assumed the same GPU model means the same speed | 30–45% fewer updates per dollar | speed gate |
| 7 | Quoted a third-party description of the official optimizer | a wrong statement, later corrected | grep the official source; check defaults in the class that is called, not only the call site |
| 8 | Exported CPU model had foot friction 0.6, not 1.0 | none (same result), but a silent mismatch | the sim2sim tool sets and records friction explicitly |
| 9 | Scored 8 steps for 4× mining candidates | 3.3 → 2.5 it/s, stopped by the speed gate | estimate extra model calls per update before changing training |
| 10 | Warm restarts drop the replay buffer | about 1.5k iterations of regression after each restart | evaluate at least 2k iterations after a restart |

## 13. Open questions

- Stairs: a second training seed; separating the two changes of the final line; a latent-versus-raw scan comparison;
  slopes and rough terrain once heightfield contact is checked in MuJoCo Warp.
- Stairs on hardware: a real-to-sim scan model (measured noise, blind cells, latency) and retraining with a validity
  mask; see the caveat in Section 10.
- Multiple seeds for every row of the results table.
- Muon versus weight decay alone (Section 8).
- The remaining official-contract differences on the world-model side (listed in the variant's `not_aligned` field).
- Training on BONES-SEED, the paper's main dataset; we only evaluate on 40 held-out BONES clips.
- Why mining helps although the world model's near-ground error on fixed trajectories does not shrink.
- A direct ablation of the terminal-label fix.

## 14. References

Core references, in the order we leaned on them:

- **SuperMimic** (method): Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
  Model*, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480). Official code:
  [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) (read at `5efbf11`; observation layout and
  optimizers in Sections 2 and 8).
- **SuperMimic-Reproduced** by Yuxi Liu: [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced),
  pinned at `ba61053`. Our base implementation; its `docs/EXPERIMENTS.md` already reported the drift plateau.
- **InstantTrack**: [Huy-Quang-Dao/instantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (v1.3), our PPO
  comparison under the same ruler (Sections 2 and 7). It includes code from
  [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) and [BlinkMimic](https://github.com/fuyichen0725/blinkmimic).
- **supermimic_vibed** (Chong Zhang): [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), a second
  independent implementation used for cross-checking anchor and root conventions.
- **allmimic**: [Shimingyi/allmimic](https://github.com/Shimingyi/allmimic), a source of single-variable ideas
  (residual actions, wider latent, noise-free collection).
- **supermimic-reproduce**: [Go-Youngster/supermimic-reproduce](https://github.com/Go-Youngster/supermimic-reproduce),
  used for diagnostics and as a non-BPTT reference.
- **BeyondMimic**, whose anchor-relative tracking interface the paper follows (Appendix A).
- **InstinctMJ** ([project-instinct/InstinctMJ](https://github.com/project-instinct/InstinctMJ)), its SuperMimic and ACP
  branches: the source of several stairs decisions (full scan for the world model, scan coverage behind the pelvis, no
  horizontal gradient through the height lookup, climb-aware success). Its code is CC BY-NC; we took ideas only, no code.
- **legged_gym** ([leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)) for the stair-curriculum
  parameters, and **unitree_rl_gym** ([unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym))
  for the G1 swing-height term that motivated foot clearance.
- **mjlab** ([mujocolab/mjlab](https://github.com/mujocolab/mjlab), arXiv:2601.22074) and **MuJoCo / MuJoCo Warp**.
- **Data**: LAFAN1 (Harvey et al., SIGGRAPH 2020; G1 retargeting by
  [lvhaidong](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)) and
  [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed). Neither may be redistributed.
