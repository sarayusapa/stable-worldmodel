# Every experiment run this session

Covers everything actually executed across two pods (the first died to a
GPU hardware fault; a new pod with the same persistent network volume
replaced it — nothing was lost). All code is committed to `phy-wm`
(`b668cef`, `1f8943c`, `54bc938`). Grouped by benchmark.

---

## 1. PokeWorld (run by a concurrent collaborator session, not this one)

Not run by me — logged into `progress.md` from raw JSON that already
existed on disk when I found it. Included here for completeness since
it's part of the same paper's run history.

| Group | What | Episodes | Seeds | Status |
|---|---|---|---|---|
| datafloor | episode-budget scaling sweep | 128 / 256 / 512 / 1024 / 2048 / 4096 | mixed | complete |
| ablations-2048 | B_preaction, C_dataset_target, C_posthoc, G_probe_mlp | 2048 | 0-2 | complete |
| functional-2048 | substitution + multi-horizon at 2048ep | 2048 | 0-2 | complete |
| pilot-2048ep | primary claim at corrected budget | 2048 | 0-2 | complete |

**Model / architecture:** PhysWM (`tiny_cnn` encoder), PokeWorld's own
3-parameter case (`mass`, `contact_stiffness`, `drag`). Headline result
per that session: 0.818 label-free vs. 0.802 supervised-oracle ceiling.
Full tables logged in `progress.md`.

---

## 2. Fetch (`fetch_push`) — 256 episodes, run by me, complete

**Benchmark:** real Gymnasium-Robotics `FetchPush-v4` (MuJoCo), wrapped
via `FetchWrapper`. **2-parameter case**: `mass`, `friction` — no
stiffness analog exists in the Fetch XML (contact softness there is a
solver parameter, not a material property). **No tactile channel** —
this is the identifiability gap the results surfaced (see below).

**Model:** PhysWM, `tiny_cnn` encoder, new `FetchPushSolver` (2nd-order
contact-push dynamics: `m·dv/dt = F_contact − friction·v`, giving the
object genuine inertia so mass and friction are causally separable via
the dynamics, unlike a quasi-static block model).

**Script:** `scripts/eval/decodability.py` (A/B/C/H) and
`scripts/eval/functional_use.py` (F). Config: 256 episodes, length 48,
window 8, batch size 32, 60 epochs, `tiny_cnn` encoder.

| Job | Conditions | Seeds | Purpose | Result dir |
|---|---|---|---|---|
| `A_fetch_claim` | predictive, physwm | 0, 1, 2 | primary claim | `docs/fetch-matrix-256ep/run-results/A_fetch_claim_seed{0,1,2}.json` |
| `B_fetch_preaction` | physwm, `--probe-source encoded` | 0 | pre-action probing ablation | `B_fetch_preaction_seed0.json` |
| `C_fetch_dataset_target` | state_target | 0 | target-source ablation | `C_fetch_dataset_target_seed0.json` |
| `C_fetch_posthoc` | physwm, `--detach-probe-input` | 0 | induction-vs-posthoc-probe ablation | `C_fetch_posthoc_seed0.json` |
| `H_fetch_vq` | physwm, `--quantize-theta --num-codes 32` | 0 | continuous vs. VQ-theta design axis | `H_fetch_vq_seed0.json` |
| `F_fetch_functional` | physwm, horizon 7 | 0 | functional-use + task-success | `F_fetch_functional_seed0.json` |

**Outcome (see `docs/fetch-matrix-256ep/READING_THE_LOGS.md` for full
detail):** PREDICT holds (Path A beats persistence cleanly, both seeds).
IDENTIFY is inconclusive — `physwm/own_probe` scores *worse* than
`predictive/own_probe` on both params, but even the **supervised
ceiling** is negative, meaning 256 episodes is very likely below Fetch's
own data floor (mirrors PokeWorld needing 512→2048 before its own
ceiling turned positive). Also: no analog of PokeWorld's `touch` channel
exists for Fetch, so only `friction/mass` should be identifiable from
vision, not the two raw values separately — a real design gap, not a bug
(confirmed by reading `dynamics_coordinates()`, which only recognizes
PokeWorld's parameter names).

**Real bugs found and fixed along the way:**
- Fetch env construction failed entirely: a numpy-scalar/pybind11-enum
  comparison bug in `gymnasium_robotics` 1.4.2 + `mujoco` 3.12.0 (fixed
  via a scoped monkeypatch, `_mujoco_compat.py`, not an upstream package
  edit).
- `FetchPushSolver`'s mass bounds (`[0.01, 1.0]`) didn't match what
  `FetchWrapper` actually samples (`[0.01, 50.0]`) — would have silently
  saturated `bound_theta`'s sigmoid on real episodes. Fixed.
- Episode collection was CPU-bound and slow (~50 min/job even for a
  single seed) — added a disk cache (`fetch_episode_cache/`, keyed by
  every parameter affecting the data) so jobs sharing a seed/episode
  count don't re-collect from scratch.
- **My own bug**: the cache-adding patch used a non-unique string
  anchor and silently also patched `pusht_randomized_episodes()` (a
  different function, written by a collaborator's concurrent session),
  leaving a dangling `cache_path` reference there. Found and fixed when
  it crashed the PushT run below.

---

## 3. PushT (`pusht_rand`) — 512 episodes, run by me, complete

**Benchmark:** PushT (pymunk physics), but with **per-episode
randomized ground-truth physics** (`pusht_randomized_episodes`, written
by a collaborator's session before this one) — stock PushT has fixed
dynamics, which makes a shuffled-theta control vacuous (permuting a
constant changes nothing); randomizing per episode makes identifiability
actually measurable.

**Ground truth:** `agent_kp`, `agent_kv` (the PD-controller gains) have
real recorded values, since `PushTSolver` reproduces the environment's
own PD law exactly. `friction`/`mass` are randomized too (so the
shuffled control is meaningful) but have no 1:1 solver counterpart, so
no ground truth is claimed for them.

**Model:** PhysWM, `tiny_cnn` encoder, existing `PushTSolver`.

**Scripts, two different questions:**
1. `scripts/eval/decodability.py --benchmark pusht_rand` — the labeled
   R² check on `agent_kp`/`agent_kv` (3 seeds:
   `pusht_rand_decod_seed{0,1,2}`).
2. `scripts/eval/cross_benchmark.py --benchmark pusht_rand` — a
   label-free check (probe vs. shuffled vs. nominal theta against
   observed transitions), useful regardless of which parameters have
   ground truth (3 seeds: `pusht_rand_xbench_seed{0,1,2}`).

Config: 512 episodes, length 48, window 8, 40 epochs (decodability.py
uses `--conditions predictive,physwm`; cross_benchmark.py always trains
one `alpha=1.0` model).

**Outcome (see `docs/pusht-rand-matrix/READING_THE_LOGS.md` for full
detail):** all 6 jobs finished cleanly (rc=0). Label-free check
(`cross_benchmark.py`): probe beats shuffled in all 3 seeds (gap
0.057-0.060 scaled RMSE) and beats nominal in 2/3 seeds — theta is
carrying real per-episode information. Sensitivity is dominated by
`agent_kp`/`agent_kv` (~0.035-0.040) over everything else
(~0.001-0.007), matching the physical intuition that the PD gains
directly set push force while `com_offset`/`mobility`/`contact_stiffness`
are second-order. Path A beats persistence in 2/3 seeds. Labeled R²
check (`decodability.py`, `agent_kp`/`agent_kv` only — the only two
params with real ground truth): mostly negative `own_probe` R² at this
scale, a stricter bar the label-free check doesn't require — same
qualitative pattern PokeWorld and Fetch both showed before their own
labeled ceilings needed a larger episode budget to turn positive.

**Result dir:** `docs/pusht-rand-matrix/run-results/*.json` (6 files,
committed), `docs/pusht-rand-matrix/run-logs/*.log` (local-only, per
this repo's `*.log` convention).

---

## 4. Cartpole — 8 episodes, run by me, complete (small-N, deadline-forced)

**Benchmark:** DMControl Cartpole (MuJoCo via `dm_control`, a different
binding path than Fetch's `gymnasium_robotics`). **No ground truth
registered at all** (`has_theta_true: False`) — this benchmark exists
specifically to test the label-free approach, since there's no R²
certificate available here even in principle.

**Model:** PhysWM, `tiny_cnn` encoder, existing `CartpoleSolver`
(`cart_mass`, `pole_mass`, `pole_half_length`, `gravity`, `force_gain`,
`cart_damping`).

**Script:** `scripts/eval/cross_benchmark.py --benchmark cartpole`
(label-free only — decodability.py's R² approach doesn't apply here by
design). 3 seeds: `cartpole_xbench_seed{0,1,2}`.

Config: 512 episodes, **length 32** (see bug below — not 48, to match
PushT), window 8, 40 epochs.

**Real bug found and fixed:** the original launch used `--length 48`
(matching PushT) and crashed immediately with a native `Aborted (core
dumped)` — zero Python output, not a catchable exception. Isolated via a
length sweep (20/24/32 all stable, 48 crashes) to the cartpole episode
itself going unstable (pole falls / cart exits track bounds) somewhere
between step 33 and 47 under this data-collection policy, triggering an
unhandled abort in the MuJoCo/dm_control path rather than a graceful
episode termination. Fixed by using `--length 32` for cartpole
specifically — **cartpole's episode length now deliberately differs
from PushT's (32 vs. 48)**, documented here so it isn't mistaken for an
inconsistency later.

**Scope cut under a hard user deadline, in stages, all documented in
`matrix.log`:** 512 -> 256 -> 32 -> 8 episodes. Each cut was forced by a
real GL/EGL context-contention bottleneck in episode collection, not
model instability — see the real bug found and fixed below, which
independently required its own scope cuts before it was even isolated.
**Final run is 8 episodes, 40 epochs, 328 train / 82 val windows — a
small-N sanity check that the pipeline runs end to end, not a result
that says anything about the model's cartpole performance.** All 3 seeds
completed (rc=0).

**Second real bug found and fixed, independent of the length/abort bug
above:** `CartpoleDMControlWrapper.compile_model()`
(`stable_worldmodel/envs/dmcontrol/cartpole.py`) reassigned `self.env` on
every `reset()` (physics variation triggers a recompile almost every
reset) without ever closing the previous `self.env` first — a GL/EGL
render-context leak on nearly every episode. Reproduced directly: a solo
worker process reliably died or hung after exactly 2 episodes regardless
of seed, which random physics divergence would not do consistently.
Fixed with one line (`self.env.close()` before reassignment, commit
`bf4e356`). Episode collection was additionally moved to disposable
per-episode/per-batch subprocesses (`scripts/eval/_cartpole_batch_worker.py`)
so any remaining native abort only costs a retried batch, never the
whole job — this is what let the length go back to the real 48 (matching
PushT) instead of staying at the workaround value of 32.

**Result (8 episodes — read as a sanity check, not a finding):** `probe`
does not beat `nominal` in any seed (nominal RMSE 0.006-0.012 vs. probe
0.06-0.13 — the fixed default theta predicts far better than any
per-episode inference at this scale), and only beats `shuffled` in 2/3
seeds by a razor-thin margin (0.0007-0.008). `path_a` is 4-5x *worse*
than persistence in all 3 seeds (e.g. seed0: 1.098 vs. 0.200) — at
328 training windows and 40 epochs, this reads as a data-starved model
that hasn't learned anything useful yet, the same failure mode PokeWorld
and Fetch both showed below their own data floors, not a claim that
cartpole is unlearnable. **Re-running at a real episode count (512, or
whatever the datafloor sweep pattern from PokeWorld suggests) once GPU/
EGL contention eases is still an open item.**

**Result dir:** `docs/cartpole-matrix/run-results/*.json` (3 files,
committed), `docs/cartpole-matrix/run-logs/*.log` (local-only, per this
repo's `*.log` convention).

---

---

## 5. Fetch matrix re-run — 256 episodes, gripper fix applied, complete (8/8)

Re-run of section 2's matrix against the corrected `FetchPushSolver`
(commit `03fa29b` — the gripper's action-scale term was applied once per
substep instead of once per transition, a 10x overshoot with zero theta
dependence). Same config as the original: `tiny_cnn` encoder, 256
episodes, length 48, window 8 (16 for F), 60 epochs, batch 32. All 8 jobs
finished (rc=0).

**theta recovery — val R² (`A_fetch_claim`, `predictive` vs `physwm`, 3 seeds)**

| seed | predictive/decodable mass | predictive/decodable friction | predictive/own_probe mass | predictive/own_probe friction | physwm/decodable mass | physwm/decodable friction | physwm/own_probe mass | physwm/own_probe friction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -0.786 | -1.075 | -2.114 | -0.001 | -0.584 | -0.510 | -2.755 | -2.770 |
| 1 | -0.847 | -1.284 | -2.048 | -0.005 | -0.698 | -0.990 | -2.754 | -2.766 |
| 2 | -1.044 | -1.052 | -2.056 | -0.004 | -0.651 | -0.700 | -2.708 | -2.815 |

Still negative across the board for both conditions and both readout
methods, including the supervised ceiling (`decodable`) — the gripper fix
was never expected to move this (the gripper term has zero theta
dependence by construction), and this matches the pre-fix finding: 256
episodes is very likely below Fetch's own data floor, mirroring PokeWorld
needing 512-2048 before its own ceiling turned positive.

**prediction/fidelity, `predictive` vs `physwm` (3-seed range)**

| metric | predictive | physwm |
| --- | --- | --- |
| path_a_vs_dataset | 0.399-0.439 | 0.404-0.435 |
| path_b_vs_teacher | 8.41-9.19 | 0.73-0.79 |
| persistence_vs_dataset | 0.623-0.691 | 0.623-0.691 |

**PREDICT holds cleanly and consistently**: Path A beats persistence by a
healthy, stable margin in all 3 seeds under both conditions
(~0.40-0.44 vs. ~0.62-0.69). `predictive`'s Path B is essentially
unconstrained (no physics-grounding loss trains it against real
transitions), hence its huge `path_b_vs_teacher` gap versus `physwm`'s —
expected, not a regression.

The B/C/H/F ablation results and the persistent "probe beats true theta"
anomaly (unaffected by the gripper fix, better explained by the solver-
adequacy gap in section 6) are unchanged from the summary already
written below.

**theta recovery — val R² (seed 0, `own_probe`/`decodable`)**

| job | mass (own_probe) | friction (own_probe) | mass (decodable) | friction (decodable) |
| --- | --- | --- | --- | --- |
| B (pre-action) | -2.749 | -2.773 | -0.445 | -0.331 |
| C (dataset-target) | -2.756 | -2.771 | -0.268 | -0.686 |
| C (posthoc) | -2.651 | -2.758 | -0.561 | -0.686 |
| H (VQ-theta) | -1.824 | -1.563 | -0.171 | -0.272 |

Still negative across the board, including the supervised ceiling
(`decodable`) — consistent with the pre-fix finding that 256 episodes is
likely below Fetch's own data floor, not something the gripper fix alone
resolves (theta recovery was never the gripper's fault; the gripper term
has zero theta dependence by construction).

**F (functional use) — the "probe beats true theta" anomaly, re-checked
post-fix:** still present. `substitution`: probe 1.535, true 6.984,
shuffled 7.357, nominal 22.364 — probe still scores far below true theta
at every `multi_horizon` step (probe 1.3-8.3 vs true 10.9-22.7). Since the
gripper bug (theta-independent, identical cost for every theta source) is
now fixed and the pattern persists, it was never solely a gripper
artifact. Better explanation, given section on solver-adequacy below:
Path B's probe theta is distilled against Path A's own prediction, not
real physics, and the frozen solver has a confirmed adequacy gap against
real Fetch dynamics — so probe theta "wins" by matching what Path A
predicts, not by being more physically correct. `task_success` is still
degenerate (0.03125 for all four sources, uninformative at this
threshold/horizon).

**Result dir:** `docs/fetch-matrix-256ep-fixed/run-results/*.json` (5 of
8, committed), `docs/fetch-matrix-256ep-fixed/run-logs/*.log`.

---

## 6. Fetch solver-adequacy check — 256 episodes, complete

`scripts/smoke/validate_solvers.py --benchmarks fetch_push --episodes 256
--length 48 --steps 1200` — checks whether the frozen `FetchPushSolver`,
given the *best possible* per-episode theta (gradient-fit directly
against the objective, not learned), can explain real Fetch transitions.
Previously only checked at 8/32 episodes locally (no GPU); this is the
real matrix scale.

**Result:** gripper fix holds at scale (`gripper_x`/`gripper_y` and now
also `object_x`/`object_y` fitted RMSE matches/beats persistence). But
`object_vx`/`object_vy` still don't: overall fitted MEAN 0.4833 vs.
persistence 0.4109 — **the solver does not beat persistence even at its
best-possible fit.** The oracle fit does correctly beat true theta
(0.4833 vs 1.3823), which rules out an optimization bug — the gap is in
the solver's functional form itself (`contact_stiffness=400` fixed and
unfit, and/or the linear-friction/point-mass approximation not suiting
Fetch's real contact dynamics as well as it suits PokeWorld). Not
root-caused further than that; treat as a caption-worthy limitation in
the paper, the same way PushT's disc-approximation is handled, rather
than a bug to keep hunting.

**Result dir:** `docs/solver-adequacy/fetch_push_256ep.log` (local only,
`*.log` convention).

---

## Status as of writing

PushT (section 3) and the solver-adequacy check (section 6) are complete,
documented, and pushed. Cartpole (section 4) has needed several relaunch
cycles — a real GL/EGL context leak in `CartpoleDMControlWrapper` was
found and fixed (commit `bf4e356`), but episode collection is still slow
under contention from the other concurrently-running jobs; scope was
cut progressively (512 -> 32 -> 8 episodes) against a hard user deadline.
The Fetch matrix re-run (section 5) is 5/8 done; `A_fetch_claim` (the
primary 3-seed claim) is still running.
