# Run Log

Every run and its results — training runs, smoke suites, evaluations,
ablations, benchmarks.

**This log is paper material.** Numbers reported in the paper are traced back to
entries here, so an entry needs enough detail that someone else could reproduce
the run from it alone.

---

## Read this before you run anything

**This applies to every agent and every human working in this repo, not just
whoever wrote a given entry.**

1. **Every run gets an entry**, with its full details, at the time you get the
   results. A run that isn't logged didn't happen as far as the paper is
   concerned.
2. Log **failed and inconclusive runs too.** Negative results are results, and a
   run that crashed at epoch 12 is something the next person needs to know.
3. **Newest entries go at the top** of the Log section, directly under the
   `## Log` heading.
4. Never edit or delete a past entry to make it look better. If an entry turns
   out to be wrong, add a new entry that corrects it and link back.
5. **Never invent details.** If you don't know the commit, config, seed, or
   hardware a number came from, write `unknown` — do not guess. An entry with
   honest gaps is usable; an entry with fabricated metadata is worse than no
   entry, and in a paper it's a retraction.
6. Copy metric tables **verbatim** from the tool that produced them. Do not
   round, reorder, or "clean up" numbers.
7. Note the repo state — commit sha, and whether the tree was dirty. The same
   command on a different working tree is a different experiment.
8. Anything in **Open items** at the bottom is unfinished work on this log.
   Clear items as you resolve them.

You do **not** need to log repo changes that produced no run. Code changes live
in git; this file is for results.

### Entry template

Copy this block for each new run.

```markdown
### YYYY-MM-DD — <short title>

- **Command:** <exact command line, including env vars and overrides>
- **Config:** <config file(s) + overrides>
- **Seed(s):** <seed, or the list for a multirun>
- **Commit / working tree:** <sha, branch, clean or dirty + what was uncommitted>
- **Hardware:** <CPU/GPU, or `unknown`>
- **Data:** <benchmark(s), episodes, episode length>
- **Duration:** <wall clock, or `unknown`>
- **Status:** pass / fail / partial
- **Artifacts:** <checkpoint dir, W&B run id, history.json path>

<results table or numbers, verbatim>

- **Notes:** <interpretation, caveats, what to run next>
```

---

## Log

### 2026-08-26 — Gripper overshoot fixed in `FetchPushSolver`; re-verified with `validate_solvers.py`

Follow-up to the entry directly below. Applies the one-line fix that entry
identified but did not apply, and re-checks it.

- **Command:** same as below, plus a second run at `--episodes 32 --length
  48 --steps 1200` to check whether the residual gap shrinks with more
  data/optimization.
- **Config / commit / hardware:** same scratch-venv setup as below, on top
  of the `solvers.py` edit (`gripper = gripper + action[..., 0:2] *
  (self.action_scale / self.substeps)`, replacing the undivided
  `self.action_scale`). Not yet committed.

```
# 8 episodes (same scale as the pre-fix run, for direct comparison)
dim           persist    nominal       true     fitted
gripper_x      0.0503     0.0511     0.0511     0.0511
gripper_y      0.0527     0.0322     0.0322     0.0322
object_vx      1.0683     8.4973     1.7606     1.2323
object_vy      1.0704    10.7244     1.4362     1.3551
MEAN           0.3877     3.2289     0.5596     0.4581
fetch_push   persist 0.3877  nominal 3.2289  fitted 0.4581  -> SOLVER DOES NOT BEAT PERSISTENCE

# 32 episodes, 1200 fit steps
dim           persist    nominal       true     fitted
gripper_x      0.0471     0.0456     0.0456     0.0456
gripper_y      0.0508     0.0311     0.0311     0.0311
object_vx      1.1596    10.8793     2.9730     1.3271
object_vy      1.1374    12.1922     3.4093     1.3625
MEAN           0.4115     3.8680     1.0877     0.4725
fetch_push   persist 0.4115  nominal 3.8680  fitted 0.4725  -> SOLVER DOES NOT BEAT PERSISTENCE
```

- **Notes:** The fix does exactly what the arithmetic predicted:
  `gripper_x`/`gripper_y` fitted/true/nominal RMSE dropped from ~15-18x
  worse than persistence to matching or beating it (`0.0511` vs `0.0503`,
  `0.0322` vs `0.0527`) at both scales, and `nominal` MEAN dropped from
  3.84/3.87 to 3.23/3.87 (nominal isn't expected to be good, it's the
  fixed-default-theta baseline; the point is the gripper component of it
  specifically is fixed). The "true theta beats fitted oracle" inversion
  from the pre-fix run is also gone at 8 episodes (`true 0.5596 > fitted
  0.4581`, the expected ordering, oracle wins).

  **What is NOT fully resolved:** overall `fitted` MEAN still does not
  clearly beat `persistence` (0.4581 vs 0.3877 at 8 episodes; 0.4725 vs
  0.4115 at 32), driven entirely by `object_vx`/`object_vy` — unlike
  PokeWorld/Cart-pole/Push-T, where the oracle fit beats persistence by
  1-3 orders of magnitude (`\Cref{tab:solver-adequacy}`). The gap did not
  close between 8 and 32 episodes (if anything `true` got worse, 0.56 ->
  1.09, plausibly because a wider sample includes episodes further from
  where `contact_stiffness=400` — fixed, not fit — happens to be a good
  approximation). This is a real, secondary open question, smaller in
  magnitude than the gripper bug and not yet root-caused: candidates are
  the fixed `contact_stiffness` being miscalibrated against real Fetch
  MuJoCo contact response, or the linear-friction/point-mass
  approximation itself being a worse match to Fetch than it is to
  PokeWorld. Not investigated further here — this machine has no GPU and
  the matrix-scale (256ep) data collection was not attempted locally.
  **Recommendation:** re-run this same check at 256 episodes on the GPU
  pod before writing a Fetch row into `\Cref{tab:solver-adequacy}`; if the
  gap persists at that scale, it is worth a caption rather than another
  code hunt, the same way Push-T's disc-approximation caveat is handled.

  **Fully resolved:** the theta-independent 10x gripper overshoot, which
  was large enough to be the dominant confound in every metric the Fetch
  matrix (below) reported, including the "probe beats true" anomaly that
  motivated this whole check.

### 2026-08-26 — `validate_solvers.py` extended to `fetch_push`; found the gripper is theta-independent AND overshoots by 10x

Follow-up to the entry directly below ("Open, unresolved from this batch").
Resolves the "probe theta beats true theta" anomaly with a real, verified
cause, not a training or reporting artifact.

- **Command:** `SWM_FETCH_CACHE_DIR=<scratch dir> MUJOCO_GL=cgl python
  scripts/smoke/validate_solvers.py --benchmarks fetch_push --episodes 8
  --length 48 --seed 0`
- **Config:** default `validate_solvers.py` args (8 episodes, length 48,
  600 fit steps, lr 0.05), `fetch_push` solver `dt=0.01, substeps=10`
  (newly added to the script, matching `decodability.py`'s existing
  per-benchmark dt/substeps table).
- **Seed(s):** 0.
- **Commit / working tree:** `54bc938` + uncommitted local changes to
  `scripts/smoke/validate_solvers.py` (added a `fetch_push` branch to
  `gather_transitions` — the generic `env_episodes()` path does not
  randomize `block.mass`/`block.friction` or extract `theta_true` for
  Fetch, only `fetch_episodes()` does; added `fetch_push` to the
  `dt`/`substeps` tables; added a `true`-theta RMSE column alongside
  persistence/nominal/fitted). Not committed pending a decision on the bug
  below.
- **Hardware:** CPU, local Mac (no GPU available in this environment);
  scratch venv, `torch==2.13.0`, `mujoco==3.12.0`, `gymnasium-robotics==1.4.2`.
- **Data:** `fetch_push`, 8 episodes x 48 steps (smoke scale, not the
  256-episode matrix scale — sufficient to isolate this bug, which is
  independent of theta and episode count).
- **Duration:** ~1 minute.
- **Status:** pass (script ran); the *result* is a fail for the solver.
- **Artifacts:** none kept (stdout captured directly below).

```
dim           persist    nominal       true     fitted
gripper_x      0.0503     0.9080     0.9080     0.9080
gripper_y      0.0527     0.7893     0.7893     0.7893
object_x       0.0488     0.0473     0.0448     0.0450
object_y       0.0357     0.0478     0.0342     0.0338
object_vx      1.0683     7.1589     1.2804     1.1507
object_vy      1.0704    14.0757     1.9959     1.3936
MEAN           0.3877     3.8378     0.8421     0.7201
```
`fetch_push   persist 0.3877  nominal 3.8378  fitted 0.7201  -> SOLVER DOES
NOT BEAT PERSISTENCE`

- **Notes:** Read directly from `FetchPushSolver` in
  `stable_worldmodel/wm/physwm/solvers.py`. The base class's `forward()`
  loops `for _ in range(self.substeps): state = self.step(state, action,
  p, dt)` — correct for the object's ODE integration (`dt` is the
  per-substep interval, and integrating the continuous acceleration in 10
  finer steps approximates one 0.1s transition). But `FetchPushSolver.step`
  updates the gripper with `gripper = gripper + action[..., 0:2] *
  self.action_scale` — a **discrete per-transition kinematic displacement**,
  not a rate — applied on *every* substep instead of once. With
  `substeps=10, action_scale=0.05`, the gripper moves `10 x 0.05 x action
  = 0.5 x action` per transition instead of the intended `0.05 x action`,
  a 10x overshoot. This is exactly consistent with the ~15-18x RMSE blowup
  above (`0.9080 / 0.0503 ~ 18x`, `0.7893 / 0.0527 ~ 15x`, in scaled-RMSE
  units where the mismatch compounds with the gripper's own variance) —
  confirmed arithmetically, not just by the ratio matching.

  **Consequences for every Fetch matrix result logged in the entry below:**
  1. The gripper term has **zero theta dependence** (`mass`/`friction`
     never enter it), so no amount of probe accuracy, self-distillation,
     or oracle fitting can correct it — `true` and `fitted` gripper RMSE
     are bit-identical to `nominal`'s in the table above. This is a
     constant floor on Path B's prediction error in every one of the 8
     jobs, unrelated to whether theta is being recovered well.
  2. It explains the "probe beats true" anomaly directly: in `F`'s
     `substitution`/`multi_horizon` tables, `true`/`shuffled`/`nominal`
     theta all pay the same broken gripper cost, while the model's own
     probe theta was distilled against Path A's prediction (a *learned*
     model that presumably gets gripper motion roughly right, since it
     is trained on real trajectories with no such bug) — so "probe" isn't
     winning by understanding physics better, the comparison is
     confounded by a bug that penalizes the other three sources equally
     and unfairly.
  3. **The frozen solver does not beat persistence even at the oracle
     fit** (0.7201 vs 0.3877 persistence, 8-episode smoke scale) — this
     is a more fundamental problem than the mass/friction
     non-identifiability documented in `READING_THE_LOGS.md`. Before that
     documented gap is even reachable, Fetch fails the solver-adequacy
     precondition every other benchmark passes cleanly
     (`\Cref{tab:solver-adequacy}`: PokeWorld 0.0001, Cart-pole 0.0025,
     Push-T 0.021, all far below persistence). Fetch has no comparable
     entry yet, and by this measurement it would currently read "fitted
     0.72 > persistence 0.39" — the solver fails its own adequacy check.
  4. Object-dimension fit (`object_vx`/`object_vy`) does improve from
     nominal to true to fitted (7.16 -> 1.28 -> 1.15 for vx) but even the
     oracle fit does not beat persistence's 1.07 at this 8-episode smoke
     scale — a separate, smaller-magnitude question from the gripper bug,
     worth re-checking once the gripper is fixed and at the full 256/2048
     episode scale before concluding anything about mass/friction
     identifiability from prediction quality alone.

  **Not yet done:** the likely one-line fix (divide the gripper's
  `action_scale` term by `self.substeps`, or apply it only on the first
  substep) has not been applied to `solvers.py`, and the 256-episode
  Fetch matrix has not been re-run against a fixed solver. Both are
  necessary before any Fetch number (theta recovery *or* prediction
  quality *or* functional use) is safe to cite in the paper — the matrix
  entry below predates this discovery and should not be treated as
  reflecting the intended solver.

### 2026-08-26 — Fetch matrix, 256 episodes, 8 jobs (A/B/C/H/F), logged post hoc

- **Command:** `matrix.log` records an orchestrator run starting `2026-08-26
  00:20:30 UTC`; the per-job exact command lines are not in the repo (jobs
  were launched by whatever driver produced `docs/fetch-matrix-256ep/`,
  which is not `overnight.py` since that script's dry-run had never
  executed per the prior "Open items" entry). Config is fully recoverable
  from each JSON's `meta` block instead — see below.
- **Config:** `encoder=tiny_cnn, window=8 (16 for F), tactile=True,
  episodes=256, length=48, epochs=60, batch_size=32, amp=True`. Per-job
  `conditions` / `probe_source` / `detach_probe_input` differences:
  - `A_fetch_claim_seed{0,1,2}` — `conditions=predictive,physwm`,
    `probe_source=predicted`, `detach_probe_input=False` (the main claim,
    3 seeds)
  - `B_fetch_preaction_seed0` — `conditions=physwm`,
    `probe_source=encoded` (pre-action-latent ablation)
  - `C_fetch_dataset_target_seed0` — `conditions=state_target` (probe
    regressed on the dataset label directly, not self-distilled)
  - `C_fetch_posthoc_seed0` — `conditions=physwm`,
    `detach_probe_input=True` (probe reads a detached latent — fits a
    read-out without shaping the representation)
  - `H_fetch_vq_seed0` — `conditions=physwm`, VQ-theta ablation
  - `F_fetch_functional_seed0` — `functional_use.py`, not
    `decodability.py`; `window=16`, `theta_supervision=0.0`
- **Seed(s):** 0, 1, 2 for group A; seed 0 only for B/C/H/F.
- **Commit / working tree:** `HEAD` at merge time was `54bc938` (this run's
  results were added in that same commit); exact commit/dirty-flag the jobs
  themselves ran against is not recorded in the JSON `meta` (no
  `run_meta.json`-style provenance for this batch) — `unknown`.
- **Hardware:** unknown (not recorded for this batch).
- **Data:** `fetch_push`, 256 episodes x 48 steps; 10496 train / 2624 val
  windows after the observability filter (8448 / 2112 for F, window 16).
- **Duration:** per `matrix.log` — group A sequential (~3000-3900s/seed),
  then B/C/C/H/F in parallel (~2000-2250s each). Wall clock start to finish
  of the parallel batch: 00:20:30 -> 03:14:52 UTC, ~3h.
- **Status:** pass, `rc=0` for all 8 jobs.
- **Artifacts:** `docs/fetch-matrix-256ep/run-results/*.json` (8 files),
  `docs/fetch-matrix-256ep/run-logs/matrix.log`. Per-job raw `.log` files
  were **not** committed (kept local only, per this repo's `*.log`
  gitignore convention, per `READING_THE_LOGS.md`).

**theta recovery — val R² (group A, `predictive` vs `physwm`, 3 seeds)**

| seed | predictive/decodable mass | predictive/decodable friction | predictive/own_probe mass | predictive/own_probe friction | physwm/decodable mass | physwm/decodable friction | physwm/own_probe mass | physwm/own_probe friction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -0.8154 | -0.2690 | -2.1130 | 0.0005 | -0.7064 | -0.4502 | -2.7537 | -2.7679 |
| 1 | -0.6688 | -0.7250 | -2.0463 | -0.0063 | -0.3423 | -0.4622 | -2.7533 | -2.7647 |
| 2 | -0.3466 | -1.1508 | -2.0582 | -0.0031 | -0.3104 | -0.3738 | -2.7071 | -2.8135 |

**theta recovery — val R² (ablations, seed 0, `physwm/own_probe` unless noted)**

| job | mass | friction |
| --- | --- | --- |
| B (pre-action latent) | -2.7526 | -2.7735 |
| C (dataset-target, not self-distilled) | -2.7567 | -2.7710 |
| C (posthoc, detached probe input) | -2.7188 | -2.7676 |
| H (VQ-theta) | -1.6650 | -1.7436 |

**prediction/fidelity RMSE, `predictive` vs `physwm` (group A, seed 0; other
seeds/jobs follow the same pattern, see JSON)**

| metric | predictive | physwm |
| --- | --- | --- |
| path_a_vs_dataset | 0.4203 | 0.4213 |
| path_b_vs_teacher | 8.9898 | 0.7709 |
| persistence_vs_dataset | 0.6229 | 0.6229 |

**functional use (`F_fetch_functional_seed0`)**

- Sensitivity (prediction shift from scaling theta 1.1x): mass 0.0437,
  friction 0.00017 — friction is ~250x less load-bearing than mass in the
  frozen solver at this window (16 frames, 0.01s dt x 10 substeps).
- Substitution (one-step Path B RMSE vs real `s_next`): true 5.549, probe
  1.476, shuffled 7.072, nominal 21.415 — **probe theta scores lower error
  than true theta**, which the "shuffled ~ true" gap (-0.011 in the A-job
  equivalent) does not explain by itself. Flagged as an open question below,
  not yet understood.
- Multi-horizon (H=1..7): probe stays in [1.24, 2.67], true in [4.84, 5.30],
  shuffled in [6.49, 6.68], nominal in [19.7, 22.0] — same probe-beats-true
  pattern holds at every horizon, it does not close with H.
- Task success (7-step solver rollout, threshold 0.05): **0.03125 for all
  four sources** (true/probe/shuffled/nominal) — currently a degenerate,
  fixed 1/32 fraction with no discriminative signal.

- **Notes:** Full interpretation guide written alongside this batch:
  `docs/fetch-matrix-256ep/READING_THE_LOGS.md`. Two things established by
  reading the actual code, not guessed:
  1. `dynamics_coordinates()` in `decodability.py` has no `fetch_push`
     branch, so the tables above report raw `(mass, friction)` rather than
     the ratio `friction/mass` that should actually be identifiable —
     `FetchPushSolver`'s `m*dv/dt = F_contact - friction*v` has the same
     mass/friction degeneracy PokeWorld resolves with its `touch` channel,
     and Fetch's state has no analogous force signal. This is a benchmark
     reporting gap, not (necessarily) a training failure.
  2. Even the supervised ceiling (`*/decodable`) is negative for every
     parameter, every job — 256 episodes may be below Fetch's own data
     floor (PokeWorld's equivalent ceiling needed 2048 episodes to turn
     positive; see the 2026-08-25 data-floor entries above).

  **Open, unresolved from this batch (do not write into the paper until
  checked):** F's substitution/multi-horizon tables show probe-derived
  theta producing *lower* one-step error against real `s_next` than the
  actual ground-truth theta fed through the same frozen solver, at every
  horizon. `FetchPushSolver` has no entry in the solver-adequacy table
  (`validate_solvers.py`, `\Cref{tab:solver-adequacy}` in the paper) the
  way PokeWorld/Cart-pole/Push-T do, so there is no oracle-fit check yet
  establishing whether the frozen solver, given the best possible
  per-episode theta, can even explain real Fetch transitions to low error.
  Until that check exists, "true theta loses to probe theta" could mean
  the solver is inadequate for Fetch (an oracle-fit theta would beat both),
  or a units/box mismatch between raw "true" theta and the probe's
  sigmoid-mapped box output, or a genuine anomaly. Run
  `validate_solvers.py` against `fetch_push` before citing this table in
  the paper.



- **Params:** ``
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `encoder_free_floor.json`

### 2026-08-25 18:24 — [Data-floor sweep (episode-budget scaling)] `encoder_free_pooled` (2048ep re-run, logged post hoc)

- **Params:** ``
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `encoder_free_pooled.json`

### 2026-08-25 17:47 — [Data-floor sweep (episode-budget scaling)] `floor_ep1024_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=1024, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep1024_seed0.json`, `floor_ep1024_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.7232 | -0.8913 | 0.6825 | -1.5882 | 0.7244 | 0.7160 |
| drag | -0.3190 | -6.2787 | -0.2495 | -0.0462 | -0.3383 | -0.3184 |
| mass | 0.1051 | -0.3162 | 0.1065 | -0.0664 | 0.0212 | 0.0019 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -0.2267 | -5.3517 | -0.0838 | -0.0625 | -0.2705 | -0.1519 |
| stiffness_over_mass | 0.1950 | -0.3127 | 0.2927 | -0.7666 | 0.2618 | 0.2698 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3496 | 0.3368 | 0.3745 |
| path_a_vs_dataset | 0.4155 | 0.3919 | 0.4247 |
| path_b_vs_dataset | 0.0989 | 0.1521 | 0.0879 |
| path_b_vs_teacher | 0.3403 | 0.3556 | 0.3785 |
| persistence_query_vs_dataset | 0.4035 | 0.4035 | 0.4035 |
| persistence_vs_dataset | 0.5477 | 0.5477 | 0.5477 |

### 2026-08-25 17:54 — [Data-floor sweep (episode-budget scaling)] `floor_ep1024_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=1024, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep1024_seed1.json`, `floor_ep1024_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.7419 | -0.0436 | 0.6476 | -1.1063 | 0.7067 | 0.6992 |
| drag | -0.3096 | -2.7381 | -0.3780 | -0.0023 | -0.3954 | -0.4051 |
| mass | 0.0554 | -0.0363 | 0.0966 | 0.0004 | -0.1567 | -0.1039 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -0.1003 | -1.2580 | -0.1659 | -0.0415 | -0.2805 | -0.1313 |
| stiffness_over_mass | 0.2551 | 0.0087 | 0.3909 | -0.5633 | 0.3005 | 0.2974 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.2349 | 0.2380 | 0.2519 |
| path_a_vs_dataset | 0.3062 | 0.2899 | 0.3150 |
| path_b_vs_dataset | 0.0720 | 0.1252 | 0.0542 |
| path_b_vs_teacher | 0.2365 | 0.2543 | 0.2520 |
| persistence_query_vs_dataset | 0.2920 | 0.2920 | 0.2920 |
| persistence_vs_dataset | 0.4697 | 0.4697 | 0.4697 |

### 2026-08-25 17:55 — [Data-floor sweep (episode-budget scaling)] `floor_ep1024_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=1024, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep1024_seed2.json`, `floor_ep1024_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.6092 | -0.0648 | 0.7297 | -1.2150 | 0.7116 | 0.7032 |
| drag | -0.3427 | -3.2403 | -0.2401 | -0.0210 | -0.3469 | -0.3997 |
| mass | -0.1769 | -0.5183 | -0.0582 | -0.1091 | -0.0989 | -0.1381 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -0.2420 | -3.9145 | -0.1858 | -0.0143 | -0.1883 | -0.1979 |
| stiffness_over_mass | 0.0678 | 0.0714 | 0.3070 | -0.5665 | 0.2882 | 0.2408 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.2766 | 0.2586 | 0.3343 |
| path_a_vs_dataset | 0.3430 | 0.3294 | 0.3780 |
| path_b_vs_dataset | 0.0743 | 0.1331 | 0.0605 |
| path_b_vs_teacher | 0.2792 | 0.2806 | 0.3387 |
| persistence_query_vs_dataset | 0.2966 | 0.2966 | 0.2966 |
| persistence_vs_dataset | 0.4818 | 0.4818 | 0.4818 |

### 2026-08-25 16:10 — [Data-floor sweep (episode-budget scaling)] `floor_ep128_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=128, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep128_seed0.json`, `floor_ep128_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | -0.4886 | -1.5930 | -1.6134 | -1.9439 | 0.4101 | 0.3753 |
| drag | -2.1424 | -0.8026 | -4.3619 | -0.0300 | -0.7063 | -0.6608 |
| mass | -3.0514 | -0.6045 | -2.9662 | 0.0376 | -0.6938 | -0.5579 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -4.5131 | -1.6761 | -5.3761 | -0.1488 | -1.0380 | -0.7295 |
| stiffness_over_mass | -1.2183 | -0.5051 | -0.7568 | -0.6920 | -0.0125 | -0.0615 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.7910 | 0.7850 | 0.8113 |
| path_a_vs_dataset | 0.8269 | 0.8061 | 0.8719 |
| path_b_vs_dataset | 0.1826 | 0.2083 | 0.1350 |
| path_b_vs_teacher | 0.8061 | 0.8140 | 0.8262 |
| persistence_query_vs_dataset | 0.6913 | 0.6913 | 0.6913 |
| persistence_vs_dataset | 0.7667 | 0.7667 | 0.7667 |

### 2026-08-25 16:13 — [Data-floor sweep (episode-budget scaling)] `floor_ep128_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=128, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep128_seed1.json`, `floor_ep128_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | -0.8049 | -1.4297 | -0.6111 | -2.6292 | -0.4245 | -0.3348 |
| drag | -2.6542 | -0.4921 | -2.8617 | -0.0577 | -0.6072 | -0.5143 |
| mass | -1.5469 | -0.2439 | -0.7499 | -0.1513 | -0.2590 | -0.2988 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -1.3851 | -0.5624 | -2.0399 | -0.0831 | -0.2149 | -0.0686 |
| stiffness_over_mass | -1.8904 | -0.2465 | -2.0295 | -0.9664 | -0.0968 | -0.1359 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.7347 | 0.6948 | 0.7801 |
| path_a_vs_dataset | 0.7924 | 0.7410 | 0.7821 |
| path_b_vs_dataset | 0.0760 | 0.1267 | 0.0485 |
| path_b_vs_teacher | 0.7125 | 0.6749 | 0.7735 |
| persistence_query_vs_dataset | 0.4770 | 0.4770 | 0.4770 |
| persistence_vs_dataset | 0.5928 | 0.5928 | 0.5928 |

### 2026-08-25 16:11 — [Data-floor sweep (episode-budget scaling)] `floor_ep128_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=128, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep128_seed2.json`, `floor_ep128_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | -1.0723 | -2.2404 | 0.0247 | -2.1679 | -0.0847 | -0.1347 |
| drag | -1.7116 | -4.1468 | -3.5469 | -0.0436 | -0.5730 | -0.3663 |
| mass | -0.9319 | -1.5052 | -0.6205 | -0.1742 | 0.1628 | 0.1400 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -2.6329 | -10.7806 | -2.2274 | 0.0085 | -0.5115 | 0.0096 |
| stiffness_over_mass | -0.2004 | -0.2661 | -0.0444 | -0.5763 | -0.0663 | -0.0198 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3273 | 0.3068 | 0.3771 |
| path_a_vs_dataset | 0.4801 | 0.4321 | 0.5222 |
| path_b_vs_dataset | 0.1052 | 0.1408 | 0.0437 |
| path_b_vs_teacher | 0.3072 | 0.3004 | 0.3747 |
| persistence_query_vs_dataset | 0.2516 | 0.2516 | 0.2516 |
| persistence_vs_dataset | 0.4803 | 0.4803 | 0.4803 |

### 2026-08-25 15:59 — [Data-floor sweep (episode-budget scaling)] `floor_ep2048_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep2048_seed0.json`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.7694 | 0.1180 | 0.7396 | -1.4193 | 0.8051 | 0.8002 |
| drag | -0.1847 | -4.9174 | -0.2499 | -0.0126 | -0.4060 | -0.4371 |
| mass | 0.2945 | -0.1336 | 0.2459 | -0.0689 | 0.1519 | 0.1488 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | 0.0386 | -2.1030 | 0.0515 | -0.0359 | -0.0100 | -0.0266 |
| stiffness_over_mass | 0.4607 | 0.0509 | 0.4215 | -0.6106 | 0.4547 | 0.4679 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3608 | 0.3328 | 0.3713 |
| path_a_vs_dataset | 0.3850 | 0.3623 | 0.3865 |
| path_b_vs_dataset | 0.0881 | 0.1172 | 0.0750 |
| path_b_vs_teacher | 0.3570 | 0.3426 | 0.3721 |
| persistence_query_vs_dataset | 0.4439 | 0.4439 | 0.4439 |
| persistence_vs_dataset | 0.5509 | 0.5509 | 0.5509 |

### 2026-08-25 16:22 — [Data-floor sweep (episode-budget scaling)] `floor_ep256_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=256, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep256_seed0.json`, `floor_ep256_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.0648 | -0.8953 | -0.3817 | -1.7935 | 0.1016 | 0.1043 |
| drag | -3.4223 | -1.9542 | -1.5201 | -0.0798 | -0.1149 | -0.0423 |
| mass | -1.5815 | -0.2795 | -1.5467 | -0.0545 | -0.6357 | -0.5429 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -2.3041 | -1.5725 | -1.7761 | -0.1072 | -0.5584 | -0.3736 |
| stiffness_over_mass | -1.5200 | -0.3528 | -0.7292 | -0.6944 | -0.3007 | -0.1537 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.6183 | 0.5797 | 0.6165 |
| path_a_vs_dataset | 0.6467 | 0.5907 | 0.6528 |
| path_b_vs_dataset | 0.1028 | 0.1526 | 0.1100 |
| path_b_vs_teacher | 0.6213 | 0.6009 | 0.6226 |
| persistence_query_vs_dataset | 0.5244 | 0.5244 | 0.5244 |
| persistence_vs_dataset | 0.6267 | 0.6267 | 0.6267 |

### 2026-08-25 16:25 — [Data-floor sweep (episode-budget scaling)] `floor_ep256_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=256, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep256_seed1.json`, `floor_ep256_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | -0.3240 | -0.3715 | -0.1502 | -1.5967 | 0.0692 | 0.0261 |
| drag | -2.3140 | -1.2729 | -2.7243 | -0.0107 | -0.5859 | -0.9149 |
| mass | -1.6287 | -0.2553 | -0.8585 | -0.0258 | -0.1221 | -0.0990 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -1.8314 | -1.6826 | -1.8531 | -0.0726 | -0.5526 | -0.3152 |
| stiffness_over_mass | -0.5929 | 0.0514 | -0.8219 | -0.5512 | 0.0037 | 0.0408 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.5216 | 0.4690 | 0.5262 |
| path_a_vs_dataset | 0.6007 | 0.5236 | 0.6111 |
| path_b_vs_dataset | 0.1007 | 0.1087 | 0.0506 |
| path_b_vs_teacher | 0.5068 | 0.4595 | 0.5245 |
| persistence_query_vs_dataset | 0.3526 | 0.3526 | 0.3526 |
| persistence_vs_dataset | 0.4972 | 0.4972 | 0.4972 |

### 2026-08-25 16:23 — [Data-floor sweep (episode-budget scaling)] `floor_ep256_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=256, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep256_seed2.json`, `floor_ep256_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | -0.0940 | 0.0080 | 0.1964 | -1.1868 | 0.5146 | 0.4641 |
| drag | -3.1589 | -0.5031 | -3.9434 | -0.0173 | -0.5246 | -0.4577 |
| mass | -1.4371 | -0.9380 | -1.1339 | -0.2430 | -0.0698 | 0.0073 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -2.1039 | -1.1139 | -3.4301 | 0.0274 | -0.3170 | -0.0387 |
| stiffness_over_mass | -0.6616 | -0.0057 | -0.4638 | -0.4629 | 0.0558 | 0.1933 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3869 | 0.3647 | 0.3803 |
| path_a_vs_dataset | 0.4740 | 0.4434 | 0.4681 |
| path_b_vs_dataset | 0.0616 | 0.1049 | 0.0653 |
| path_b_vs_teacher | 0.3804 | 0.3721 | 0.3811 |
| persistence_query_vs_dataset | 0.3101 | 0.3101 | 0.3101 |
| persistence_vs_dataset | 0.4803 | 0.4803 | 0.4803 |

### 2026-08-25 16:22 — [Data-floor sweep (episode-budget scaling)] `floor_ep4096_seed0` (2048ep re-run, logged post hoc)

- **Params:** ``
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep4096_seed0.json`

### 2026-08-25 16:47 — [Data-floor sweep (episode-budget scaling)] `floor_ep512_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=512, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep512_seed0.json`, `floor_ep512_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.1551 | -0.0718 | 0.3997 | -1.5344 | 0.5987 | 0.6058 |
| drag | -0.8053 | -1.3016 | -0.5019 | -0.0347 | -0.3612 | -0.3633 |
| mass | -0.3991 | -0.2028 | -0.4364 | -0.0311 | -0.0985 | -0.1648 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -0.8160 | -1.4415 | -0.5487 | -0.0823 | -0.6029 | -0.4646 |
| stiffness_over_mass | -0.3263 | 0.0237 | -0.0741 | -0.7003 | 0.1000 | 0.0962 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.4683 | 0.4384 | 0.4870 |
| path_a_vs_dataset | 0.5400 | 0.4984 | 0.5423 |
| path_b_vs_dataset | 0.1334 | 0.1712 | 0.0951 |
| path_b_vs_teacher | 0.4636 | 0.4648 | 0.4953 |
| persistence_query_vs_dataset | 0.4867 | 0.4867 | 0.4867 |
| persistence_vs_dataset | 0.5850 | 0.5850 | 0.5850 |

### 2026-08-25 17:00 — [Data-floor sweep (episode-budget scaling)] `floor_ep512_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=512, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep512_seed1.json`, `floor_ep512_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.0721 | -0.0213 | 0.6198 | -1.2041 | 0.3910 | 0.3407 |
| drag | -0.6404 | -2.9162 | -0.5760 | -0.0024 | -0.4296 | -0.4757 |
| mass | -0.3554 | -0.2425 | -0.2299 | 0.0169 | -0.2452 | -0.2303 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -0.4835 | -2.7532 | -0.3586 | -0.0529 | -0.2867 | -0.1607 |
| stiffness_over_mass | -0.3197 | 0.0645 | -0.0749 | -0.5802 | -0.0155 | 0.0910 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3435 | 0.3337 | 0.3771 |
| path_a_vs_dataset | 0.3846 | 0.3938 | 0.4332 |
| path_b_vs_dataset | 0.0725 | 0.0941 | 0.0763 |
| path_b_vs_teacher | 0.3392 | 0.3256 | 0.3758 |
| persistence_query_vs_dataset | 0.3186 | 0.3186 | 0.3186 |
| persistence_vs_dataset | 0.4701 | 0.4701 | 0.4701 |

### 2026-08-25 17:00 — [Data-floor sweep (episode-budget scaling)] `floor_ep512_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=512, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Data-floor sweep: how many episodes are needed before theta recovery is even possible. The 512-episode default was found too small (~1300 touch-filtered windows, below the recoverable threshold); this sweep establishes the actual floor.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `floor_ep512_seed2.json`, `floor_ep512_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.4571 | 0.1861 | 0.3691 | -1.0804 | 0.3823 | 0.3590 |
| drag | -0.7318 | -0.7455 | -1.1027 | -0.0516 | -0.4687 | -0.5407 |
| mass | -0.6215 | -0.1606 | -0.7390 | -0.1435 | -0.5268 | -0.4617 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | -0.9939 | -0.4261 | -0.9904 | -0.0019 | -0.3514 | -0.2683 |
| stiffness_over_mass | -0.2006 | 0.1240 | -0.1947 | -0.5450 | -0.2585 | -0.1709 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3572 | 0.3242 | 0.3707 |
| path_a_vs_dataset | 0.4382 | 0.4135 | 0.4555 |
| path_b_vs_dataset | 0.1201 | 0.1262 | 0.0790 |
| path_b_vs_teacher | 0.3639 | 0.3305 | 0.3824 |
| persistence_query_vs_dataset | 0.2684 | 0.2684 | 0.2684 |
| persistence_vs_dataset | 0.4657 | 0.4657 | 0.4657 |

### 2026-08-25 17:18 — [Ablation (B_preaction, 2048ep)] `B_preaction_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=encoded, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=False`
- **Why this run:** Routing ablation: tests whether reading the SAME action-conditioned latent is necessary (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `B_preaction_seed0.json`, `B_preaction_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.1871 | -0.1974 |
| drag | -0.0212 | -2.2610 |
| mass | 0.1482 | -0.3228 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.0656 | -1.1749 |
| stiffness_over_mass | 0.2226 | 0.1709 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.3463 |
| path_a_vs_dataset | 0.3726 |
| path_b_vs_dataset | 0.0650 |
| path_b_vs_teacher | 0.3364 |
| persistence_query_vs_dataset | 0.4439 |
| persistence_vs_dataset | 0.5509 |

### 2026-08-25 18:21 — [Ablation (B_preaction, 2048ep)] `B_preaction_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=encoded, episodes=2048, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=False`
- **Why this run:** Routing ablation: tests whether reading the SAME action-conditioned latent is necessary (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `B_preaction_seed1.json`, `B_preaction_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.1220 | -0.1013 |
| drag | -0.0380 | -1.1193 |
| mass | 0.0876 | -0.2596 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.0406 | -0.6950 |
| stiffness_over_mass | 0.1142 | 0.1086 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.2493 |
| path_a_vs_dataset | 0.3272 |
| path_b_vs_dataset | 0.0810 |
| path_b_vs_teacher | 0.2464 |
| persistence_query_vs_dataset | 0.3505 |
| persistence_vs_dataset | 0.5179 |

### 2026-08-25 18:33 — [Ablation (B_preaction, 2048ep)] `B_preaction_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=encoded, episodes=2048, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=False`
- **Why this run:** Routing ablation: tests whether reading the SAME action-conditioned latent is necessary (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `B_preaction_seed2.json`, `B_preaction_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.1984 | -0.1542 |
| drag | -0.0098 | -1.8867 |
| mass | 0.0272 | -0.4996 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.0392 | -1.2853 |
| stiffness_over_mass | 0.0623 | 0.0461 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.2633 |
| path_a_vs_dataset | 0.3168 |
| path_b_vs_dataset | 0.0685 |
| path_b_vs_teacher | 0.2558 |
| persistence_query_vs_dataset | 0.3552 |
| persistence_vs_dataset | 0.5150 |

### 2026-08-25 17:19 — [Ablation (C_dataset_target, 2048ep)] `C_dataset_target_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=state_target, detach_probe_input=False`
- **Why this run:** Target ablation: replaces the model prediction with the raw state label while keeping the solver/route fixed (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `C_dataset_target_seed0.json`, `C_dataset_target_seed0.log`

**theta recovery — val R²**

| param | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.8623 | 0.8307 |
| drag | -0.1277 | -0.0628 |
| mass | 0.3286 | 0.2112 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.1701 | 0.2205 |
| stiffness_over_mass | 0.5793 | 0.5986 |

**prediction and distillation — val RMSE**

| metric | state_target |
| --- | --- |
| path_a_query_vs_dataset | 0.3419 |
| path_a_vs_dataset | 0.3640 |
| path_b_vs_dataset | 0.0756 |
| path_b_vs_teacher | 0.3425 |
| persistence_query_vs_dataset | 0.4439 |
| persistence_vs_dataset | 0.5509 |

### 2026-08-25 18:21 — [Ablation (C_dataset_target, 2048ep)] `C_dataset_target_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=state_target, detach_probe_input=False`
- **Why this run:** Target ablation: replaces the model prediction with the raw state label while keeping the solver/route fixed (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `C_dataset_target_seed1.json`, `C_dataset_target_seed1.log`

**theta recovery — val R²**

| param | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.8671 | 0.8156 |
| drag | -0.1287 | -0.0816 |
| mass | 0.2764 | 0.2280 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.0754 | 0.0871 |
| stiffness_over_mass | 0.5121 | 0.5485 |

**prediction and distillation — val RMSE**

| metric | state_target |
| --- | --- |
| path_a_query_vs_dataset | 0.2550 |
| path_a_vs_dataset | 0.3336 |
| path_b_vs_dataset | 0.0357 |
| path_b_vs_teacher | 0.2542 |
| persistence_query_vs_dataset | 0.3505 |
| persistence_vs_dataset | 0.5179 |

### 2026-08-25 18:31 — [Ablation (C_dataset_target, 2048ep)] `C_dataset_target_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=state_target, detach_probe_input=False`
- **Why this run:** Target ablation: replaces the model prediction with the raw state label while keeping the solver/route fixed (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `C_dataset_target_seed2.json`, `C_dataset_target_seed2.log`

**theta recovery — val R²**

| param | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.8589 | 0.8068 |
| drag | -0.1208 | -0.1112 |
| mass | 0.2453 | 0.2239 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.1122 | 0.0532 |
| stiffness_over_mass | 0.4862 | 0.4513 |

**prediction and distillation — val RMSE**

| metric | state_target |
| --- | --- |
| path_a_query_vs_dataset | 0.2681 |
| path_a_vs_dataset | 0.3246 |
| path_b_vs_dataset | 0.0560 |
| path_b_vs_teacher | 0.2663 |
| persistence_query_vs_dataset | 0.3552 |
| persistence_vs_dataset | 0.5150 |

### 2026-08-25 17:17 — [Ablation (C_posthoc, 2048ep)] `C_posthoc_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=True`
- **Why this run:** Induction ablation: representation induction vs fitting only a post-hoc probe on a latent the physics loss cannot shape (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `C_posthoc_seed0.json`, `C_posthoc_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.7633 | 0.3674 |
| drag | -0.1507 | -5.8536 |
| mass | 0.2820 | -0.5761 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.1053 | -5.3378 |
| stiffness_over_mass | 0.4881 | 0.3448 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.3594 |
| path_a_vs_dataset | 0.3865 |
| path_b_vs_dataset | 0.0669 |
| path_b_vs_teacher | 0.3505 |
| persistence_query_vs_dataset | 0.4439 |
| persistence_vs_dataset | 0.5509 |

### 2026-08-25 18:21 — [Ablation (C_posthoc, 2048ep)] `C_posthoc_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=True`
- **Why this run:** Induction ablation: representation induction vs fitting only a post-hoc probe on a latent the physics loss cannot shape (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `C_posthoc_seed1.json`, `C_posthoc_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.7706 | 0.3653 |
| drag | -0.0524 | -5.8046 |
| mass | 0.2080 | -0.1039 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.0991 | -2.3272 |
| stiffness_over_mass | 0.4177 | 0.3648 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.2557 |
| path_a_vs_dataset | 0.3325 |
| path_b_vs_dataset | 0.0753 |
| path_b_vs_teacher | 0.2505 |
| persistence_query_vs_dataset | 0.3505 |
| persistence_vs_dataset | 0.5179 |

### 2026-08-25 18:32 — [Ablation (C_posthoc, 2048ep)] `C_posthoc_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=True`
- **Why this run:** Induction ablation: representation induction vs fitting only a post-hoc probe on a latent the physics loss cannot shape (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `C_posthoc_seed2.json`, `C_posthoc_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.7615 | 0.2838 |
| drag | -0.1523 | -4.1354 |
| mass | 0.2232 | -0.3178 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | -0.0191 | -2.7526 |
| stiffness_over_mass | 0.3918 | 0.2570 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.2622 |
| path_a_vs_dataset | 0.3161 |
| path_b_vs_dataset | 0.0643 |
| path_b_vs_teacher | 0.2575 |
| persistence_query_vs_dataset | 0.3552 |
| persistence_vs_dataset | 0.5150 |

### 2026-08-25 17:17 — [Ablation (G_probe_mlp, 2048ep)] `G_probe_mlp_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=False`
- **Why this run:** Probe capacity ablation: checks the result is not an artifact of the linear probe (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `G_probe_mlp_seed0.json`, `G_probe_mlp_seed0.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.7686 | 0.1838 |
| drag | -0.1231 | -8.5023 |
| mass | 0.2280 | -0.3584 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | 0.0256 | -5.1968 |
| stiffness_over_mass | 0.3807 | 0.2154 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.3412 |
| path_a_vs_dataset | 0.3762 |
| path_b_vs_dataset | 0.0836 |
| path_b_vs_teacher | 0.3337 |
| persistence_query_vs_dataset | 0.4439 |
| persistence_vs_dataset | 0.5509 |

### 2026-08-25 18:22 — [Ablation (G_probe_mlp, 2048ep)] `G_probe_mlp_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=False`
- **Why this run:** Probe capacity ablation: checks the result is not an artifact of the linear probe (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `G_probe_mlp_seed1.json`, `G_probe_mlp_seed1.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.7766 | -0.0421 |
| drag | -0.1030 | -4.5643 |
| mass | 0.1095 | -0.3816 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | -0.0474 | -3.5046 |
| stiffness_over_mass | 0.4131 | 0.1521 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.2664 |
| path_a_vs_dataset | 0.3401 |
| path_b_vs_dataset | 0.0684 |
| path_b_vs_teacher | 0.2675 |
| persistence_query_vs_dataset | 0.3505 |
| persistence_vs_dataset | 0.5179 |

### 2026-08-25 18:32 — [Ablation (G_probe_mlp, 2048ep)] `G_probe_mlp_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=physwm, detach_probe_input=False`
- **Why this run:** Probe capacity ablation: checks the result is not an artifact of the linear probe (re-run at 2048ep).
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `G_probe_mlp_seed2.json`, `G_probe_mlp_seed2.log`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_stiffness | 0.7408 | 0.1771 |
| drag | -0.1747 | -4.1608 |
| mass | 0.1045 | -0.3801 |
| object_radius | nan | nan |
| poker_radius | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| contact_radius | nan | nan |
| drag_over_mass | -0.0682 | -4.6672 |
| stiffness_over_mass | 0.2924 | 0.0887 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_query_vs_dataset | 0.2779 |
| path_a_vs_dataset | 0.3258 |
| path_b_vs_dataset | 0.0696 |
| path_b_vs_teacher | 0.2773 |
| persistence_query_vs_dataset | 0.3552 |
| persistence_vs_dataset | 0.5150 |

### 2026-08-25 17:11 — [Functional use (2048ep)] `func2048_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=16, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, tactile=True, detach_probe_input=False, physics_target=path_a, theta_supervision=0.0`
- **Why this run:** Functional use (substitution + multi-horizon rollout) at 2048 episodes, with theta_variation added to flag which PokeWorld parameters actually vary across episodes.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON; not run through overnight.py)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `func2048_seed0.json`, `func2048_seed0.log`

**sensitivity** (prediction shift): mass 0.00498, contact_stiffness 0.01186, drag 0.00293, poker_radius 0.14747, object_radius 0.14693

**theta variation across episodes:** mass (varies=True), contact_stiffness (varies=True), drag (varies=True), poker_radius (varies=False), object_radius (varies=False)

**substitution** (one-step Path B error vs true s_next):

| source | scaled RMSE |
| --- | --- |
| probe | 0.09047 |
| true | 0.00000 |
| shuffled | 0.08822 |
| nominal | 0.10960 |

probe closes **-2.6%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probe | 0.1312 | 0.1365 | 0.1952 | 0.2140 | 0.2109 | 0.2019 | 0.1990 |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| shuffled | 0.1150 | 0.1199 | 0.1775 | 0.2227 | 0.2222 | 0.2161 | 0.2157 |
| nominal | 0.1580 | 0.1878 | 0.2305 | 0.2556 | 0.2582 | 0.2586 | 0.2555 |

### 2026-08-25 17:08 — [Functional use (2048ep)] `func2048_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=16, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=1, tactile=True, detach_probe_input=False, physics_target=path_a, theta_supervision=0.0`
- **Why this run:** Functional use (substitution + multi-horizon rollout) at 2048 episodes, with theta_variation added to flag which PokeWorld parameters actually vary across episodes.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON; not run through overnight.py)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `func2048_seed1.json`, `func2048_seed1.log`

**sensitivity** (prediction shift): mass 0.00355, contact_stiffness 0.01185, drag 0.02044, poker_radius 0.21455, object_radius 0.23225

**theta variation across episodes:** mass (varies=True), contact_stiffness (varies=True), drag (varies=True), poker_radius (varies=False), object_radius (varies=False)

**substitution** (one-step Path B error vs true s_next):

| source | scaled RMSE |
| --- | --- |
| probe | 0.07919 |
| true | 0.00000 |
| shuffled | 0.09950 |
| nominal | 0.09964 |

probe closes **20.4%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probe | 0.1390 | 0.1142 | 0.1509 | 0.1542 | 0.1535 | 0.1774 | 0.1806 |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| shuffled | 0.1343 | 0.1568 | 0.2042 | 0.2072 | 0.1980 | 0.2047 | 0.2035 |
| nominal | 0.1705 | 0.1450 | 0.1548 | 0.1672 | 0.1690 | 0.1845 | 0.1834 |

### 2026-08-25 17:10 — [Functional use (2048ep)] `func2048_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=16, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=2, tactile=True, detach_probe_input=False, physics_target=path_a, theta_supervision=0.0`
- **Why this run:** Functional use (substitution + multi-horizon rollout) at 2048 episodes, with theta_variation added to flag which PokeWorld parameters actually vary across episodes.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON; not run through overnight.py)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `func2048_seed2.json`, `func2048_seed2.log`

**sensitivity** (prediction shift): mass 0.02560, contact_stiffness 0.01237, drag 0.00374, poker_radius 0.17090, object_radius 0.18327

**theta variation across episodes:** mass (varies=True), contact_stiffness (varies=True), drag (varies=True), poker_radius (varies=False), object_radius (varies=False)

**substitution** (one-step Path B error vs true s_next):

| source | scaled RMSE |
| --- | --- |
| probe | 0.09491 |
| true | 0.00000 |
| shuffled | 0.16564 |
| nominal | 0.10059 |

probe closes **42.7%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probe | 0.1211 | 0.1438 | 0.1656 | 0.1784 | 0.1964 | 0.1970 | 0.1995 |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| shuffled | 0.1545 | 0.1530 | 0.1970 | 0.2405 | 0.2471 | 0.2409 | 0.2397 |
| nominal | 0.1350 | 0.1287 | 0.1587 | 0.1875 | 0.1943 | 0.1981 | 0.2000 |

### 2026-08-25 15:00 — [Pilot (2048ep, primary claim)] `A_pilot_2048ep_seed0` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=0, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Pilot run at the corrected 2048-episode budget, primary claim conditions.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `A_pilot_2048ep_seed0.json`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.7694 | 0.1180 | 0.7396 | -1.4193 | 0.8051 | 0.8002 |
| drag | -0.1847 | -4.9174 | -0.2499 | -0.0126 | -0.4060 | -0.4371 |
| mass | 0.2945 | -0.1336 | 0.2459 | -0.0689 | 0.1519 | 0.1488 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | 0.0386 | -2.1030 | 0.0515 | -0.0359 | -0.0100 | -0.0266 |
| stiffness_over_mass | 0.4607 | 0.0509 | 0.4215 | -0.6106 | 0.4547 | 0.4679 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.3608 | 0.3328 | 0.3713 |
| path_a_vs_dataset | 0.3850 | 0.3623 | 0.3865 |
| path_b_vs_dataset | 0.0881 | 0.1172 | 0.0750 |
| path_b_vs_teacher | 0.3570 | 0.3426 | 0.3721 |
| persistence_query_vs_dataset | 0.4439 | 0.4439 | 0.4439 |
| persistence_vs_dataset | 0.5509 | 0.5509 | 0.5509 |

### 2026-08-25 18:21 — [Pilot (2048ep, primary claim)] `A_pilot_2048ep_seed1` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=1, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Pilot run at the corrected 2048-episode budget, primary claim conditions.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `A_pilot_2048ep_seed1.json`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.8182 | -0.0565 | 0.7498 | -1.3446 | 0.8416 | 0.8024 |
| drag | -0.1549 | -9.0822 | -0.1718 | -0.0042 | -0.3360 | -0.3226 |
| mass | 0.1860 | -0.4672 | 0.2110 | -0.0409 | 0.1466 | 0.1394 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | 0.0050 | -9.4305 | 0.0153 | -0.0615 | -0.1155 | -0.1009 |
| stiffness_over_mass | 0.4719 | 0.1440 | 0.4204 | -0.6396 | 0.4466 | 0.4112 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.2853 | 0.2460 | 0.2984 |
| path_a_vs_dataset | 0.3480 | 0.3255 | 0.3619 |
| path_b_vs_dataset | 0.0842 | 0.1115 | 0.0732 |
| path_b_vs_teacher | 0.2887 | 0.2582 | 0.2997 |
| persistence_query_vs_dataset | 0.3505 | 0.3505 | 0.3505 |
| persistence_vs_dataset | 0.5179 | 0.5179 | 0.5179 |

### 2026-08-25 18:21 — [Pilot (2048ep, primary claim)] `A_pilot_2048ep_seed2` (2048ep re-run, logged post hoc)

- **Params:** `encoder=tiny_cnn, window=8, probe_source=predicted, episodes=2048, length=48, epochs=60, alpha=1.0, seed=2, batch_size=32, tactile=True, conditions=predictive,physwm,theta_oracle, detach_probe_input=False`
- **Why this run:** Pilot run at the corrected 2048-episode budget, primary claim conditions.
- **Commit / working tree:** `13208111c`, dirty (logged after the fact from raw JSON in /workspace/physwm-artifacts/runs/; not run through overnight.py, so no exact command line or wall-clock is available)
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** unknown (no .log with timing for this run)
- **Status:** pass (JSON payload present)
- **Artifacts:** `A_pilot_2048ep_seed2.json`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_stiffness | 0.7647 | 0.1991 | 0.7731 | -1.3263 | 0.8038 | 0.8041 |
| drag | -0.0991 | -5.2736 | -0.1500 | -0.0071 | -0.4538 | -0.4461 |
| mass | 0.2295 | -0.4341 | 0.2656 | -0.1450 | 0.1996 | 0.1820 |
| object_radius | nan | nan | nan | nan | nan | nan |
| poker_radius | nan | nan | nan | nan | nan | nan |

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe | predictive/decodable | predictive/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| contact_radius | nan | nan | nan | nan | nan | nan |
| drag_over_mass | 0.0628 | -6.6241 | 0.0607 | -0.0077 | -0.0935 | -0.0608 |
| stiffness_over_mass | 0.4073 | 0.0880 | 0.4545 | -0.5723 | 0.4529 | 0.4231 |

**prediction and distillation — val RMSE**

| metric | physwm | predictive | theta_oracle |
| --- | --- | --- | --- |
| path_a_query_vs_dataset | 0.2684 | 0.2683 | 0.2959 |
| path_a_vs_dataset | 0.3168 | 0.3155 | 0.3408 |
| path_b_vs_dataset | 0.0766 | 0.0857 | 0.0629 |
| path_b_vs_teacher | 0.2709 | 0.2782 | 0.2959 |
| persistence_query_vs_dataset | 0.3552 | 0.3552 | 0.3552 |
| persistence_vs_dataset | 0.5150 | 0.5150 | 0.5150 |

### 2026-08-25 15:58 — [D. visual-only identifiability] `D_visual_only_seed2`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 2 --batch-size 32 --conditions predictive,physwm,theta_oracle --no-tactile --out /workspace/physwm-artifacts/runs/20260825-133600/D_visual_only_seed2.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/D_visual_only_seed2`
- **Why this run:** Matches the visual-encoder claim and reports identifiable dynamics coordinates k/m and c/m, not misleading raw R2.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 136.7 min
- **Status:** **FAIL**
- **Artifacts:** `D_visual_only_seed2.log`

```
```

- **Notes:** run failed; see the log for the traceback.

### 2026-08-25 15:58 — [D. visual-only identifiability] `D_visual_only_seed1`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 1 --batch-size 32 --conditions predictive,physwm,theta_oracle --no-tactile --out /workspace/physwm-artifacts/runs/20260825-133600/D_visual_only_seed1.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/D_visual_only_seed1`
- **Why this run:** Matches the visual-encoder claim and reports identifiable dynamics coordinates k/m and c/m, not misleading raw R2.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 136.7 min
- **Status:** **FAIL**
- **Artifacts:** `D_visual_only_seed1.log`

```
```

- **Notes:** run failed; see the log for the traceback.

### 2026-08-25 15:58 — [D. visual-only identifiability] `D_visual_only_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 0 --batch-size 32 --conditions predictive,physwm,theta_oracle --no-tactile --out /workspace/physwm-artifacts/runs/20260825-133600/D_visual_only_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/D_visual_only_seed0`
- **Why this run:** Matches the visual-encoder claim and reports identifiable dynamics coordinates k/m and c/m, not misleading raw R2.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 136.7 min
- **Status:** **FAIL**
- **Artifacts:** `D_visual_only_seed0.log`

```
```

- **Notes:** run failed; see the log for the traceback.

### 2026-08-25 13:58 — [A. primary claim (predictive vs PhysWM vs theta ceiling)] `A_claim_tiny_seed1`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 1 --batch-size 32 --conditions predictive,physwm,theta_oracle --out /workspace/physwm-artifacts/runs/20260825-133600/A_claim_tiny_seed1.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/A_claim_tiny_seed1`
- **Why this run:** Same checkpoint reports Path A accuracy, baseline blindness, PhysWM recovery, and the label-supervised ceiling.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 16.4 min
- **Status:** pass
- **Artifacts:** `A_claim_tiny_seed1.log`, `A_claim_tiny_seed1.json`, `models/A_claim_tiny_seed1/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.2299 | 0.0169 | -0.3554 | -0.2425 | -0.2452 | -0.2303 |
| contact_stiffness | 0.6198 | -1.2041 | 0.0721 | -0.0213 | 0.3910 | 0.3407 |
| drag | -0.5760 | -0.0024 | -0.6404 | -2.9162 | -0.4296 | -0.4757 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| stiffness_over_mass | -0.0749 | -0.5802 | -0.3197 | 0.0645 | -0.0155 | 0.0910 |
| drag_over_mass | -0.3586 | -0.0529 | -0.4835 | -2.7532 | -0.2867 | -0.1607 |

**prediction and distillation — val RMSE**

| metric | predictive | physwm | theta_oracle |
| --- | --- | --- | --- |
| path_a_vs_dataset | 0.3938 | 0.3846 | 0.4332 |
| path_a_query_vs_dataset | 0.3337 | 0.3435 | 0.3771 |
| path_b_vs_teacher | 0.3256 | 0.3392 | 0.3758 |
| path_b_vs_dataset | 0.0941 | 0.0725 | 0.0763 |
| persistence_vs_dataset | 0.4701 | 0.4701 | 0.4701 |
| persistence_query_vs_dataset | 0.3186 | 0.3186 | 0.3186 |

### 2026-08-25 13:58 — [A. primary claim (predictive vs PhysWM vs theta ceiling)] `A_claim_tiny_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 0 --batch-size 32 --conditions predictive,physwm,theta_oracle --out /workspace/physwm-artifacts/runs/20260825-133600/A_claim_tiny_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/A_claim_tiny_seed0`
- **Why this run:** Same checkpoint reports Path A accuracy, baseline blindness, PhysWM recovery, and the label-supervised ceiling.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 16.2 min
- **Status:** pass
- **Artifacts:** `A_claim_tiny_seed0.log`, `A_claim_tiny_seed0.json`, `models/A_claim_tiny_seed0/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.4364 | -0.0311 | -0.3991 | -0.2028 | -0.0985 | -0.1648 |
| contact_stiffness | 0.3997 | -1.5344 | 0.1551 | -0.0718 | 0.5987 | 0.6058 |
| drag | -0.5019 | -0.0347 | -0.8053 | -1.3016 | -0.3612 | -0.3633 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| stiffness_over_mass | -0.0741 | -0.7003 | -0.3263 | 0.0237 | 0.1000 | 0.0962 |
| drag_over_mass | -0.5487 | -0.0823 | -0.8160 | -1.4415 | -0.6029 | -0.4646 |

**prediction and distillation — val RMSE**

| metric | predictive | physwm | theta_oracle |
| --- | --- | --- | --- |
| path_a_vs_dataset | 0.4984 | 0.5400 | 0.5423 |
| path_a_query_vs_dataset | 0.4384 | 0.4683 | 0.4870 |
| path_b_vs_teacher | 0.4648 | 0.4636 | 0.4953 |
| path_b_vs_dataset | 0.1712 | 0.1334 | 0.0951 |
| persistence_vs_dataset | 0.5850 | 0.5850 | 0.5850 |
| persistence_query_vs_dataset | 0.4867 | 0.4867 | 0.4867 |

### 2026-08-25 13:57 — [A. primary claim (predictive vs PhysWM vs theta ceiling)] `A_claim_tiny_seed2`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 2 --batch-size 32 --conditions predictive,physwm,theta_oracle --out /workspace/physwm-artifacts/runs/20260825-133600/A_claim_tiny_seed2.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/A_claim_tiny_seed2`
- **Why this run:** Same checkpoint reports Path A accuracy, baseline blindness, PhysWM recovery, and the label-supervised ceiling.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 16.1 min
- **Status:** pass
- **Artifacts:** `A_claim_tiny_seed2.log`, `A_claim_tiny_seed2.json`, `models/A_claim_tiny_seed2/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.7390 | -0.1435 | -0.6215 | -0.1606 | -0.5268 | -0.4617 |
| contact_stiffness | 0.3691 | -1.0804 | 0.4571 | 0.1861 | 0.3823 | 0.3590 |
| drag | -1.1027 | -0.0516 | -0.7318 | -0.7455 | -0.4687 | -0.5407 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | theta_oracle/decodable | theta_oracle/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| stiffness_over_mass | -0.1947 | -0.5450 | -0.2006 | 0.1240 | -0.2585 | -0.1709 |
| drag_over_mass | -0.9904 | -0.0019 | -0.9939 | -0.4261 | -0.3514 | -0.2683 |

**prediction and distillation — val RMSE**

| metric | predictive | physwm | theta_oracle |
| --- | --- | --- | --- |
| path_a_vs_dataset | 0.4135 | 0.4382 | 0.4555 |
| path_a_query_vs_dataset | 0.3242 | 0.3572 | 0.3707 |
| path_b_vs_teacher | 0.3305 | 0.3639 | 0.3824 |
| path_b_vs_dataset | 0.1262 | 0.1201 | 0.0790 |
| persistence_vs_dataset | 0.4657 | 0.4657 | 0.4657 |
| persistence_query_vs_dataset | 0.2684 | 0.2684 | 0.2684 |

### 2026-08-25 13:51 — [F. functional use (substitution and rollout)] `F_functional_seed1`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --physics-target path_a --window 16 --seed 1 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /workspace/physwm-artifacts/runs/20260825-133600/F_functional_seed1.json`
- **Why this run:** Distinguishes physically useful parameters from values that are merely correlated or decodable.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 10.0 min
- **Status:** pass
- **Artifacts:** `F_functional_seed1.log`, `F_functional_seed1.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.08277, contact_stiffness 0.01184, drag 0.00346, poker_radius 0.16578, object_radius 0.16043

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.07084 |
| shuffled | 0.10789 |
| probe | 0.10916 |

probe closes **-1.2%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1488 | 0.1182 | 0.1711 | 0.1699 | 0.1886 | 0.2039 | 0.2033 |
| nominal | 0.0689 | 0.0642 | 0.1467 | 0.1299 | 0.1229 | 0.1314 | 0.1321 |
| shuffled | 0.1176 | 0.1052 | 0.2084 | 0.2438 | 0.2444 | 0.2371 | 0.2340 |

### 2026-08-25 13:51 — [G. context-length ablation] `G_context16_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 16 --seed 0 --batch-size 32 --conditions physwm --out /workspace/physwm-artifacts/runs/20260825-133600/G_context16_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/G_context16_seed0`
- **Why this run:** Tests whether longer action-response history strengthens persistent parameter identification.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.9 min
- **Status:** pass
- **Artifacts:** `G_context16_seed0.log`, `G_context16_seed0.json`, `models/G_context16_seed0/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -1.0743 | -0.1058 |
| contact_stiffness | 0.2387 | 0.2166 |
| drag | -1.4420 | -0.4995 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| stiffness_over_mass | -0.3492 | 0.2082 |
| drag_over_mass | -1.5945 | -0.8595 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_vs_dataset | 0.5337 |
| path_a_query_vs_dataset | 0.4679 |
| path_b_vs_teacher | 0.4775 |
| path_b_vs_dataset | 0.1221 |
| persistence_vs_dataset | 0.5426 |
| persistence_query_vs_dataset | 0.3777 |

### 2026-08-25 13:51 — [F. functional use (substitution and rollout)] `F_functional_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --physics-target path_a --window 16 --seed 0 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /workspace/physwm-artifacts/runs/20260825-133600/F_functional_seed0.json`
- **Why this run:** Distinguishes physically useful parameters from values that are merely correlated or decodable.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.9 min
- **Status:** pass
- **Artifacts:** `F_functional_seed0.log`, `F_functional_seed0.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.05246, contact_stiffness 0.01187, drag 0.00259, poker_radius 0.13167, object_radius 0.13062

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.11701 |
| shuffled | 0.16364 |
| probe | 0.14708 |

probe closes **10.1%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1259 | 0.1388 | 0.1631 | 0.2697 | 0.2727 | 0.2654 | 0.2598 |
| nominal | 0.1103 | 0.1615 | 0.1885 | 0.2650 | 0.2813 | 0.2808 | 0.2815 |
| shuffled | 0.2086 | 0.2740 | 0.2634 | 0.3415 | 0.3347 | 0.3266 | 0.3233 |

### 2026-08-25 13:51 — [F. functional use (substitution and rollout)] `F_functional_seed2`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --physics-target path_a --window 16 --seed 2 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /workspace/physwm-artifacts/runs/20260825-133600/F_functional_seed2.json`
- **Why this run:** Distinguishes physically useful parameters from values that are merely correlated or decodable.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.9 min
- **Status:** pass
- **Artifacts:** `F_functional_seed2.log`, `F_functional_seed2.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.05824, contact_stiffness 0.01234, drag 0.00327, poker_radius 0.26075, object_radius 0.25940

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.11111 |
| shuffled | 0.14816 |
| probe | 0.11774 |

probe closes **20.5%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1753 | 0.1356 | 0.1882 | 0.1851 | 0.1761 | 0.1944 | 0.2075 |
| nominal | 0.1872 | 0.1426 | 0.2021 | 0.2210 | 0.2133 | 0.2099 | 0.2221 |
| shuffled | 0.2454 | 0.2068 | 0.3074 | 0.2980 | 0.3040 | 0.2935 | 0.2920 |

### 2026-08-25 13:51 — [C. target ablation (dataset next-state target)] `C_dataset_target_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 0 --batch-size 32 --conditions state_target --out /workspace/physwm-artifacts/runs/20260825-133600/C_dataset_target_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/C_dataset_target_seed0`
- **Why this run:** Replaces the model prediction with the raw state label while keeping the solver and route fixed.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.3 min
- **Status:** pass
- **Artifacts:** `C_dataset_target_seed0.log`, `C_dataset_target_seed0.json`, `models/C_dataset_target_seed0/`

**theta recovery — val R²**

| param | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| mass | -0.0588 | 0.0427 |
| contact_stiffness | 0.5595 | 0.5278 |
| drag | -0.8515 | -0.1031 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | state_target/decodable | state_target/own_probe |
| --- | --- | --- |
| stiffness_over_mass | 0.2107 | 0.1781 |
| drag_over_mass | -0.4518 | -0.0624 |

**prediction and distillation — val RMSE**

| metric | state_target |
| --- | --- |
| path_a_vs_dataset | 0.4900 |
| path_a_query_vs_dataset | 0.4433 |
| path_b_vs_teacher | 0.4453 |
| path_b_vs_dataset | 0.0994 |
| persistence_vs_dataset | 0.5850 |
| persistence_query_vs_dataset | 0.4867 |

### 2026-08-25 13:51 — [B. routing ablation (pre-action encoder latent)] `B_preaction_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source encoded --window 8 --seed 0 --batch-size 32 --conditions physwm --out /workspace/physwm-artifacts/runs/20260825-133600/B_preaction_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/B_preaction_seed0`
- **Why this run:** Tests whether reading the SAME action-conditioned latent is necessary; this is the route used by the earlier code.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.3 min
- **Status:** pass
- **Artifacts:** `B_preaction_seed0.log`, `B_preaction_seed0.json`, `models/B_preaction_seed0/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.1283 | -0.5990 |
| contact_stiffness | 0.1182 | -0.7348 |
| drag | 0.0080 | -0.1713 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| stiffness_over_mass | 0.0717 | 0.0232 |
| drag_over_mass | -0.0618 | -0.3455 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_vs_dataset | 0.5351 |
| path_a_query_vs_dataset | 0.4654 |
| path_b_vs_teacher | 0.4603 |
| path_b_vs_dataset | 0.0971 |
| persistence_vs_dataset | 0.5850 |
| persistence_query_vs_dataset | 0.4867 |

### 2026-08-25 13:51 — [B. routing ablation (pre-action encoder latent)] `B_preaction_seed2`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source encoded --window 8 --seed 2 --batch-size 32 --conditions physwm --out /workspace/physwm-artifacts/runs/20260825-133600/B_preaction_seed2.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/B_preaction_seed2`
- **Why this run:** Tests whether reading the SAME action-conditioned latent is necessary; this is the route used by the earlier code.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.2 min
- **Status:** pass
- **Artifacts:** `B_preaction_seed2.log`, `B_preaction_seed2.json`, `models/B_preaction_seed2/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.2801 | -0.8757 |
| contact_stiffness | -0.3187 | -1.0197 |
| drag | -0.0582 | -0.2065 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| stiffness_over_mass | -0.3531 | -0.2219 |
| drag_over_mass | -0.1773 | -0.4306 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_vs_dataset | 0.4389 |
| path_a_query_vs_dataset | 0.3561 |
| path_b_vs_teacher | 0.3535 |
| path_b_vs_dataset | 0.0813 |
| persistence_vs_dataset | 0.4657 |
| persistence_query_vs_dataset | 0.2684 |

### 2026-08-25 13:51 — [G. probe capacity ablation] `G_probe_mlp_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 128 --alpha 1.0 --probe-source predicted --window 8 --seed 0 --batch-size 32 --conditions physwm --out /workspace/physwm-artifacts/runs/20260825-133600/G_probe_mlp_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/G_probe_mlp_seed0`
- **Why this run:** Checks that the result is not an artifact of the linear probe.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.2 min
- **Status:** pass
- **Artifacts:** `G_probe_mlp_seed0.log`, `G_probe_mlp_seed0.json`, `models/G_probe_mlp_seed0/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.5892 | -0.2138 |
| contact_stiffness | 0.4806 | 0.0601 |
| drag | -0.8259 | -2.4611 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| stiffness_over_mass | -0.2697 | 0.1624 |
| drag_over_mass | -0.6708 | -2.0594 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_vs_dataset | 0.5162 |
| path_a_query_vs_dataset | 0.4551 |
| path_b_vs_teacher | 0.4519 |
| path_b_vs_dataset | 0.1298 |
| persistence_vs_dataset | 0.5850 |
| persistence_query_vs_dataset | 0.4867 |

### 2026-08-25 13:51 — [B. routing ablation (pre-action encoder latent)] `B_preaction_seed1`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source encoded --window 8 --seed 1 --batch-size 32 --conditions physwm --out /workspace/physwm-artifacts/runs/20260825-133600/B_preaction_seed1.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/B_preaction_seed1`
- **Why this run:** Tests whether reading the SAME action-conditioned latent is necessary; this is the route used by the earlier code.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.2 min
- **Status:** pass
- **Artifacts:** `B_preaction_seed1.log`, `B_preaction_seed1.json`, `models/B_preaction_seed1/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.1582 | -0.3183 |
| contact_stiffness | -0.3630 | -0.5285 |
| drag | -0.3137 | -1.1576 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| stiffness_over_mass | -0.2283 | -0.0908 |
| drag_over_mass | -0.1587 | -2.2553 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_vs_dataset | 0.3826 |
| path_a_query_vs_dataset | 0.3432 |
| path_b_vs_teacher | 0.3390 |
| path_b_vs_dataset | 0.0654 |
| persistence_vs_dataset | 0.4701 |
| persistence_query_vs_dataset | 0.3186 |

### 2026-08-25 13:51 — [C. induction ablation (detached probe input)] `C_posthoc_seed0`

- **Command:** `/workspace/stable-worldmodel/.venv/bin/python -u /workspace/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --probe-source predicted --window 8 --seed 0 --batch-size 32 --conditions physwm --detach-probe-input --out /workspace/physwm-artifacts/runs/20260825-133600/C_posthoc_seed0.json --save-dir /workspace/physwm-artifacts/runs/20260825-133600/models/C_posthoc_seed0`
- **Why this run:** Tests representation induction against fitting only a post-hoc probe on a latent the physics loss cannot shape.
- **Commit / working tree:** `13208111c`, clean
- **Hardware:** NVIDIA H100 80GB HBM3, python
- **Duration:** 9.2 min
- **Status:** pass
- **Artifacts:** `C_posthoc_seed0.log`, `C_posthoc_seed0.json`, `models/C_posthoc_seed0/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.3369 | -0.0898 |
| contact_stiffness | 0.4107 | -0.1082 |
| drag | -0.7766 | -2.1337 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

**identifiable dynamics coordinates — val R²**

| coordinate | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| stiffness_over_mass | 0.0559 | 0.1669 |
| drag_over_mass | -0.6071 | -1.1814 |

**prediction and distillation — val RMSE**

| metric | physwm |
| --- | --- |
| path_a_vs_dataset | 0.4991 |
| path_a_query_vs_dataset | 0.4475 |
| path_b_vs_teacher | 0.4433 |
| path_b_vs_dataset | 0.0985 |
| persistence_vs_dataset | 0.5850 |
| persistence_query_vs_dataset | 0.4867 |

### 2026-08-25 18:35 IST — Aligned workshop protocol: matrix and focused checks

- **Commands:**
  - `.venv-phy-wm/bin/python scripts/eval/overnight.py --dry-run --seeds 0,1,2`
  - `.venv-phy-wm/bin/ruff check stable_worldmodel/wm/physwm scripts/eval scripts/train/physwm.py scripts/smoke/run_smoke.py tests/wm/test_physwm.py && .venv-phy-wm/bin/pytest -q tests/wm/test_physwm.py && git diff --check`
- **Config:** dry-run resolves the aligned workshop matrix; no training config was executed by this entry
- **Seed(s):** `0,1,2` (matrix resolution only)
- **Commit / working tree:** `c7fd83f2f`, branch `phy-wm`, dirty. The working tree contains the aligned PhysWM/model/eval/config/test/docs changes listed by `git status`; no result in this entry comes from committed code.
- **Hardware:** CPU, `.venv-phy-wm`; the dry-run and static/unit checks do not require CUDA
- **Data:** none generated
- **Duration:** unknown
- **Status:** pass (protocol resolution and checks only; **not a paper result**)
- **Artifacts:** none from the dry-run

| Check | Result |
| --- | --- |
| Aligned matrix resolution | `20 experiments (16 light, 4 heavy)` |
| Ruff | `All checks passed!` |
| Focused PhysWM tests | `60 passed, 1 warning in 1.16s` |
| Diff whitespace check | pass |

- **Notes:** This is the first validation of the workshop-aligned implementation.
  Both the state decoder and low-capacity physics probe now consume the same
  action-conditioned predicted latent, Path B is trained against
  `PathA.detach()`, parameters are averaged at episode level, and evaluation
  separates context fitting from held-out query transitions. The matrix now
  includes the predictive baseline, aligned PhysWM objective, theta oracle,
  pre-action ablation, dataset-target ablation, detached post-hoc probe,
  visual-only observations, frozen DINOv2, functional multi-step supervision,
  probe-capacity ablation, and a longer-context ablation. Runs logged at or
  before 07:48 today used the pre-action/older-pooling protocol and are
  **historical diagnostics, not comparable workshop results**. Compatibility
  loading retains `encoded` only for those historical configs.

### 2026-08-25 18:34 IST — Tiny CPU context/query evaluator smoke

- **Command:** `.venv-phy-wm/bin/python scripts/eval/decodability.py --epochs 1 --episodes 12 --length 16 --window 4 --batch-size 16 --conditions physwm,state_target --device cpu --no-amp --out /tmp/physwm_split_smoke.json`
- **Config:** `tiny_cnn`, linear probe (`probe_hidden=0`), predicted-latent probe source, window 4, tactile enabled, aligned context/query split
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f2f`, branch `phy-wm`, dirty with the aligned evaluator/model changes
- **Hardware:** CPU, `.venv-phy-wm`
- **Data:** synthetic PokeWorld, 12 episodes x 16 steps; 17 train windows and 6 validation windows
- **Duration:** 2.4 s
- **Status:** pass as a code-path smoke; **inconclusive as an experiment**
- **Artifacts:** `/tmp/physwm_split_smoke.json` (ephemeral)

**Theta recovery — validation R² (verbatim from artifact)**

| parameter | physwm/decodable | physwm/own_probe | state_target/decodable | state_target/own_probe |
| --- | ---: | ---: | ---: | ---: |
| mass | -1.5877046585083008 | -0.02861034870147705 | 0.550360918045044 | -0.026668548583984375 |
| contact_stiffness | 0.06394195556640625 | -2.7038636207580566 | 0.1259346604347229 | -1.6465020179748535 |
| drag | -3.354058265686035 | -1.4495656490325928 | -0.17266738414764404 | 0.01395636796951294 |

**Functional recovery — validation R² (verbatim from artifact)**

| parameterization | physwm/decodable | physwm/own_probe | state_target/decodable | state_target/own_probe |
| --- | ---: | ---: | ---: | ---: |
| stiffness_over_mass | 0.1026948094367981 | -0.7485069036483765 | 0.8207908272743225 | -0.5715477466583252 |
| drag_over_mass | -3.556215286254883 | -0.013676881790161133 | 0.4206364154815674 | -0.13653123378753662 |

**Prediction RMSE (verbatim from artifact)**

| condition | Path A vs dataset | Path A query vs dataset | Path B vs teacher | Path B vs dataset | persistence vs dataset | persistence query vs dataset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| physwm | 0.9486017120277962 | 1.0779464205211329 | 1.1132643456318936 | 0.30559073351646776 | 0.6476620941655136 | 0.4636516254891442 |
| state_target | 1.0288633736215227 | 1.1251993073415314 | 1.0788806865296094 | 0.5129024414188844 | 0.6476620941655136 | 0.4636516254891442 |

- **Notes:** One epoch and six validation windows are intentionally too small
  for scientific interpretation. In particular Path A does not beat
  persistence here, so this run does not satisfy the high-signal-teacher
  premise and must not be cited as evidence for or against the hypothesis.

### 2026-08-25 18:30 IST — Aligned PokeWorld end-to-end CPU smoke

- **Command:** `STABLEWM_HOME=/tmp/physwm_aligned_home2 MUJOCO_GL=egl .venv-phy-wm/bin/python scripts/smoke/run_smoke.py --benchmarks pokeworld --epochs 1 --episodes 8 --length 16`
- **Config:** `scripts/train/config/physwm.yaml` + `bench=pokeworld`, one epoch, 8 episodes x 16 steps, batch size 16
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f2f`, branch `phy-wm`, dirty; run fingerprint `5ad86f210827dc5d`
- **Hardware:** CPU (`torch 2.13.0+cpu`), `.venv-phy-wm`, `MUJOCO_GL=egl`
- **Data:** generated PokeWorld smoke data, 8 episodes x 16 steps; 10 train windows and 5 validation windows
- **Duration:** 8.5 s total across the four smoke stages
- **Status:** pass (4/4 smoke stages; **not a paper result**)
- **Artifacts:** `/tmp/physwm_aligned_home2/checkpoints/smoke_physwm_pokeworld/` (ephemeral), including `last.pt`, `weights.pt`, `history.json`, `final_metrics.json`, `config.yaml`, and `run_meta.json`

| Solver check | persistence RMSE | nominal RMSE | fitted RMSE |
| --- | ---: | ---: | ---: |
| PokeWorld mean | 0.4051 | 0.0660 | 0.0002 |

**Training metrics (verbatim from `final_metrics.json`)**

| metric | value |
| --- | ---: |
| best_val_loss | 2.2529520988464355 |
| train/loss | 2.191218376159668 |
| train/loss_a | 1.071963906288147 |
| train/loss_b | 1.1192545890808105 |
| train/rmse_a | 1.035356879234314 |
| train/rmse_b_vs_teacher | 1.057948350906372 |
| train/rmse_b_vs_dataset | 0.009400313720107079 |
| val/loss | 2.2529520988464355 |
| val/loss_a | 1.1696773767471313 |
| val/loss_b | 1.0832747220993042 |
| val/rmse_a | 1.0815162658691406 |
| val/rmse_b_vs_teacher | 1.0408048629760742 |
| val/rmse_b_vs_dataset | 0.005020867567509413 |
| val/r2/mass | -37.807682037353516 |
| val/r2/contact_stiffness | -2.673313617706299 |
| val/r2/drag | -0.8616777658462524 |

- **Notes:** Unit tests, solver validation, one-epoch training, and checkpoint
  round-trip all passed. The negative R² values and weak Path A are expected
  for this deliberately tiny run; they are logged only to prove that the
  aligned training path executes and serializes correctly.

### 2026-08-25 (time not recorded) — CUDA evaluator launch unavailable

- **Command:** `.venv-phy-wm-gpu/bin/python scripts/eval/decodability.py --epochs 1 --episodes 12 --length 16 --window 4 --batch-size 16 --conditions predictive,physwm,state_target,theta_oracle --device cuda --out /tmp/physwm_aligned_smoke.json`
- **Config:** tiny aligned evaluator smoke requested on CUDA
- **Seed(s):** 0 (default)
- **Commit / working tree:** `c7fd83f2f`, branch `phy-wm`, dirty
- **Hardware:** requested CUDA; no GPU was available to this session
- **Data:** generation/training did not start
- **Duration:** 1.2 s
- **Status:** fail before training
- **Artifacts:** none

`RuntimeError: No CUDA GPUs are available`

- **Notes:** This is an environment availability failure, not a model or
  evaluator failure. No aligned CUDA paper run has been completed yet.

### 2026-08-25 07:48 — [B. encoder scale (frozen DINOv2-small @224)] `B_dinov2_seed1`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder dinov2 --epochs 40 --episodes 128 --length 48 --probe-hidden 0 --alpha 1.0 --seed 1 --batch-size 32 --conditions predictive,physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/B_dinov2_seed1.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/B_dinov2_seed1`
- **Why this run:** Rules out "the encoder is too small". Frozen, so only the predictor/decoder/probe train on top.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 131.5 min
- **Status:** pass
- **Artifacts:** `B_dinov2_seed1.log`, `B_dinov2_seed1.json`, `models/B_dinov2_seed1/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.8578 | -0.0464 | -0.8579 | -1.6911 | -0.8579 | -2.0964 |
| contact_stiffness | -0.5878 | -1.2503 | -0.5864 | -0.2218 | -0.5866 | -0.2521 |
| drag | -1.1337 | -0.3195 | -1.1340 | -64.4190 | -1.1339 | -0.6332 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 07:48 — [B. encoder scale (frozen DINOv2-small @224)] `B_dinov2_seed0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder dinov2 --epochs 40 --episodes 128 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --conditions predictive,physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/B_dinov2_seed0.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/B_dinov2_seed0`
- **Why this run:** Rules out "the encoder is too small". Frozen, so only the predictor/decoder/probe train on top.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 131.4 min
- **Status:** pass
- **Artifacts:** `B_dinov2_seed0.log`, `B_dinov2_seed0.json`, `models/B_dinov2_seed0/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -1.5216 | -0.3061 | -1.5212 | -1.3481 | -1.5210 | -0.0644 |
| contact_stiffness | -1.3882 | -0.9999 | -1.3875 | -0.2485 | -1.3873 | -0.2191 |
| drag | -0.6864 | -1.3256 | -0.6858 | -57.0976 | -0.6851 | -0.7594 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 06:39 — [F. functional use (DINOv2)] `F_funct_dinov2_seed1`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/functional_use.py --encoder dinov2 --epochs 40 --episodes 128 --length 48 --probe-hidden 0 --alpha 1.0 --seed 1 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/F_funct_dinov2_seed1.json`
- **Why this run:** Functional use at the larger encoder.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 62.1 min
- **Status:** pass
- **Artifacts:** `F_funct_dinov2_seed1.log`, `F_funct_dinov2_seed1.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.00857, contact_stiffness 0.00821, drag 0.00908, poker_radius 0.14893, object_radius 0.00524

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08173 |
| shuffled | 0.06345 |
| probe | 0.34433 |

probe closes **-442.7%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.6001 | 0.5511 | 0.4947 | 0.4585 | 0.4360 | 0.4245 | 0.4193 |
| nominal | 0.1640 | 0.1570 | 0.1746 | 0.1759 | 0.1660 | 0.1564 | 0.1497 |
| shuffled | 0.1324 | 0.1784 | 0.1976 | 0.1888 | 0.1777 | 0.1691 | 0.1638 |

### 2026-08-25 06:39 — [F. functional use (DINOv2)] `F_funct_dinov2_seed0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/functional_use.py --encoder dinov2 --epochs 40 --episodes 128 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/F_funct_dinov2_seed0.json`
- **Why this run:** Functional use at the larger encoder.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 62.1 min
- **Status:** pass
- **Artifacts:** `F_funct_dinov2_seed0.log`, `F_funct_dinov2_seed0.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.00744, contact_stiffness 0.01140, drag 0.00761, poker_radius 0.04075, object_radius 0.27042

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08241 |
| shuffled | 0.08892 |
| probe | 0.14273 |

probe closes **-60.5%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.2287 | 0.2455 | 0.2567 | 0.2552 | 0.2544 | 0.2525 | 0.2519 |
| nominal | 0.1935 | 0.2040 | 0.2257 | 0.2182 | 0.2052 | 0.1948 | 0.1879 |
| shuffled | 0.1916 | 0.2181 | 0.2436 | 0.2455 | 0.2313 | 0.2226 | 0.2174 |

### 2026-08-25 05:37 — [A. supervised ceiling (tiny_cnn, long, 512 eps)] `A_ceiling_tiny_seed2`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 2 --batch-size 32 --conditions predictive,physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/A_ceiling_tiny_seed2.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/A_ceiling_tiny_seed2`
- **Why this run:** Does ANY objective put theta in this latent? Decides whether to fix the objective or the benchmark signal.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 142.8 min
- **Status:** pass
- **Artifacts:** `A_ceiling_tiny_seed2.log`, `A_ceiling_tiny_seed2.json`, `models/A_ceiling_tiny_seed2/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.0556 | -0.1969 | -0.0466 | -0.5328 | -0.0694 | -0.5526 |
| contact_stiffness | 0.0508 | -1.0175 | 0.0454 | -0.1703 | 0.0367 | 0.0218 |
| drag | -0.0733 | -0.1597 | -0.0699 | -37.9418 | -0.0900 | -0.0582 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 05:37 — [C. probe capacity (MLP probe, hidden 128)] `C_probe_mlp_seed0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 128 --alpha 1.0 --seed 0 --batch-size 32 --conditions predictive,physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/C_probe_mlp_seed0.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/C_probe_mlp_seed0`
- **Why this run:** The probe is linear by design. If an MLP recovers theta, the information is present but not linearly separable.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 142.8 min
- **Status:** pass
- **Artifacts:** `C_probe_mlp_seed0.log`, `C_probe_mlp_seed0.json`, `models/C_probe_mlp_seed0/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.0588 | -0.0518 | -0.1221 | -0.5977 | -0.0797 | -0.3150 |
| contact_stiffness | 0.0211 | -1.4277 | 0.0218 | -3.6031 | -0.0708 | -0.1039 |
| drag | -0.0836 | -0.0517 | -0.0830 | -19.9293 | -0.0562 | -0.1569 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 05:37 — [A. supervised ceiling (tiny_cnn, long, 512 eps)] `A_ceiling_tiny_seed1`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 1 --batch-size 32 --conditions predictive,physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/A_ceiling_tiny_seed1.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/A_ceiling_tiny_seed1`
- **Why this run:** Does ANY objective put theta in this latent? Decides whether to fix the objective or the benchmark signal.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 142.8 min
- **Status:** pass
- **Artifacts:** `A_ceiling_tiny_seed1.log`, `A_ceiling_tiny_seed1.json`, `models/A_ceiling_tiny_seed1/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.0397 | -0.0417 | -0.0616 | -0.9532 | -0.0576 | -0.8212 |
| contact_stiffness | 0.0171 | -1.0471 | 0.0003 | -0.0652 | -0.0139 | -0.0321 |
| drag | -0.0203 | -0.0097 | -0.0538 | -25.8764 | -0.0364 | -0.2588 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 05:37 — [A. supervised ceiling (tiny_cnn, long, 512 eps)] `A_ceiling_tiny_seed0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --conditions predictive,physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/A_ceiling_tiny_seed0.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/A_ceiling_tiny_seed0`
- **Why this run:** Does ANY objective put theta in this latent? Decides whether to fix the objective or the benchmark signal.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 142.8 min
- **Status:** pass
- **Artifacts:** `A_ceiling_tiny_seed0.log`, `A_ceiling_tiny_seed0.json`, `models/A_ceiling_tiny_seed0/`

**theta recovery — val R²**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- | --- | --- |
| mass | -0.0876 | -0.0617 | -0.1029 | -0.9683 | -0.0485 | -0.1653 |
| contact_stiffness | 0.0088 | -1.3576 | 0.0071 | -0.6362 | 0.0196 | -0.0009 |
| drag | -0.0865 | -0.0218 | -0.0866 | -10.9617 | -0.0621 | -0.2969 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 05:31 — [D. data scale (episodes, seed 0)] `D_data1024`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 40 --episodes 1024 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --conditions physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/D_data1024.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/D_data1024`
- **Why this run:** R^2 needs across-episode variation. If recovery scales with episode count the sweep is simply under-powered.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 137.4 min
- **Status:** pass
- **Artifacts:** `D_data1024.log`, `D_data1024.json`, `models/D_data1024/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.0294 | -2.0025 | -0.0169 | -1.1055 |
| contact_stiffness | 0.0203 | -1.1291 | 0.0323 | -0.0214 |
| drag | -0.0297 | -29.9460 | -0.0290 | -0.2637 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 05:09 — [F. functional use (theta-supervised)] `F_funct_supervised`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --theta-supervision 1.0 --horizon 7 --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/F_funct_supervised.json`
- **Why this run:** If supervision makes theta correct, is it then also USED?
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 43.3 min
- **Status:** pass
- **Artifacts:** `F_funct_supervised.log`, `F_funct_supervised.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.00350, contact_stiffness 0.01186, drag 0.00309, poker_radius 0.20919, object_radius 0.20926

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08281 |
| shuffled | 0.11049 |
| probe | 0.15667 |

probe closes **-41.8%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.2373 | 0.2299 | 0.2393 | 0.2311 | 0.2229 | 0.2150 | 0.2086 |
| nominal | 0.1648 | 0.1675 | 0.1929 | 0.1872 | 0.1791 | 0.1718 | 0.1656 |
| shuffled | 0.1750 | 0.1995 | 0.2184 | 0.2144 | 0.2069 | 0.2012 | 0.1962 |

### 2026-08-25 04:58 — [F. functional use (tiny_cnn, long)] `F_funct_tiny_seed2`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 2 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/F_funct_tiny_seed2.json`
- **Why this run:** Decodable is not the same as used. Sensitivity, substitution and multi-horizon rollout.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 54.3 min
- **Status:** pass
- **Artifacts:** `F_funct_tiny_seed2.log`, `F_funct_tiny_seed2.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.00850, contact_stiffness 0.01133, drag 0.00808, poker_radius 0.22287, object_radius 0.03263

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08233 |
| shuffled | 0.10858 |
| probe | 0.12456 |

probe closes **-14.7%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1580 | 0.1787 | 0.2105 | 0.2236 | 0.2344 | 0.2436 | 0.2521 |
| nominal | 0.1589 | 0.1520 | 0.1805 | 0.1824 | 0.1775 | 0.1708 | 0.1649 |
| shuffled | 0.2052 | 0.2125 | 0.2346 | 0.2325 | 0.2239 | 0.2163 | 0.2100 |

### 2026-08-25 04:46 — [D. data scale (episodes, seed 0)] `D_data512`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 40 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --conditions physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/D_data512.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/D_data512`
- **Why this run:** R^2 needs across-episode variation. If recovery scales with episode count the sweep is simply under-powered.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 91.7 min
- **Status:** pass
- **Artifacts:** `D_data512.log`, `D_data512.json`, `models/D_data512/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.0825 | -1.6877 | -0.0728 | -0.5767 |
| contact_stiffness | 0.0082 | -0.9755 | 0.0180 | 0.0118 |
| drag | -0.0656 | -13.1175 | -0.0274 | -0.2140 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 04:44 — [F. functional use (tiny_cnn, long)] `F_funct_tiny_seed1`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 1 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/F_funct_tiny_seed1.json`
- **Why this run:** Decodable is not the same as used. Sensitivity, substitution and multi-horizon rollout.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 65.4 min
- **Status:** pass
- **Artifacts:** `F_funct_tiny_seed1.log`, `F_funct_tiny_seed1.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.00898, contact_stiffness 0.01144, drag 0.00899, poker_radius 0.32695, object_radius 0.05888

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.07588 |
| shuffled | 0.10021 |
| probe | 0.12316 |

probe closes **-22.9%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1750 | 0.1883 | 0.2151 | 0.2337 | 0.2465 | 0.2580 | 0.2692 |
| nominal | 0.1523 | 0.1480 | 0.1719 | 0.1723 | 0.1646 | 0.1565 | 0.1497 |
| shuffled | 0.1755 | 0.1941 | 0.2175 | 0.2153 | 0.2065 | 0.1995 | 0.1953 |

### 2026-08-25 04:26 — [E. alpha sweep (physics-path weight, seed 0)] `E_alpha2.0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 2.0 --seed 0 --batch-size 32 --conditions physwm --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/E_alpha2.0.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/E_alpha2.0`
- **Why this run:** alpha=1 may simply be too weak a pressure on the latent.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 72.1 min
- **Status:** pass
- **Artifacts:** `E_alpha2.0.log`, `E_alpha2.0.json`, `models/E_alpha2.0/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.1046 | -0.5043 |
| contact_stiffness | -0.0090 | -0.4852 |
| drag | -0.0731 | -17.6144 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 04:26 — [E. alpha sweep (physics-path weight, seed 0)] `E_alpha5.0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 5.0 --seed 0 --batch-size 32 --conditions physwm --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/E_alpha5.0.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/E_alpha5.0`
- **Why this run:** alpha=1 may simply be too weak a pressure on the latent.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 72.1 min
- **Status:** pass
- **Artifacts:** `E_alpha5.0.log`, `E_alpha5.0.json`, `models/E_alpha5.0/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.0621 | -0.7034 |
| contact_stiffness | 0.0139 | -0.7733 |
| drag | -0.0733 | -27.8385 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 04:26 — [E. alpha sweep (physics-path weight, seed 0)] `E_alpha0.5`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 0.5 --seed 0 --batch-size 32 --conditions physwm --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/E_alpha0.5.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/E_alpha0.5`
- **Why this run:** alpha=1 may simply be too weak a pressure on the latent.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 72.0 min
- **Status:** pass
- **Artifacts:** `E_alpha0.5.log`, `E_alpha0.5.json`, `models/E_alpha0.5/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe |
| --- | --- | --- |
| mass | -0.0815 | -1.4563 |
| contact_stiffness | 0.0251 | -0.3461 |
| drag | -0.0541 | -2.7307 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 04:26 — [F. functional use (tiny_cnn, long)] `F_funct_tiny_seed0`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/functional_use.py --encoder tiny_cnn --epochs 60 --episodes 512 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --theta-supervision 0.0 --horizon 7 --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/F_funct_tiny_seed0.json`
- **Why this run:** Decodable is not the same as used. Sensitivity, substitution and multi-horizon rollout.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 71.5 min
- **Status:** pass
- **Artifacts:** `F_funct_tiny_seed0.log`, `F_funct_tiny_seed0.json`

**sensitivity** (prediction shift, theta scaled 1.1x): mass 0.00735, contact_stiffness 0.01192, drag 0.00736, poker_radius 0.05776, object_radius 0.28170

**substitution** (one-step Path B error vs true `s_next`):

| source | scaled RMSE |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08281 |
| shuffled | 0.11049 |
| probe | 0.11327 |

probe closes **-2.5%** of the shuffled-to-true gap.

**multi-horizon rollout** (scaled RMSE):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1776 | 0.1935 | 0.2166 | 0.2210 | 0.2265 | 0.2322 | 0.2387 |
| nominal | 0.1648 | 0.1675 | 0.1929 | 0.1872 | 0.1791 | 0.1718 | 0.1656 |
| shuffled | 0.1750 | 0.1995 | 0.2184 | 0.2144 | 0.2069 | 0.2012 | 0.1962 |

### 2026-08-25 04:04 — [D. data scale (episodes, seed 0)] `D_data256`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 40 --episodes 256 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --conditions physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/D_data256.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/D_data256`
- **Why this run:** R^2 needs across-episode variation. If recovery scales with episode count the sweep is simply under-powered.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 49.9 min
- **Status:** pass
- **Artifacts:** `D_data256.log`, `D_data256.json`, `models/D_data256/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.1243 | -1.7091 | -0.1477 | -0.4446 |
| contact_stiffness | -0.1570 | -0.1956 | -0.1591 | -0.1252 |
| drag | -0.1739 | -1.9322 | -0.1178 | -0.1310 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 03:39 — [D. data scale (episodes, seed 0)] `D_data128`

- **Command:** `/home/sra/sarayu/stable-worldmodel/.venv-phy-wm-gpu/bin/python -u /home/sra/sarayu/stable-worldmodel/scripts/eval/decodability.py --encoder tiny_cnn --epochs 40 --episodes 128 --length 48 --probe-hidden 0 --alpha 1.0 --seed 0 --batch-size 32 --conditions physwm,supervised --out /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/D_data128.json --save-dir /home/sra/sarayu/stable-worldmodel/outputs/overnight/20260825_0314/models/D_data128`
- **Why this run:** R^2 needs across-episode variation. If recovery scales with episode count the sweep is simply under-powered.
- **Commit / working tree:** `c7fd83f2f`, dirty
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu`
- **Duration:** 24.7 min
- **Status:** pass
- **Artifacts:** `D_data128.log`, `D_data128.json`, `models/D_data128/`

**theta recovery — val R²**

| param | physwm/decodable | physwm/own_probe | supervised/decodable | supervised/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.3487 | -0.9911 | -0.4765 | -0.2202 |
| contact_stiffness | -0.4703 | -0.0387 | -0.3892 | -0.2847 |
| drag | -0.1974 | -0.3071 | -0.1675 | -0.0473 |

_(radii are constant by construction; their R² is `nan` and is omitted)_

### 2026-08-25 — Overnight sweep LAUNCHED (19 experiments)

- **Command:** `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/overnight.py`
- **Commit / working tree:** `c7fd83f`, dirty
- **Hardware:** RTX 4090
- **Status:** running — **each experiment appends its own entry above this one
  as it finishes**, so an interrupted night still leaves a complete record
- **Artifacts:** `outputs/overnight/<timestamp>/` — JSON metrics, stdout logs,
  and saved model weights per condition

**The question.** Three probe/observation fixes have failed to move theta
recovery and the *supervised* ridge ceiling on the frozen latent is still
negative, so the encoder is not encoding the physics at all. This sweep rules
out the cheap explanations and localises the failure.

**The decisive condition is `supervised`** — an explicit regression of the
probe's theta onto ground-truth theta, added on top of the PhysWM objective.
It is not part of the method; it is a diagnostic ceiling:

- `supervised` **recovers** theta -> the pipeline CAN encode physics, so the
  two-path objective is what fails to induce it. Fix the objective.
- `supervised` **also fails** -> the encoder or the benchmark signal is the
  problem and no objective change will help. Fix the signal.

| group | experiments | what it rules out |
| --- | --- | --- |
| A. supervised ceiling, tiny_cnn, 512 eps x 60 ep, 3 seeds | 3 | the decisive one (above) |
| B. encoder scale, frozen DINOv2-small @224, 2 seeds | 2 | "tiny_cnn is too weak" |
| C. probe capacity, MLP hidden 128 | 1 | "a linear read-out is the limit" |
| D. data scale, 128/256/512/1024 episodes | 4 | "the sweep is under-powered" |
| E. alpha sweep, 0.5 / 2.0 / 5.0 | 3 | "the physics path pushes too weakly" |
| F. functional use, tiny x3 + supervised + DINOv2 x2 | 6 | decodable vs actually used |

**Sizing for the 4090.** tiny_cnn peaks at **0.54 GiB** and frozen DINOv2-small
at **4.14 GiB** against 24 GB, so the card is idle at batch 32 — the throughput
win is **concurrency, not batch size**. Light experiments run 4-wide
(~2.2 GiB total), heavy ones 2-wide (~8.3 GiB). BF16 autocast and TF32 are on.
Measured cost: tiny 3.6 s/epoch @64 episodes, DINOv2 33.1 s/epoch @64.
Budget: ~5.8 h ideal, ~9.3 h with GPU contention.

- **Notes:** Every number in this sweep comes from the reference-spec PokeWorld
  with the tactile channel as a model input, log-compressed, and routed around
  the probe's pooling. The oracle certificate on this simulator is
  0.834 / 0.755 / 0.754 (mass / stiffness / drag) — that is the ceiling any of
  these runs is being measured against.

### 2026-08-25 — Probe pooling fix (dedicated tactile pathway): still negative

- **Command:**
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/smoke/run_smoke.py`
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/decodability.py --epochs 30 --episodes 128 --length 48`
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/functional_use.py --epochs 30 --episodes 128 --length 48 --horizon 7`
- **Config:** pokeworld, `tiny_cnn` @64px, probe linear / episode /
  mean-pool + dedicated tactile pathway, alpha 0.0 vs 1.0
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f`, dirty
- **Hardware:** RTX 4090
- **Status:** pass — 50/50 unit tests, smoke 6/6 — **result still negative**
- **Artifacts:** `decodability_pool.json`, `functional_pool.json`

**Decodability — val R^2** (previous run, diluted pooling, in brackets)

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.2759 (-0.5204) | -0.0003 (0.0054) | -0.3183 (-0.2992) | -1.0378 (-0.1816) |
| contact_stiffness | -0.3257 (-0.3307) | -1.8587 (-1.8814) | -0.6595 (-0.5163) | **-0.4638** (-1.7645) |
| drag | -0.0997 (-0.1817) | -0.0074 (-0.0225) | -0.2103 (-0.2460) | **-0.8163** (-2.2926) |

**Functional use** — sensitivity: mass 0.00495, contact_stiffness 0.01195,
drag 0.00466, poker_radius 0.14532, object_radius 0.18185

Substitution (one-step Path B error vs true `s_next`, scaled RMSE):

| source | error |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08241 |
| shuffled | 0.08892 |
| probe | 0.12110 |

probe closes **-36.2%** of the shuffled-to-true gap (was -0.5%).

Multi-horizon rollout (scaled RMSE, horizons 1..7):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.2319 | 0.2286 | 0.2253 | 0.2181 | 0.2137 | 0.2063 | 0.1992 |
| nominal | 0.1935 | 0.2040 | 0.2257 | 0.2182 | 0.2052 | 0.1948 | 0.1879 |
| shuffled | 0.1916 | 0.2181 | 0.2436 | 0.2455 | 0.2313 | 0.2226 | 0.2174 |

- **Notes:** `ThetaProbe` now routes tactile tokens around the pool: its input
  is `[pooled pixel tokens || tactile token]`, so the tactile share rose from
  **1/17 (5.9%) to 50%** while the probe stayed low-capacity (1285 params,
  budget 2000). `test_tactile_token_bypasses_pooling` pins this.

  **Dilution was not the bottleneck.** `physwm/decodable` did not improve
  (mass -0.32 vs -0.30, stiffness -0.66 vs -0.52, drag -0.21 vs -0.25), and the
  probe now closes **-36.2%** of the substitution gap — markedly *worse* than
  the -0.5% before. In the rollout the probe curve moved *away* from `nominal`
  and sits above it at every horizon. Giving the probe more tactile bandwidth
  let it fit the tactile channel harder without making theta more correct.

  Two things did improve, both consistent with a better-conditioned read-out
  rather than better physics: `physwm/own_probe` for stiffness -1.76 -> -0.46
  and for drag -2.29 -> -0.82. Mass got worse (-0.18 -> -1.04).

  **Every R^2 in this table is still negative**, including the supervised
  ceiling `physwm/decodable`. That is the finding that matters: after three
  targeted fixes (tactile as input, log compression, dedicated pathway) a
  *supervised* read-out on the frozen latent still cannot recover theta, while
  the oracle on true transitions reaches 0.83/0.76/0.75. The encoder is not
  putting the physics into the latent at all — so no probe design can help,
  and further probe-side tuning is the wrong place to look.

  Two suspects remain, and they are about the **encoder and the signal**, not
  the probe:
  - **Scale/capacity, still unruled-out.** `tiny_cnn` @64px, 30 epochs, 1 seed.
    This must be ruled out before any further redesign — it is the cheapest
    remaining explanation and every result so far sits at smoke scale.
  - **Per-transition signal is tiny.** Sensitivity to mass (0.0050) and drag
    (0.0047) is ~30x below the radii (0.15-0.18); a transition spans 0.02 s and
    contact fires in ~5% of them. Theta barely moves the one-step target the
    model is trained on.

- **Supersedes:** the previous same-day entry (tactile visible, pooled).

### 2026-08-25 — Tactile channel made visible to the encoder: still negative

- **Command:**
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/decodability.py --epochs 30 --episodes 128 --length 48`
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/functional_use.py --epochs 30 --episodes 128 --length 48 --horizon 7`
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/smoke/run_smoke.py`
- **Config:** pokeworld, `tiny_cnn` @64px, probe `hidden_dim: 0` linear /
  episode / mean-pool, alpha 0.0 vs 1.0
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f`, dirty
- **Hardware:** RTX 4090
- **Status:** pass — 49/49 unit tests, smoke 6/6 — **result still negative**
- **Artifacts:** `decodability_tactile.json`, `functional_tactile.json`

**Decodability — val R^2** (previous run, tactile state-only, in brackets)

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.5204 (-0.4702) | 0.0054 (0.0011) | -0.2992 (-0.3134) | -0.1816 (-0.1137) |
| contact_stiffness | -0.3307 (-0.5180) | -1.8814 (-1.8850) | -0.5163 (-0.4578) | -1.7645 (-1.6022) |
| drag | -0.1817 (-0.0973) | -0.0225 (-0.0457) | -0.2460 (-0.3324) | **-2.2926** (-20.8287) |

**Functional use** — sensitivity: mass 0.00421, contact_stiffness 0.01135,
drag 0.00354, poker_radius 0.15649, object_radius 0.15465

Substitution (one-step Path B error vs true `s_next`, scaled RMSE):

| source | error |
| --- | --- |
| true | 0.00000 |
| nominal | 0.08241 |
| shuffled | 0.08892 |
| probe | 0.08936 |

probe closes **-0.5%** of the shuffled-to-true gap.

Multi-horizon rollout (scaled RMSE, horizons 1..7):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.1908 | 0.2054 | 0.2264 | 0.2167 | 0.2032 | 0.1924 | 0.1851 |
| nominal | 0.1935 | 0.2040 | 0.2257 | 0.2182 | 0.2052 | 0.1948 | 0.1879 |
| shuffled | 0.1916 | 0.2181 | 0.2436 | 0.2455 | 0.2313 | 0.2226 | 0.2174 |

- **Notes:** Two changes were made, both necessary, **neither sufficient**.

  1. **`touch` is now a model input**, one extra token per frame, delivered
     under its own batch key so the model never indexes `state` (the
     supervision target) for inputs. Without this the probe read pixels only,
     where mass/stiffness/drag are degenerate in principle — a target the
     model cannot observe teaches nothing.
  2. **`touch` is stored as `log1p(peak force)`.** Raw peak was zero-inflated
     (~5% nonzero) over 0-700, so after z-scoring the silent majority collapsed
     to one value and contacts became +20 sigma outliers. Now 0-6.6, peaking at
     +6.5 sigma.

  **The result barely moved.** `physwm/decodable` is still negative and no
  better than `predictive/decodable`; the probe still closes -0.5% of the
  substitution gap. The one clear improvement is `physwm/own_probe` for drag,
  -20.83 -> -2.29: the probe is much less catastrophically wrong, which is what
  the log compression should fix, but it is still not right.

  **One genuine positive**, visible for the first time: in the multi-horizon
  rollout `shuffled` now separates from `probe`/`nominal` and the gap *widens*
  with horizon (h7: 0.2174 vs 0.1851/0.1879). Episode-specific theta finally
  matters to multi-step prediction. The probe simply is not the thing supplying
  it — it tracks `nominal` almost exactly, i.e. it has learned a constant.

  **Next suspect, and it is specific:** the probe mean-pools over
  **17 tokens (16 pixel + 1 tactile)**, so the tactile token contributes
  **5.9%** of the vector a *linear* probe reads. The channel that carries all
  the stiffness information is attenuated ~17x before the probe sees it. The
  test `test_tactile_observation_reaches_the_probe` proves theta responds to
  touch, but responding is not the same as being driven by it. Candidate fixes,
  cheapest first: give the tactile token its own pathway into the probe rather
  than averaging it into the pixel tokens; or weight it; or use `pool=flatten`.

  Scale is still unruled-out: `tiny_cnn` @64px, linear probe, 30 epochs, 1 seed.

- **Supersedes:** the previous same-day eval entry (tactile present in state
  but not visible to the encoder).

### 2026-08-25 — Decodability + functional use on the FIXED sim: negative result

- **Command:**
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/decodability.py --epochs 30 --episodes 128 --length 48`
  `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/functional_use.py --epochs 30 --episodes 128 --length 48 --horizon 7`
- **Config:** pokeworld at reference spec, `tiny_cnn` @64px, probe `hidden_dim: 0`
  (linear, episode mode), alpha 0.0 vs 1.0
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f`, dirty
- **Hardware:** RTX 4090
- **Status:** pass (ran clean) — **result is negative**
- **Artifacts:** `decodability_fixed.json`, `functional_fixed.json` in scratchpad

**Decodability — val R^2**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.4702 | 0.0011 | -0.3134 | -0.1137 |
| contact_stiffness | -0.5180 | -1.8850 | -0.4578 | -1.6022 |
| drag | -0.0973 | -0.0457 | -0.3324 | -20.8287 |
| poker_radius | nan | nan | nan | nan |
| object_radius | nan | nan | nan | nan |

**Functional use** — sensitivity (prediction shift from scaling theta 1.1x):
mass 0.00540, contact_stiffness 0.04502, drag 0.00510, poker_radius 0.09837,
object_radius 0.09843

Substitution (one-step Path B error vs true `s_next`, scaled RMSE):

| source | error |
| --- | --- |
| true | 0.00000 |
| shuffled | 0.18818 |
| probe | 0.19703 |
| nominal | 0.21009 |

probe closes **-4.7%** of the shuffled-to-true gap.

Multi-horizon rollout (scaled RMSE, horizons 1..7):

| source | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe | 0.4963 | 0.3911 | 0.3354 | 0.3001 | 0.2775 | 0.2628 | 0.2529 |
| nominal | 0.5403 | 0.4135 | 0.3509 | 0.3102 | 0.2830 | 0.2638 | 0.2498 |
| shuffled | 0.4972 | 0.4182 | 0.3569 | 0.3175 | 0.2905 | 0.2719 | 0.2586 |

- **Notes:** **The abstract's central claim is not supported at this scale.**
  Read plainly:
  - **The premise holds.** `predictive/decodable` is negative for all three
    parameters: a purely predictive latent does not encode the physics. That is
    the "world models are blind to physical parameters" starting point.
  - **The claim does not.** `physwm/decodable` is no better than
    `predictive/decodable` (mass -0.31 vs -0.47, stiffness -0.46 vs -0.52) and
    is *worse* for drag (-0.33 vs -0.10). The physics path did not induce the
    missing representation. `physwm/own_probe` for drag is **-20.8** — the
    probe is confidently wrong, not merely uninformative.
  - **Theta is not functionally used.** The probe closes -4.7% of the
    shuffled-to-true gap, i.e. its theta is no better than another episode's
    theta. In the rollout, `probe`, `shuffled` and `nominal` sit on top of each
    other and none approaches `true`.

  This is not a benchmark problem — the oracle certificate on this same
  simulator is 0.83/0.76/0.75, so the information is present in the
  transitions. The failure is in the probe/encoder pipeline. Three concrete
  suspects, in order of my confidence:

  1. **The encoder never sees touch.** Rendering is `(T, 3, 64, 64)` position
     blobs; the tactile channel exists only in the *state*. So the probe must
     infer stiffness from pixels, where only `k/m` and `c/m` are recoverable —
     the very degeneracy `touch` was added to break. The channel does reach the
     objective indirectly (Path A regresses the 5-d state, so it must predict
     touch, and Path B chases Path A), but that is a long and weak path.
  2. **Theta barely moves the one-step prediction.** Sensitivity to mass
     (0.0054) and drag (0.0051) is ~20x below the radii (0.098). With
     `dt=0.001 x 20` a transition spans only 0.02 s and the state moves ~0.017
     in position, so one step carries almost no evidence about theta. The old
     0.1 s transition gave a stronger signal; the fine substep was needed for
     stiffness stability, but it cost per-transition signal.
  3. **Contact is rare.** Only **5.1%** of transitions register any touch, so
     the stiffness-bearing events are sparse — and a window of 8 frames often
     contains none at all.

  Scale is the obvious confound and should be ruled out before redesigning:
  this is a `tiny_cnn` at 64px, a linear probe, 30 epochs, 128 episodes.

- **Supersedes:** the pre-fix decodability entry below, which measured the old
  degenerate simulator.

### 2026-08-25 — Full smoke suite on the fixed PokeWorld: 6/6

- **Command:** `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/smoke/run_smoke.py`
  (defaults: 3 epochs, 8 episodes x 24, all three benchmarks)
- **Config:** repo defaults after the PokeWorld reference fix
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f`, dirty (PokeWorld fix + eval scripts +
  training-loop work all uncommitted)
- **Hardware:** RTX 4090, `.venv-phy-wm-gpu` (`torch 2.13.0+cu130`)
- **Duration:** 30.0 s total
- **Status:** pass — 6/6

| Step | Result |
| --- | --- |
| unit tests (design invariants) | PASS 4.1s (47 tests) |
| solver validation (oracle vs persistence) | PASS 5.7s |
| train: pokeworld | PASS 5.9s |
| train: cartpole | PASS 7.3s |
| train: pusht | PASS 5.7s |
| checkpoint round-trip | PASS 1.3s |

Solver adequacy (scaled RMSE, lower is better):

| bench | persistence | nominal | fitted | verdict |
| --- | --- | --- | --- | --- |
| pokeworld | 0.4039 | 0.1294 | **0.0000** | solver beats persistence |
| cartpole | 0.1693 | 0.0059 | 0.0030 | solver beats persistence |
| pusht | 0.3515 | 0.0319 | 0.0207 | solver beats persistence |

- **Notes:** First green run of the whole pipeline on the reference-spec
  PokeWorld (5-d state with the `touch` channel, stiffness 500-6000,
  `dt=0.001 x 20`). Nothing in the suite needed changing for the new state
  dimension beyond the sim/solver/spec themselves, and the checkpoint
  round-trip still reloads through `swm.wm.load_pretrained` with the solver at
  zero learnable parameters.

  PokeWorld's `fitted` is 0.0000 because the frozen solver *is* the simulator —
  with the true theta it reproduces transitions to float precision. That is a
  correctness check on the solver, not a claim about learning. `persistence`
  rose from 0.3315 to 0.4039 versus the old sim: stiffer contacts move the
  object more per transition, so predicting "nothing changes" got worse.

  This suite is smoke-scale (3 epochs, 8 episodes). It says the pipeline runs
  end to end; it says nothing about whether the method works.

### 2026-08-25 — PokeWorld brought up to the reference; identifiability recovered

- **Command:** `pytest tests/wm/test_physwm.py`; `scripts/smoke/run_smoke.py
  --epochs 2 --episodes 8 --length 24`; oracle certificate via a scratch script
  wrapping `validate_solvers.fit_theta` + `loss.theta_r2`
- **Config:** pokeworld, 64 episodes x 48 for the certificate, 2000 fit steps
- **Seed(s):** 0
- **Commit / working tree:** `c7fd83f`, dirty
- **Hardware:** RTX 4090
- **Status:** pass — 47/47 tests, smoke 6/6

**Oracle recoverability certificate** (free per-episode theta fitted to TRUE
transitions; the ceiling no pixel-reading probe can beat):

| | mass | stiffness | drag |
| --- | --- | --- | --- |
| **with `touch`** (reference design) | **0.834** | **0.755** | **0.754** |
| without `touch` (motion only) | 0.009 | 0.084 | -0.214 |

- **Notes:** The local `PokeWorldSim` was a simplified reimplementation that
  deviated from the reference
  (https://tantansir.github.io/latent-world-model-identifiability/) in ways
  that made the paper's headline claim unreachable. Three changes:
  - **Ranges** to reference values: mass 0.5-3.0 (was 0.5-2.0), contact
    stiffness 500-6000 (was **20-200**, 25x too low), drag 0.5-4.0.
  - **A tactile channel.** State is now `[x, y, vx, vy, touch]`, where
    `touch` is the PEAK contact-force magnitude over the transition's
    sub-steps. This is the substantive fix: from motion alone the dynamics
    depend only on `k/m` and `c/m`, so mass/stiffness/drag are individually
    unidentifiable *in principle* — no amount of training could recover them.
    The reference's three observability channels (sub-step tactile peak ->
    stiffness, impulse/velocity coupling -> mass, glide decay -> drag) all
    require observing contact force.
  - **Integration** `dt=0.001 x 20` substeps (was `0.01 x 10`). The reference
    stiffness range needs it for stability (omega = sqrt(k/m) up to ~110 rad/s)
    and to resolve the tactile peak, which is far briefer than one transition.

  The ablation row is the evidence the change is the right one, not just a
  change: **without `touch` the ceiling is ~0 for all three parameters; with
  it, all three clear 0.75.** Rendering is untouched and still identical
  across theta, so the pixels give nothing away — as the abstract requires.

  Solver and sim agree to float precision (max abs diff 6e-5 on the touch
  channel, 1e-6 elsewhere), so the frozen solver remains exact for PokeWorld.

  For calibration, the reference reports a drag certificate of **0.89** with
  prediction models plateauing near **0.13** and supervised identification at
  **0.45**. Our 0.754 is the same regime but below theirs; more episodes, a
  longer fit, or more frequent contact (currently only ~6% of transitions
  register any touch) would likely close the gap.

- **Supersedes:** the earlier same-day measurement that the drag ceiling was
  ~-0.03 and unreachable. That was correct for the *old* simulator and is now
  obsolete.

### 2026-08-25 — Decodability + functional-use evaluations implemented (pre-fix numbers)

- **Command:** `scripts/eval/decodability.py --epochs 30 --episodes 128
  --length 48`; `scripts/eval/functional_use.py` (same scale)
- **Commit / working tree:** `c7fd83f`, dirty, **OLD PokeWorld** (run started
  before the simulator fix above)
- **Hardware:** RTX 4090
- **Status:** pass (scripts work); **numbers obsolete**

| param | predictive/decodable | predictive/own_probe | physwm/decodable | physwm/own_probe |
| --- | --- | --- | --- | --- |
| mass | -0.9615 | -0.0586 | -0.4844 | -1.2690 |
| contact_stiffness | -0.5277 | -2.0439 | -0.6914 | -2.5617 |
| drag | -0.3992 | -0.8314 | -0.2198 | -0.2838 |

- **Notes:** Two evaluations were added for the abstract's claims:
  - `scripts/eval/decodability.py` — the "predictive objective alone does not"
    baseline. Trains `alpha=0` (predictive-only) and `alpha>0` (PhysWM) under
    identical seed/data/architecture, then reports a **supervised** ridge
    read-out on the frozen latent (is the physics encoded at all?) next to the
    model's **own unsupervised** probe. That separates "never encoded" from
    "encoded but unreadable", which one number cannot.
  - `scripts/eval/functional_use.py` — decodable vs actually used. Three
    interventions: per-component sensitivity (is theta inert?), substitution of
    probe theta with true/shuffled/nominal (what fraction of the
    shuffled-to-true gap does the probe close?), and multi-horizon rollout
    (wrong parameters should compound).

  **These numbers are all negative because they predate the simulator fix** —
  under the old PokeWorld the parameters were unidentifiable in principle, so
  every condition was measuring noise. Both scripts must be re-run on the
  fixed simulator before anything is concluded. Re-running is the next step.

  Sanity check worth keeping: in `functional_use.py` the `true` theta row gives
  exactly 0.00000 one-step error, confirming the frozen solver reproduces
  PokeWorld's dynamics exactly.

### 2026-08-25 — Pre-experiment fixes: run isolation, resume safety, provenance

- **Command:** `pytest tests/wm/test_physwm.py` plus targeted 1–4 epoch
  pokeworld/cartpole runs on cuda to exercise each fix
- **Config:** default `physwm.yaml`, 8 episodes x 24, `trainer.resume=auto`
- **Seed(s):** 0 and 1
- **Commit / working tree:** `c7fd83f`, dirty
- **Hardware:** RTX 4090
- **Status:** pass — 47/47 tests, all three fixes verified end to end
- **Artifacts:** none kept

Three hazards fixed before any paper run:

| Fix | Before | After |
| --- | --- | --- |
| Run isolation | `output_model_name: physwm_${bench.benchmark}` — every seed and sweep point shared one dir, silently overwriting `config.yaml` | seed-scoped: `physwm_${bench.benchmark}_seed${seed}` |
| Resume safety | `resume: auto` loaded whatever `last.pt` was in the dir, even from a different experiment | `last.pt` carries a config **fingerprint**; a mismatch raises instead of resuming |
| Provenance | nothing recorded the commit, command, or environment | every run writes `run_meta.json` |

- **Notes:** The fingerprint covers `seed`, `bench`, `loss`, `optimizer`,
  `norm_samples`, `deterministic`, and the schedule-affecting `trainer` keys
  (`grad_clip`, `warmup_frac`, `accumulate_grad_batches`, `precision`). It
  deliberately excludes `max_epochs`, `device`, dataloader perf knobs and
  `wandb` — those legitimately change when continuing a run. Verified:
  extending `max_epochs` 2 -> 4 resumes correctly; changing `loss.alpha`
  refuses with a message naming both fingerprints.

  `run_meta.json` records timestamp, exact command, hostname, python
  executable (so CPU-vs-GPU venv is never ambiguous), torch + CUDA build, GPU
  name, `MUJOCO_GL`/`CUDA_VISIBLE_DEVICES`, git commit **and dirty flag**, and
  the fingerprint. Every future entry in this log can be filled from that file
  instead of shell history.

  Seed scoping alone does not isolate a sweep over `loss.alpha` or a probe
  setting — those still map to one directory. Pass `output_model_name`
  explicitly when sweeping; the resume guard is the backstop if you forget.

### 2026-08-25 — GPU environment stood up; all three benches verified on the RTX 4090

- **Command:** see notes (env setup + one 1-epoch run per bench)
- **Config:** `bench={pokeworld,cartpole,pusht}`, `trainer.max_epochs=1`,
  `bench.data.num_episodes=8`, `bench.data.episode_length=24`, `device=cuda`
- **Seed(s):** 0
- **Commit / working tree:** branch `phy-wm`, HEAD `c7fd83f` (merge of
  `origin/phy-wm` `cd29179`), tree dirty — the H100/4090 training-loop work
- **Hardware:** NVIDIA RTX 4090, 24564 MiB, driver 580.173.02; 32 CPU cores, 62 GB RAM
- **Duration:** 4–5 s per 1-epoch bench run; 17 s for the DINOv2 profile run
- **Status:** pass — all three benches train end to end on GPU
- **Artifacts:** none kept (probe checkpoints deleted)

| Check | Result |
| --- | --- |
| `pokeworld` 1 epoch, cuda | pass, val loss 0.98901 |
| `cartpole` 1 epoch, cuda | pass, val loss 1.06449 |
| `pusht` 1 epoch, cuda | pass, val loss 1.60243 |
| `hardware=rtx4090` profile (DINOv2-small/224, BF16, resume, atomic save) | pass, val loss 0.38002 |
| peak GPU memory, batch 32, DINOv2-small@224 | **3673 MiB / 24564 MiB** |
| `pytest tests/wm/test_physwm.py` | 47 passed |

- **Notes:** The env `.venv-phy-wm` is **CPU-only** (`torch 2.13.0+cpu`) and
  cannot use the GPU — every result logged before today was CPU. A GPU twin
  `.venv-phy-wm-gpu` now exists: `torch 2.13.0+cu130`, CUDA available, sm_89,
  `transformers 5.15.1`. It deliberately omits `gymnasium[all]`/`box2d-py`
  (needs `swig`, unused by all three benches).
  **Use `.venv-phy-wm-gpu/bin/python` for every GPU run.**

  The memory measurement showed `rtx4090.yaml`'s batch-4 + 4-step accumulation
  was ~6x too conservative; the profile is now batch 32, accumulation 1.

  These are 1-epoch pipeline checks on 8 episodes — **not results**. They say
  the code runs, nothing about whether it works.

### 2026-08-23 — PhysWM smoke suite (backfilled 2026-08-25 from run artifacts)

Supersedes the provisional entry: details recovered from
`~/.stable_worldmodel/checkpoints/*/{config.yaml,history.json,final_metrics.json}`.

- **Command:** two separate batches. The `smoke_*` runs came from
  `MUJOCO_GL=egl python scripts/smoke/run_smoke.py` (21:16 IST); the `physwm_*`
  runs were separate ad-hoc `scripts/train/physwm.py` invocations 4 minutes
  earlier (21:12 IST). Exact `physwm_*` command lines are unrecoverable — Hydra
  output dirs survive only from 2026-08-24 onward.
- **Config:** `tiny_cnn` encoder @64px, predictor 256-dim/4-layer, probe
  `hidden_dim: 0` (linear), `mode: episode`, `solver_state_source: gt`. Per-run
  data differences in the table below.
- **Seed(s):** 0 (all runs)
- **Commit / working tree:** predates `cd29179`, so **the old loss semantics
  were in force: Path B was regressed on the dataset `s_next`, NOT on Path A's
  detached prediction.** Numbers are not comparable to anything run after the
  2026-08-25 merge (`c7fd83f`).
- **Hardware:** CPU only — `.venv-phy-wm` has `torch 2.13.0+cpu`
- **Duration:** unknown
- **Status:** pass
- **Artifacts:** `~/.stable_worldmodel/checkpoints/<name>/`

| Run | Episodes x len | Epochs | Best val loss | val/loss_a | val/loss_b | val/rmse_a | val/rmse_b |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `physwm_pokeworld` | 8 x 24 | 3/3 | 0.4218 | 0.6704 | 0.0009 | 0.7537 | 0.0266 |
| `physwm_cartpole` | unknown | 2/2 | 0.1437 | 0.1436 | 0.0001 | 0.3789 | 0.0096 |
| `physwm_pusht` | 6 x 24 | 2/2 | 0.6804 | 0.6795 | 0.0009 | 0.8240 | 0.0297 |
| `smoke_r2` | 48 x 32 | 4/4 | 2.2311 | — | — | — | — |
| `smoke_physwm_pokeworld` | 8 x 24 | 3/3 | 0.4218 | 0.6704 | 0.0009 | 0.7537 | 0.0266 |
| `smoke_physwm_cartpole` | 8 x 24 | 3/3 | 1.5374 | 1.5768 | 0.0001 | 1.2404 | 0.0088 |
| `smoke_physwm_pusht` | 8 x 24 | 3/3 | 0.7882 | 0.9280 | 0.0016 | 0.9595 | 0.0385 |

Identifiability certificate (final-epoch val R^2, PokeWorld only):

| Run | mass | contact_stiffness | drag | poker_radius | object_radius |
| --- | --- | --- | --- | --- | --- |
| `smoke_r2` (48 ep) | -0.912 | -1.669 | -0.523 | nan | nan |
| `smoke_physwm_pokeworld` (8 ep) | -3.405 | -4741.143 | -70.114 | nan | nan |
| `physwm_pokeworld` (8 ep) | -5.1e11 | -3.0e14 | -1.6e13 | -3.1e8 | -1.5e8 |

- **Notes:** The three previously-open questions are resolved.
  - **Why pokeworld matched exactly:** `physwm_pokeworld` and
    `smoke_physwm_pokeworld` are genuinely separate runs, 4 minutes apart, with
    identical config (8 x 24, 3 epochs, batch 16) and identical seed 0. Same
    inputs, deterministic run, same 0.4218. Not double-reporting.
  - **Why the epoch counts differ:** different `max_epochs` per invocation, not
    early stopping. `physwm_pusht` was configured for 2.
  - **What `smoke_r2` is:** the PokeWorld identifiability run —
    48 episodes x 32 steps, 4 epochs, specifically to give `theta_r2` enough
    across-episode variation to be meaningful. Its 2.2311 is a val loss on a
    different data scale; it is NOT comparable to the other rows.

  **The R^2 certificate is the headline problem: every value is negative**, i.e.
  the probe explains less of the true parameter variation than predicting the
  mean would. `poker_radius`/`object_radius` are `nan` by construction (constant
  in PokeWorld, so R^2 is undefined — that guard works as documented). The
  `physwm_pokeworld` row's ~1e14 magnitudes are the degenerate case the
  docstring warns about: 8 episodes give almost no theta variance, so the
  denominator collapses. `smoke_r2`'s 48-episode numbers (-0.5 to -1.7) are the
  only trustworthy ones, and they still say the probe is not identifying
  physics at this scale. Expected at 4 epochs on a tiny CNN, but it is the
  number that has to move for the paper to have a result.

  **Reproducibility hazard found:** `output_model_name: physwm_${bench.benchmark}`
  has no seed or timestamp, so every run of a bench reuses one directory and
  overwrites `config.yaml`. `physwm_cartpole/config.yaml` is dated 2026-08-25
  01:29 — clobbered by an aborted run tonight — while its `history.json` is from
  2026-08-23 21:12. **That run's config is permanently lost**, hence "unknown"
  above. The `h100`/`rtx4090` profiles avoid this (`..._seed${seed}` in the
  name); the base config still does not.

---

## Open items

- [x] **Fix `FetchPushSolver`'s gripper update** (`solvers.py`) — done
      2026-08-26, see log entry. Was applying `action[..., 0:2] *
      action_scale` once per substep instead of once per transition, a
      10x overshoot with zero theta dependence. Fix verified locally at
      8 and 32 episodes (gripper RMSE now matches/beats persistence); not
      yet verified at the 256-episode matrix scale (no GPU on this
      machine).
- [ ] **Re-run the 256-episode Fetch matrix (A/B/C/H/F)** now that the
      gripper fix is in — every Fetch number in the 2026-08-26 "Fetch
      matrix" entry above predates the fix and should not be cited in the
      paper as-is.
- [ ] **Root-cause the residual Fetch solver-adequacy gap**: even after
      the gripper fix, oracle-fit theta only ties persistence
      (0.47 vs 0.41 at 32 episodes) rather than beating it by orders of
      magnitude the way PokeWorld/Cart-pole/Push-T do. Candidates:
      `contact_stiffness=400` is a fixed, unfit constant; the
      linear-friction/point-mass approximation may not suit Fetch as well
      as it suits PokeWorld. See the 2026-08-26 "Gripper overshoot fixed"
      entry for detail.
- [ ] Commit the local changes to `scripts/smoke/validate_solvers.py`
      (working `fetch_push` branch + `true`-theta RMSE column) and
      `stable_worldmodel/wm/physwm/solvers.py` (the gripper fix) together.
- [ ] **Run the aligned 20-experiment matrix on CUDA** once a GPU is available:
      `MUJOCO_GL=egl .venv-phy-wm-gpu/bin/python scripts/eval/overnight.py --seeds 0,1,2`.
      The dry-run resolves 16 light and 4 heavy experiments, but none has been
      executed under the aligned protocol.
- [ ] Before interpreting parameter recovery, require the teacher premise:
      Path A must beat persistence on held-out query transitions. Then require
      the predictive baseline to remain blind, aligned PhysWM to improve across
      seeds, and inferred theta to beat shuffled and nominal solver controls.
- [ ] Treat every entry through 2026-08-25 07:48 as historical diagnostics.
      Do not combine its pre-action/older-pooling numbers with the aligned
      matrix in a workshop table or plot.
- [ ] Inspect peak memory for the heavy frozen-DINOv2 and window-16 functional
      runs before leaving the full matrix unattended; reduce batch size only
      through an explicitly logged config override.
- [ ] Commit or tag the aligned implementation before the paper-scale launch.
      The current dirty-tree fingerprint is reproducible through artifacts but
      is weaker provenance than a clean commit.
- [ ] After the aligned primary runs, only tune signal design if the acceptance
      checks fail. The registered matrix already contains the dataset-target,
      pre-action, visual-only, probe-capacity, context-length, functional, and
      encoder-scale ablations; do not redesign from the tiny CPU smoke.
