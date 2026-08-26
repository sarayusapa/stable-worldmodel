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

## 4. Cartpole — 512 episodes, run by me, in progress

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

**Early smoke-test signal (8 episodes, 2 epochs, toy scale):** Path A
scored *worse* than persistence (2.34 vs. 0.42) — at this toy scale
that's expected (2 epochs is essentially no training), not evidence
against the model; the real 40-epoch run is what will actually test
this.

**Result dir:** `/workspace/physwm-artifacts/runs/pusht-cartpole-matrix/cartpole_xbench_*.json`
(not yet pulled to this repo).

---

## Status as of writing

All 9 PushT/Cartpole jobs running (GPU ~100% utilized). Once complete:
JSON results + logs will be pulled down, documented the same way as the
Fetch matrix (`READING_THE_LOGS.md`-style guide), logged into
`progress.md`, and pushed to `phy-wm`.
