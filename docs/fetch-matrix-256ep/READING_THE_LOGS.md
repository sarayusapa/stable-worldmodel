# How to read the Fetch matrix logs and JSON

Two scripts produced these 8 jobs: `decodability.py` (A, B, C, H) and
`functional_use.py` (F). Each job wrote a `.log` (human-readable stdout)
and a `.json` (same numbers, structured) — this doc explains both formats
and how to actually interpret the numbers.

---

## Part 1 — `decodability.py` jobs

Applies to: `A_fetch_claim_seed0/1/2`, `B_fetch_preaction_seed0`,
`C_fetch_dataset_target_seed0`, `C_fetch_posthoc_seed0`, `H_fetch_vq_seed0`.

### `.log` structure, section by section

```
AdroitHandRelocateDense-v1, ... (deprecation warning)
```
Harmless. A `gymnasium_robotics` import-time warning about an unrelated
env family. Ignore it every time.

```
PokeWorld theta recovery -- val R^2 (256 episodes x 48, 60 epochs, seed 1, ...)

  windows after observability filter: 10496 train / 2624 val
```
The header. **The "PokeWorld" label is a cosmetic bug** — it's hardcoded
in the print statement regardless of `--benchmark`. Everything else in
the header is accurate: episode count, episode length, epoch count, seed,
encoder, and — importantly — `windows`, the actual dataset size after
filtering. If this number is small (low hundreds), treat every R² below
with suspicion; there may not be enough data to support any conclusion
(see "The data-floor problem" below).

```
param      predictive/decodable  predictive/own_probe      physwm/decodable      physwm/own_probe
mass                    -0.6688               -2.0463               -0.3423               -2.7533
friction                -0.7250               -0.0063               -0.4622               -2.7647
```
**The most important table.** Read the column names as `<condition>/<readout>`:

- **condition** — `predictive` (α=0, the physics loss is never applied —
  this is the "does the model encode physics at all if you never ask it
  to" baseline) vs. `physwm` (α=1, the real method).
- **readout** — `decodable` is a *supervised* ridge regression fit
  directly on the frozen latent using the true labels — the honest
  ceiling, "is this parameter linearly present in the latent at all,
  given full supervision." `own_probe` is the model's *own* probe output,
  trained with **no** access to ground-truth θ — this is the actual
  claim under test.

**R² interpretation**: 1.0 = perfect, 0.0 = exactly as good as predicting
the population mean every time, negative = *worse* than guessing the
mean. Negative numbers are common and not automatically a bug — they
mean the model is worse than a constant guess for that quantity, not
that something crashed.

**What to look for**: does `physwm/own_probe` beat `predictive/own_probe`?
That's the actual hypothesis (self-distillation induces label-free
identifiability). In this batch it doesn't — see "Known result" below.

```
Dynamics-coordinate recovery -- val R^2
coordinate  predictive/decodable  predictive/own_probe      physwm/decodable      physwm/own_probe
mass                    -0.6688               -2.0463               -0.3423               -2.7533
friction                -0.7250               -0.0063               -0.4622               -2.7647
```
For PokeWorld this table reports the *identifiable coordinates*
(`stiffness/mass`, `drag/mass` — the ratios that are actually observable
from vision, as opposed to the individually-unidentifiable raw values).
**For `fetch_push` this table is currently identical to the one above.**
`dynamics_coordinates()` in `decodability.py` only recognizes PokeWorld's
parameter names and passes everything else through unchanged — a
`fetch_push` branch (reporting `friction/mass`, the quantity that should
actually be identifiable here) was never added. Treat the raw `mass`/
`friction` numbers above with that caveat in mind — see "Known result."

```
Prediction/fidelity RMSE on the SAME held-out checkpoints
metric                            predictive          physwm
path_a_vs_dataset                     0.3984          0.4002
path_a_query_vs_dataset               0.4282          0.4274
path_b_vs_teacher                     8.6647          0.7687
path_b_vs_dataset                     8.6545          0.8111
persistence_vs_dataset                0.6293          0.6293
persistence_query_vs_dataset          0.6562          0.6562
```
Lower is better (these are RMSE, not R²). Read row by row:
- `path_a_vs_dataset` / `path_a_query_vs_dataset` — the world model's own
  next-state prediction error vs. the real trajectory. **Compare this to
  `persistence_vs_dataset`/`persistence_query_vs_dataset`** (the trivial
  "nothing moves" baseline) — Path A should be *lower*. In every job so
  far it is, comfortably (e.g. 0.40 vs 0.63) — this is the one part of
  the hypothesis that's holding cleanly.
- `path_b_vs_teacher` — the frozen-solver prediction (driven by the
  probe's θ) vs. Path A's own prediction (the actual training target for
  `physwm`, since it's self-distilled, not label-supervised). `physwm`
  should be far lower than `predictive` here, since `predictive` never
  trains its probe at all (α=0 means zero gradient reaches it, so it
  stays at initialization). That gap is large and consistent (~8.7 vs
  ~0.8) — the solver-fit mechanism itself is genuinely learning
  something coherent, even where the θ→ground-truth mapping (the table
  above) isn't there yet.
- `path_b_vs_dataset` — same rollout, compared against the real label
  instead of the teacher. Useful as a sanity check that Path B isn't
  diverging wildly from reality even though it was never trained against
  it directly.

```
reading it: if `predictive/decodable` is near zero the predictive
objective never encoded that parameter -- which is the claim. If it is
high but `predictive/own_probe` is low, the physics was already there
and only the read-out was missing.

wrote /workspace/physwm-artifacts/runs/fetch-matrix-256ep/....json
```
Fixed boilerplate every run, plus a confirmation the JSON was written.

### `.json` structure

```json
{
  "meta": { "encoder", "probe_hidden", "probe_source", "detach_probe_input",
            "window", "tactile", "epochs", "episodes", "length", "alpha",
            "seed", "batch_size", "conditions", "amp",
            "windows": {"train": N, "val": N} },
  "results": { "<condition>/decodable": {param: r2, ...}, "<condition>/own_probe": {...} },
  "dynamics_results": { same shape as results },
  "prediction": { "<condition>": {6 RMSE keys} }
}
```
This is exactly the tables above, unrolled into JSON — `results` is the
theta-recovery table, `dynamics_results` the coordinate table (currently
a pass-through for `fetch_push`, see above), `prediction` the RMSE table.
`meta` records the exact run configuration — this is what you'd quote if
citing a number (matches this project's `progress.md` discipline of
always attaching the exact config a number came from).

---

## Part 2 — `functional_use.py` job (`F_fetch_functional_seed0`)

Different script, different question: not "is θ decodable" but "is θ
*used*" — does substituting the wrong θ into the frozen solver actually
change predictions, and does recovering the right θ actually help.

### `.log` structure

```
Functional use of theta -- PokeWorld, 256 episodes x 48, 60 epochs, seed 0
```
Same cosmetic "PokeWorld" label bug as above — ignore it, the data is
real Fetch.

```
1. SENSITIVITY: prediction shift from scaling theta by 1.1x
   (near zero => the solver ignores that component; recovering it is meaningless)
   mass         0.05910
   friction     0.00034
```
Perturbs each θ component by +10% and measures how much the frozen
solver's prediction moves (scaled RMSE). Near-zero means the solver
doesn't actually respond to that parameter — recovering it would be a
decorative exercise even if you could. Higher = the parameter is
mechanically load-bearing in the physics.

```
2. SUBSTITUTION: one-step Path B error vs true s_next (scaled RMSE, lower better)
   true         4.85762
   probe        4.79555
   nominal     14.64257
   shuffled     4.84695

   shuffled-minus-true gap :   -0.01068   (how much episode-specific theta is worth at all)
   probe closes            :    -481.3% of it
```
Four sources of θ fed into the *same* frozen solver, one-step prediction
error against the real next state:
- `true` — the episode's actual ground-truth θ (best case).
- `probe` — what the model's own probe recovered.
- `nominal` — the solver's fixed default θ (ignores the episode
  entirely).
- `shuffled` — another episode's true θ, mismatched on purpose (tests
  whether episode-specific θ matters *at all*).

**The gap that matters**: `shuffled - true`. If that's near zero (as
here: -0.01), episode-specific θ barely matters for one-step prediction
at this data scale — there's no room for `probe` to "close" a
meaningful gap, which is why `probe closes -481.3%` reads as a wild,
uninterpretable percentage rather than a clean 0-100% figure. **A
percentage like that is a sign the denominator (the gap) is too close to
zero to be a meaningful ruler, not that the probe did something 481%
wrong.**

```
3. MULTI-HORIZON rollout error (scaled RMSE, horizon 1..7)
   source           1        2        3        4        5        6        7
   true        0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
   probe       ...
   nominal     ...
   shuffled    ...
```
Same four sources, but rolled forward multiple steps instead of one.
What to look for: does the `probe` row track `true` (both near 0) and
pull away from `shuffled`/`nominal` as the horizon grows? That widening
gap is the actual evidence for "θ is functionally used," if present.

```
4. TASK SUCCESS: fraction of episodes reaching the goal in a 7-step solver
   rollout (threshold 0.05)
   true            X.X%
   probe           X.X%
   nominal         X.X%
   shuffled        X.X%
```
Same four θ sources, but scored by whether the *frozen-solver rollout*
(not the real robot) gets the object within `threshold` of the goal.
**Important distinction**: this is "would this θ, run through our solver
approximation, have reached the goal" — not "did the real MuJoCo
simulation succeed." A low number across all four sources (including
`true`) usually means the rollout horizon/window is too short or the
scripted data-collection policy didn't push the object very far in that
window — not that θ recovery failed.

### `.json` structure

```json
{
  "meta": {...},
  "sensitivity": {"mass": v, "friction": v},
  "theta_variation": {"mass": {"std", "mean", "varies"}, "friction": {...}},
  "substitution": {"probe", "true", "shuffled", "nominal"},
  "multi_horizon": {"probe": [h1..hN], "true": [...], "shuffled": [...], "nominal": [...]},
  "task_success": {"probe", "true", "shuffled", "nominal"}
}
```
`theta_variation` is worth checking before trusting `substitution`/
`task_success`: if `varies: false` for a parameter, it was constant
across every collected episode, and shuffling it can't possibly change
anything (the whole substitution premise is vacuous for that parameter).

---

## Part 3 — `matrix.log` (the orchestrator, not a training run)

```
[00:20:30] START A_fetch_claim_seed0
[01:10:39] DONE A_fetch_claim_seed0 rc=0 dur=3009s
```
Just timestamps and outcomes for each job: `rc=0` is success, anything
else means the process exited with an error (check that job's own `.log`
for the traceback). `dur` is wall-clock seconds. This file has no
scientific content — it's purely "did it run, how long did it take."

---

## Known result from this batch (context, not a bug)

Across all completed jobs so far, `physwm/own_probe` scores *worse* than
`predictive/own_probe` on both `mass` and `friction`, in every seed. Two
things are true at once, confirmed by reading the actual code (not
guessed):

1. **Not a training bug.** `dynamics_coordinates()` genuinely only
   handles PokeWorld's parameter names; for `fetch_push` it passes raw
   theta through unchanged (verified in `decodability.py`).
2. **A real identifiability gap in the Fetch benchmark's design.**
   `FetchPushSolver`'s dynamics (`m·dv/dt = F_contact − friction·v`) have
   the same mass-vs-friction degeneracy PokeWorld solves with its
   `touch` (contact-force) channel — but Fetch's state has no analogous
   force signal, only position/velocity. From trajectory alone, only the
   *ratio* `friction/mass` should be identifiable, not the two raw
   values separately. A probe trained with no ground truth is free to
   converge on any `(mass, friction)` pair that reproduces the right
   ratio, which is consistent with it landing *further* from the true
   raw values than an untrained constant guess.

Also worth knowing: even the **supervised ceiling** (`*/decodable`) is
negative for every parameter — meaning mass/friction may not be
decodable from this latent *at all* yet, which points at 256 episodes
being below Fetch's own data floor (PokeWorld needed 2048 episodes
before its equivalent ceiling turned positive).
