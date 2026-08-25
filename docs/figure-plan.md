# PhysWM figure plan

Which figures go in the workshop paper, what claim each one carries, which
metric key it plots, and whether the data exists yet.

Written 2026-08-25 against `1320811` on `phy-wm`. Companion to
`docs/physwm-formulation.typ` (the formulation and experiment contract) and
`progress.md` (the run log every number must trace back to).

**This is a contract, not a report.** The 20-experiment matrix in
`scripts/eval/overnight.py` resolves but has never executed. Naming in advance
which plot decides each claim is the only thing that keeps a null result
publishable instead of turning it into a search for a kinder metric.

---

## 1. What we are actually measuring

The abstract makes three chained claims. Every figure plots exactly one link in
that chain. Nothing plots the architecture — the architecture is not the
contribution.

| Claim | Reads | Script | Metric key |
| --- | --- | --- | --- |
| **Premise.** Path A predicts well | `RMSE_A < persistence` | `prediction_quality.py` | `path_a`, `persistence` |
| **Premise.** ...and is blind to physics | `R² ≈ 0` at `alpha = 0` | `decodability.py` | `predictive/decodable` |
| **Claim.** Self-distillation induces theta | `R²` rises at `alpha > 0`, no labels | `decodability.py` | `physwm/own_probe` |
| **Claim.** Theta is *used*, not just readable | probe < shuffled, gap widens with H | `functional_use.py` | `substitution`, `multi_horizon` |
| **Ceiling.** The information is there at all | free-theta oracle fit | `validate_solvers.py` | fitted RMSE, certificate R² |

Cartpole and PushT are registered with `has_theta_true: False`
(`stable_worldmodel/wm/physwm/build.py`). They can appear in prediction-quality
and rollout figures and never in a theta-recovery figure. Say so in the caption
rather than letting a reviewer find it.

---

## 2. Main figures

Ordered as they would appear. F1 is the teaser, F2–F5 are the result, F6–F7
defend it.

Status legend: **[have]** data in hand · **[run]** needs the aligned run ·
**[code]** needs new code.

### F1 — Filling the hierarchy · [run] [code]

The teaser. The abstract's phrase *"tall but thin at the bottom"* rendered as a
measurement, and the one figure only this project can produce.

A dot grid, three panels side by side.

- **Rows** (hierarchy levels, top to bottom):
  - `PREDICT` — Path A's held-out accuracy on transitions where that parameter
    is observable (contact steps for `k`, free glide for `c`, post-impact for
    `m`), as `1 - RMSE_A / RMSE_persistence`.
  - `USE` — prediction degradation when *that one* theta component is replaced
    by another episode's value.
  - `IDENTIFY` — held-out R² of the model's own probe, episode-averaged.
- **Columns** — physical parameter (`m`, `k`, `c`, radii).
- **Panels** — `predictive` (alpha = 0) · `physwm` · `theta_oracle`, same seed,
  same data, same architecture.
- **Encoding** — dot fill = score; an edge is drawn between two levels only
  where the upper one measurably depends on the lower one.

The edges matter more than the dots. A filled `IDENTIFY` dot with no edge above
it means the parameter is readable but inert — precisely the failure the last
diagnostic run showed, and precisely what a system-ID paper would not notice.

- **Source:** `decodability.py` (rows 1 and 3), a per-component variant of
  `functional_use.substitution` (row 2).
- **Gap:** `substitution()` shuffles the whole theta vector; needs a
  one-component-at-a-time mode, or the `USE` row has one number for all
  parameters and the dot map loses its columns.

### F2 — The self-target residual · [run] [code]

Three-row image strip per timestep on a held-out episode:

1. Path A's prediction, rendered.
2. Path B's prediction (probe -> frozen solver), rendered.
3. Their signed difference in a diverging colormap.

Because Path B's target is `sg(Path A)` and not the dataset, **row 3 *is*
`L_B`** — this shows the loss, not a proxy for it. Inset: `rmse_b_vs_teacher`
per epoch from `history.json`, the residual closing over training. That pairing
is what distinguishes this from a system-ID paper.

- **Source:** `PhysWM.forward()` already returns `state_a_phys` and
  `state_b_phys` in physical units; `PokeWorldSim.render(states, actions, size)`
  is a pure function of state.
- **Gap:** a ~40-line `scripts/eval/render_predictions.py`. Nothing in the model
  changes.
- **Caveat:** PokeWorld renders two Gaussian blobs on a blank field, so the
  residual is a dipole, not a rich texture. Either accept it and caption it
  plainly, or run this figure on PushT where the T-block gives the difference
  map real shape — at the cost of having no ground-truth theta in the same
  figure. There is no pixel decoder in the model; the caption must not imply
  one.

### F3 — Error maps by contact regime · [run] [code]

Grid of rendered next states with error-map overlays.

- **Rows** (held-out transitions stratified by which physics they exercise):
  free glide (drag-dominated) · impact (stiffness) · grazing contact · no
  contact. Selected from `touch` and `||v||`, never from a query outcome.
- **Columns:** ground truth · persistence · Path A · Path B with probe theta ·
  with shuffled theta · with true theta.

The shuffled column is the control that turns a picture into a claim.

- **Source:** same renderer helper as F2, plus a regime selector over the val
  split.
- **Caveat:** only **5.1%** of PokeWorld transitions register any contact
  (measured, historical diagnostic). The impact row is sampled from a thin
  slice — put the count in the caption.

### F4 — Recovery against the certificate · [run] [code]

One scatter panel per theta component: inferred vs true, one point per held-out
episode, identity line dashed. PhysWM probe against the predictive baseline,
pooled over three seeds, R² annotated in-panel. Underneath, a bar strip where
the bar is the probe's R² and a tick marks the free-theta recoverability
certificate.

Identity-scatter replaces the bars-vs-ground-truth-line of the reference figure
because theta varies per episode here.

- **Visual-only version** swaps in `k/m`, `c/m`, `r1+r2` via
  `dynamics_coordinates()`.
- **Source:** `decodability.py`, plus the certificate from
  `validate_solvers.py`.
- **Gap:** `decodability.py` writes `results` / `dynamics_results` /
  `prediction`, all pre-aggregated — the scatter cannot be reconstructed from
  the artifacts. Add the per-episode `(theta_hat, theta_true)` pairs to the
  payload. `validate_solvers.py` needs an `--out`.
- **Reconcile before submission:** last measured certificate on this simulator
  is `0.83 / 0.76 / 0.75` for mass / stiffness / drag; the abstract draft cites
  `0.89`, which came from an earlier certificate.

### F5 — Decodable versus used · [have] [run]

The load-bearing figure. Decodability alone cannot separate a parameter the
model *carries* from one it *uses*; this one intervenes and measures whether the
prediction moves.

- **Left:** `substitution` — one-step Path B error under true / probe / nominal
  / shuffled theta, with the gap-closed fraction annotated.
- **Right:** `multi_horizon` — the same four sources rolled through the frozen
  solver, H = 1..16, shaded bands across seeds. The gap must *widen* with
  horizon; a flat gap means the prediction is dominated by the initial state,
  not by theta.

- **Source:** the JSON from `F_functional_seed{0,1,2}` in the overnight matrix.
  Plotting only — both panels are already computed and dumped.
- **What this currently shows:** on the last measured run the probe closed
  **-4.7%** of the shuffled-to-true gap, and probe / shuffled / nominal sat on
  top of each other at every horizon. A clean null. The figure is still the
  right figure; it just currently argues the opposite of the abstract.

### F6 — The ablation ladder · [run]

One lollipop chart replacing an ablation table. Held-out mean theta R² on the
x-axis, whiskers = seed range (n = 3), a dashed vertical line at the
certificate. Rungs, in order:

`predictive (alpha=0)` -> `detached probe` -> `pre-action latent` ->
`dataset target` -> **`PhysWM`** -> `theta-oracle`

Each rung removes exactly one design decision, and the ordering is the argument.
If `pre-action latent` scores the same as PhysWM, then "the *same*
action-conditioned latent" is decoration and should come out of the abstract.
The detached-probe rung is the one reviewers will look for: it separates
*inducing* a representation from *fitting a read-out* on a representation the
loss never touched.

- **Source:** `overnight.py` groups A / B / C, three seeds each. Plotting only.
- **Extras:** fold group G (MLP probe, context-16) in as small open markers
  rather than giving them a figure.

### F7 — What touch makes identifiable · [run]

Grouped bars: raw `(m, k, c)` on the left, `dynamics_coordinates()`
(`k/m`, `c/m`, `r1+r2`) on the right; series are visual+tactile (matrix group A)
vs `--no-tactile` (group D), three seeds each.

A defensive figure that pays for itself. Without it a reviewer sees near-zero
raw-theta R² in the visual-only condition and concludes the method fails; with
it they see that motion alone identifies only `k/m` and `c/m`, so near-zero raw
R² is the *correct* answer there.

- **Source:** `decodability.py` — `dynamics_coordinates()` is already
  implemented and `dynamics_results` is already dumped. Plotting only.

---

## 3. Supplement

| Fig | What | Why it exists | Status |
| --- | --- | --- | --- |
| S1 | Solver adequacy per benchmark: persistence / nominal-theta / fitted-theta RMSE | Proves the frozen solver can express the data before any learning is discussed. PokeWorld measured at `0.4051 / 0.0660 / 0.0002`. | have |
| S2 | Sensitivity spectrum: prediction shift from scaling each theta by 1.1x | Explains a null before a reader guesses at one. Last measured `m 0.0054`, `k 0.0450`, `c 0.0051` against `r 0.098` — a 20x spread. | have |
| S3 | Contact-event histogram and touch-channel distribution | 5.1% contact rate with an 8-frame window means many windows carry no stiffness evidence at all. The single most likely cause of a null. | small script |
| S4 | Encoder scale: tiny CNN vs frozen DINOv2-small, paired per seed | Rules out "the result is an artifact of a toy encoder", the first thing a reviewer says about a 64px CNN. | group E |
| S5 | Cross-benchmark prediction quality: PokeWorld / Cartpole / PushT vs persistence | The only place the other two benchmarks can honestly appear. They have no ground-truth theta. | needs runs |

---

## 4. What has to be written first

Five of the seven figures are pure plotting over JSON that `scripts/eval/`
already emits. These are the gaps between the aligned run and a figure set.

- [ ] **`scripts/eval/render_predictions.py`** — render `state_a_phys` and
      `state_b_phys` through `PokeWorldSim.render` and write the three-row
      strips. Unblocks F2 and F3.
- [ ] **`decodability.py` payload** — dump the per-episode
      `(theta_hat, theta_true)` pairs alongside the aggregate R². Unblocks F4.
- [ ] **`functional_use.substitution`** — shuffle one theta component at a time
      instead of the whole vector. Unblocks F1's middle row.
- [ ] **`validate_solvers.py --out`** — write the certificate to JSON instead of
      printing it. Unblocks the ceiling markers in F4 and F6.
- [ ] **`scripts/figures/`** — one module per figure, reading only the JSON
      artifacts, never re-training. A figure that needs a GPU to redraw will not
      get redrawn during rebuttal.

---

## 5. References and inspiration

Each reference is mapped to the figure it informs, so the debt is visible and
the borrowed structure is deliberate rather than accidental.

| Source | What we took | Figure |
| --- | --- | --- |
| J. Chen, *Filling the hierarchy* / *Constructive self-supervised learning* — <https://jchencxh.github.io/blog/filling-the-hierarchy-part-1/> | The dot-grid abstraction hierarchy, with filled dots and connecting edges showing which levels are populated. We quantify the fill and make the edges a measured dependency. | F1 |
| arXiv:2608.09926 (LDR) | Rows are *physical scenarios* (uniform motion, parabola, collision, bouncing, looming), columns are methods and ablations, each cell a prediction with an error-map inset. The row stratification is what makes it an argument. | F3 |
| gradSim, arXiv:2104.02646 | Error vs optimization iterations with shaded confidence bands and a flat "randomly actuated model" control line. The flat control is what makes the real curve legible; our shuffled-theta ceiling plays that role. | F5 (right) |
| *Physically Interpretable World Models via Weakly Supervised Representation Learning*, arXiv:2412.12870 | Learned physical parameters plotted against ground truth with an explicit GT reference mark, per parameter, across noise levels. | F4 |
| Our own planning note | "The self-target residual decomposition. Three-row strip per timestep: baseline latent prediction, solver-through-probe prediction, signed difference in a diverging colormap. Because the target is the model's own prediction rather than ground truth, row (iii) is literally the training signal, and showing it shrink over training epochs is a training-dynamics figure nobody else can make." | F2 |
| `latent-world-model-identifiability` — <https://github.com/tantansir/latent-world-model-identifiability> | The PokeWorld setting itself: parameter ranges, the observability spectrum across drag / mass / stiffness, and the recoverability-certificate framing. | S1, F4 ceiling |

Also in the planning doc but not retrievable at time of writing:
arXiv:2608.05720v1 and arXiv:2606.05328v1.

---

## 6. The state these figures are in

`progress.md` records the dry-run and a CUDA-unavailable failure and nothing
more. **Every entry at or before 2026-08-25 07:48 used the pre-action route and
the older pooling**, and the log explicitly forbids mixing those numbers into a
paper plot.

**If the aligned run reproduces the null**, F1 and F5 do not get thrown away.
F1 becomes the paper's actual finding: `PREDICT` filled, `IDENTIFY` empty, no
edges — a measured demonstration that self-distillation through a frozen solver
does *not* fill the bottom of the hierarchy at this scale, with S1's certificate
proving the information was there to be found and S2 and S3 showing why a single
0.02 s transition at a 5% contact rate cannot carry it. That is a real workshop
paper, and it needs exactly the same seven figures.
