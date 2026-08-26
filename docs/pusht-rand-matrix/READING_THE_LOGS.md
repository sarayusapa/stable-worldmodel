# Reading the PushT-randomized (`pusht_rand`) matrix logs

Six jobs, 512 episodes, length 48, window 8, `tiny_cnn` encoder, 3 seeds
each. Two different scripts asking two different questions — see
`docs/EXPERIMENTS_RUN.md` section 3 for the full config/rationale.

## `pusht_rand_xbench_seed{0,1,2}.json` — label-free (`cross_benchmark.py`)

- `substitution.probe/shuffled/nominal`: scaled RMSE of the frozen
  `PushTSolver` fed the model's inferred theta vs. another episode's theta
  vs. the solver's fixed default, against the real observed transition.
  Lower is better.
- `beats_nominal`/`beats_shuffled`: whether `probe` scores strictly lower
  than each baseline.
- `sensitivity`: how much the solver's output moves when each theta
  component is bumped 10% — a measure of which parameters the dynamics
  actually depend on, independent of whether the model recovers them.
- `prediction.path_a` vs `prediction.persistence`: does the learned
  next-state prediction beat "nothing moved"?

**Result across all 3 seeds:** `probe` beats `shuffled` every time
(gap 0.057-0.060) — theta is carrying real per-episode information, not
noise. `probe` beats `nominal` in 2/3 seeds (seed 2 is the exception,
0.2155 vs 0.2095, a small miss). `sensitivity` is dominated by
`agent_kp`/`agent_kv` (~0.035-0.040) with everything else an order of
magnitude smaller (~0.001-0.007) — matches physical intuition, since the
PD controller gains directly set how hard the agent pushes, while
`com_offset`/`mobility`/`contact_stiffness` are second-order effects on
the block's response. `path_a` beats `persistence` in 2/3 seeds (seed 0 is
the exception: 0.620 vs 0.584, worse than doing nothing).

## `pusht_rand_decod_seed{0,1,2}.json` — labeled R² (`decodability.py`)

Only `agent_kp`/`agent_kv` have a real recorded ground-truth value
(`PushTSolver` reproduces the environment's own PD law exactly); every
other theta component here is `NaN` by design — `friction`/`mass` are
randomized too (to make the `xbench` shuffled-control meaningful) but
have no 1:1 solver parameter to score against, and `dynamics_coordinates()`
in `decodability.py` has no `pusht_rand` branch (same reporting gap noted
for `fetch_push` in the earlier matrix), so raw values are used, not a
transformed identifiable coordinate.

**Result:** `own_probe` R² for `agent_kp`/`agent_kv` is negative in most
seed/condition combinations (worst: seed1 predictive/own_probe
`agent_kp -0.185`), i.e. the model's self-supervised readout does not
beat a mean-baseline predictor for the PD gains at this scale/config —
a weaker signal than the label-free check above. `decodable` (supervised
ridge ceiling) is mixed, some positive (seed1 `physwm/decodable agent_kp
0.135`) some negative. `physwm/path_b_vs_dataset` beats
`predictive/path_b_vs_dataset` in 2/3 seeds, consistent with the
label-free result that physics-grounded theta is doing *something* useful
even where the labeled R² isn't clean.

**Read together:** the label-free check (which only asks "does theta
distinguish episodes and beat a physically-fixed default", not "does it
match the exact PD-gain values") shows a real, consistent signal across
seeds. The labeled R² check is a stricter bar and mostly isn't cleared at
this episode count — same qualitative pattern as PokeWorld and Fetch both
needed a real data-floor before their own labeled ceilings turned
positive; not yet checked whether `pusht_rand` needs the same.
