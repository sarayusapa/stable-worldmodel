# Experiment ledger and citation status

This ledger prevents exploratory, confounded, and incomplete runs from being silently promoted into paper claims.

## Safe for current quantitative discussion

### PokeWorld, 2048 episodes, three seeds

- Decodability files: `/workspace/physwm-artifacts/runs/pilot-2048ep/A_pilot_2048ep_seed{0,1,2}.json`
- Functional files: `/workspace/physwm-artifacts/runs/functional-2048/func2048_seed{0,1,2}.json`
- Encoder: `tiny_cnn`.
- Role: establish the accurate-predictor premise, semantic parameter readout, parameter substitution, and multi-horizon equation rollout.
- Qualification: true parameters make the matched PokeWorld solver essentially exact, but inferred semantic recovery is mixed. Stiffness is strongly readable; mass is weak/moderate; drag is not reliably recovered.

### PushT randomized dynamics, 512 episodes, three seeds

- Files: `/workspace/physwm-artifacts/runs/pusht-cartpole-matrix/pusht_rand_xbench_seed{0,1,2}.json`
- Encoder: `tiny_cnn`.
- Role: label-free test under solver mismatch. The episode's inferred effective parameters are compared with shuffled and nominal parameters.
- Qualification: inferred parameters beat shuffled in all three seeds and nominal in two of three. The learned predictor fails to beat persistence in seed 0. Solver sensitivity is concentrated in agent controller gains, so this supports effective controller-dynamics grounding rather than comprehensive contact-parameter identification.

## Diagnostic only

### PokeWorld data-floor and ablation sweeps

- Directories: `datafloor/`, `ablations-2048/`, and `20260825-133600/`.
- Useful for diagnosing data scale, probe capacity, pre-action routing, direct-dataset supervision, and post-hoc probes.
- These results require a dedicated aggregation and interpretation pass before inclusion. They should not be mixed into the main table ad hoc.

### PushT decodability

- Files: `pusht_rand_decod_seed{0,1,2}.json`.
- Most physical dimensions have no ground-truth variation or are not semantically certified. Report functional substitution before label decodability.
- PhysWM does not consistently improve linear readout over the predictive baseline.

## Excluded from positive claims

### FetchPush pre-fix matrix

- Directory: `/workspace/physwm-artifacts/runs/fetch-matrix-256ep/`.
- Exclusion reason: `FetchPushSolver.step()` applied a per-transition gripper displacement once per numerical substep, causing a 10× overshoot. Commit `03fa29b` corrected this.
- Consequence: pre-fix solver rollouts failed the solver-adequacy prerequisite and confounded true, shuffled, nominal, and inferred parameter comparisons.

### Corrected FetchPush functional seed 0

- File: `fetch-matrix-256ep-fixed/F_fetch_functional_seed0.json`.
- Current values: inferred `1.535`, true `6.984`, shuffled `7.357`, nominal `22.364` scaled one-step RMSE; every source has task success `0.03125`.
- Interpretation: the inferred variables appear to compensate for residual solver mismatch rather than recover true mass/friction. This is a useful failure analysis, not clean evidence of semantic physical recovery.

## Pending

### Corrected FetchPush primary three-seed runs

- `A_fetch_claim_seed{0,1,2}` were still running when this ledger was created.
- B/C/H corrected ablations had recently completed but were not used in the main paper assets because the primary three-seed result was incomplete.

### CartPole

- Runs encountered native MuJoCo/EGL failures and a render-context leak. The leak was fixed in commit `bf4e356`, but the current jobs had been reduced to eight episodes per seed.
- Eight-episode results are not suitable as headline paper evidence.

## Minimum checks before submission

1. Complete the corrected Fetch primary matrix and rerun solver adequacy.
2. Run a stable CartPole protocol with a defensible sample size.
3. Add a frozen DINOv2 three-seed result if the paper claims pretrained visual representations rather than only a tiny CNN diagnostic.
4. Report confidence intervals or paired seed-level statistics; three seeds permit descriptive uncertainty but weak hypothesis testing.
5. Preserve the context/query anti-leakage tests and document the exact checkpoint/config fingerprint.

