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
