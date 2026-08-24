# PhysWM — a physics-grounded world model

Two next-state predictions from the **same** DINO-WM latent. Path A is
supervised on the dataset's ground-truth `s_next`; Path B is supervised on
Path A's own (detached) prediction instead — the experiment is whether a
physics parameterization can explain what the world model *itself*
believes happens next, not whether the solver can re-derive the raw
benchmark label:

```
                     ┌── predictor ──► ẑ ── decoder ──────────────► s_A   (Path A, learned, ← s_next)
pixels ── encoder ──►│
                (z)  └── probe ──► θ ──► frozen physics solver ───► s_B   (Path B, physical, ← s_A.detach())
                                          (s_t, a_t, θ)
```

```
L = L_A + α·L_B + β·L_consistency

L_A           = ‖ s_A            − s_next        ‖²
L_B           = ‖ solver(θ)      − s_A.detach()  ‖²
L_consistency = ‖ s_A − solver(θ) ‖²          (β = 0 by default)
```

## The four design rules, and where they are enforced

| Rule | Enforced by |
|---|---|
| Two predictions branch off one latent; neither is fitted to the other | `PhysWM.forward` — both paths read `z` |
| A regresses ground truth; **B regresses A's detached output, and A is never trained on B** | `loss.physwm_loss`; tested by `test_loss_b_does_not_depend_on_path_a` (L_B carries no gradient into the Path A decoder) |
| θ is a low-capacity forward-pass probe output, per-episode by default; **never** a free per-step variable | `ThetaProbe`; `assert_no_free_theta` runs in `PhysWM.__init__` |
| Solver is frozen, differentiable, **zero** learnable parameters | `PhysicsSolver.assert_frozen`; bounds live in buffers, θ is always a forward argument |

The only place in the project where θ is a free optimizable variable is
`scripts/smoke/validate_solvers.py`, which is a diagnostic that measures
solver adequacy — never the training path.

## Units and normalization

The solver is a physical model, so it must see physical units; the
networks want normalized inputs; and `L_A`/`L_B` must be commensurate.
So the batch carries physical units, the networks consume normalized
values, the solver consumes physical values, and **all losses are
computed in normalized state space**. The normalizer lives in the model
as buffers, so a checkpoint is self-contained.

## Swappable modules

Everything is chosen by name from config and wired in `build.py`.

- **Encoder** → `(B, N, D)` patch tokens: `tiny_cnn` (no downloads, used
  by the smoke tests), `dinov2` (the real frozen DINO-WM backbone),
  `state` (state-input ablation).
- **Predictor**: `DinoWMPredictor`, a block-causal spatio-temporal
  transformer over patch tokens — frame `t` attends to all patches of
  frames `≤ t`, and its output predicts frame `t+1`. With `N = 1` it
  degenerates to a pooled-latent predictor, so one module covers both.
- **Probe**: `ThetaProbe`. `hidden_dim: 0` (default) is a single linear
  map — the honest "is θ linearly decodable from the latent" probe.
  `mode: episode | step`.
- **Solver**: `pokeworld | cartpole | pusht`.
- **Data**: synthetic PokeWorld, live `swm/*` env rollouts, or the
  repo's Lance/HDF5 datasets.

## Benchmarks

| | state | θ | ground-truth θ? |
|---|---|---|---|
| `pokeworld` | `[x, y, vx, vy]` | mass, contact stiffness, drag, radii | **yes** → R² certificate |
| `cartpole` (dm_control) | `[x, angle, ẋ, anglė]` | cart mass, pole mass, half-length, gravity, force gain, damping | no |
| `pusht` | `[agent_xy, block_xy, block_angle, agent_vxy]` | PD gains, radii, contact stiffness, mobilities, COM offset | no |

## Solver adequacy (measured, not assumed)

`scripts/smoke/validate_solvers.py` fits a free per-episode θ against real
transitions and compares to a persistence baseline (`s_next = s`). Scaled
RMSE, lower is better:

| benchmark | persistence | nominal θ | fitted θ (oracle) |
|---|---|---|---|
| pokeworld | 0.312 | 0.228 | **0.0001** |
| cartpole  | 0.166 | 0.006 | **0.0025** |
| pusht     | 0.352 | 0.031 | **0.021** |

Cartpole recovers physically correct parameters from MuJoCo data
(gravity ≈ 9.81, cart mass ≈ 1.01, pole half-length ≈ 0.517), and PushT
recovers the environment's exact controller gains (`k_p` = 100,
`k_v` = 20). Two caveats are real and deliberate:

- **PushT's block channels are the weak point.** The T-block is modelled
  as a disc, so block translation improves only ~35% over persistence and
  rotation ~3%. The agent channels are exact. This structural mismatch is
  documented in `PushTSolver` and is part of what the benchmark tests.
- **PokeWorld is degenerate by construction.** Only `k/m` and `c/m`
  affect the trajectory, so mass, stiffness and drag are *not*
  individually identifiable. The R² certificate reports this rather than
  hiding it — and parameters that are constant across episodes correctly
  return `nan`, since R² is undefined there.

## Two environment gotchas this pipeline handles

- PushT's state ends in the **agent's** velocity, not the block's
  (`_get_obs` = agent pos + block pos + block angle + *agent* velocity);
  the `vel_block` name in the env's `_set_state` is a misnomer.
- PushT reports `block.angle % 2π`, whose wrap jumps no smooth model can
  predict. Episodes are unwrapped per-episode in `data.py`; without this
  the angular channel's error is ~9× larger and swamps everything.

## Running

```bash
export MUJOCO_GL=egl                      # dm_control rendering, headless

python scripts/smoke/run_smoke.py         # full smoke suite
python scripts/smoke/validate_solvers.py  # solver adequacy diagnostic
pytest tests/wm/test_physwm.py            # design invariants

python scripts/train/physwm.py bench=pokeworld
python scripts/train/physwm.py bench=cartpole trainer.max_epochs=50
python scripts/train/physwm.py bench=pusht loss.alpha=0.5 loss.beta=0.1
python scripts/train/physwm.py bench=pokeworld probe.mode=step wandb.enabled=true
```

Checkpoints land in `$STABLEWM_HOME/checkpoints/<output_model_name>/` and
reload through the repo's loader:

```python
from stable_worldmodel.wm.utils import load_pretrained
model = load_pretrained('physwm_pokeworld/weights.pt')
```

## Note on the training loop

Unlike the other `scripts/train/*.py`, this uses a plain PyTorch loop
rather than lightning + `stable-pretraining`. The two-path objective, the
θ diagnostics and the frozen-solver invariants are easier to keep
explicit, and it keeps the prototype runnable without the `train` extra.
Config, seeding, W&B logging and checkpointing follow the repo's
conventions.
