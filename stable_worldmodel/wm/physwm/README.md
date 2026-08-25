# PhysWM — a physics-grounded world model

Two next-state predictions from the **same action-conditioned** DINO-WM
latent. Path A is
supervised on the dataset's ground-truth `s_next`; Path B is supervised on
Path A's own (detached) prediction instead — the experiment is whether a
physics parameterization can explain what the world model *itself*
believes happens next, not whether the solver can re-derive the raw
benchmark label:

```
pixels ── encoder ──► z ── predictor(z, a) ──► ẑ ──┬── decoder ──────────────► s_A   (← s_next)
                                                   └── linear probe ─► θ ──► frozen solver(s_t, a_t, θ) ─► s_B
                                                                                                         (← s_A.detach())
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
| Two predictions branch off the same **action-conditioned** latent | `PhysWM.forward` — both heads read `z_hat`; `probe_source=encoded` is an ablation only |
| A regresses ground truth; **B regresses A's detached output** | `loss.physwm_loss`; `L_B` cannot train the decoder through its target edge, but deliberately shapes the shared predictor through the probe input |
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
| `pokeworld` | `[x, y, vx, vy, touch]` | mass, contact stiffness, drag, radii | **yes** → R² certificate |
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

### PokeWorld identifiability

Ranges and observability follow the reference
[PokeWorld](https://tantansir.github.io/latent-world-model-identifiability/):
mass 0.5–3, contact stiffness 500–6000, drag 0.5–4, resampled per episode
and **rendered identically** — the physics is invisible in any single
frame and shows up only in dynamics.

From motion alone the dynamics depend only on `k/m` and `c/m`, so the
three parameters are *not* individually identifiable. The state's fifth
channel, `touch` (peak contact force over the transition's sub-steps),
breaks that degeneracy: the contact peak pins `k`, contact impulse
against the resulting velocity change pins `m`, and glide decay pins `c`.

Oracle recoverability certificate — free per-episode θ fitted directly to
true transitions, 64 episodes × 48, the ceiling no pixel-reading probe can
beat:

| | mass | stiffness | drag |
|---|---|---|---|
| with `touch` | **0.834** | **0.755** | **0.754** |
| without `touch` (motion only) | 0.009 | 0.084 | −0.214 |

The radii are constant by construction, so their R² is undefined (`nan`).
Integration uses `dt=0.001 × 20` substeps: the reference stiffness range
needs a fine sub-step both for stability and to resolve the tactile peak.

**The tactile channel is a model input, not only a target, in the raw-parameter
certificate.** The pixels are
identical across θ and motion alone fixes only `k/m` and `c/m`, so a probe
reading pixels could never separate mass, stiffness and drag however long it
trained — a target the model cannot observe teaches nothing. `touch` is
therefore fed to the encoder as one extra token per frame (`tactile_index` in
the benchmark spec), alongside remaining part of the 5-d prediction target.
It arrives under its own batch key `touch`, never read out of `state`: that
array is the supervision target, and indexing it for inputs is how leakage
starts. With the strict `probe_context` split, no query-frame touch reading is
visible to the identifier.

For the paper's **visual-only** condition, set `tactile.enabled=false` and
report the dynamics coordinates that pixels can identify: `stiffness/mass`,
`drag/mass`, and the summed contact radius. Reporting raw mass, stiffness and
drag R² in that condition would be mathematically invalid. The sweep reports
both coordinate systems and treats visual+tactile raw-parameter recovery as a
separate observability setting. The unobserved touch output is also excluded
from Path A/Path B losses and RMSE in visual-only mode; leaving it in would make
the accurate-predictor premise fail by construction.

`touch` is stored as `log1p(peak force)`. Raw peak force is zero-inflated
(contact fires in ~5% of transitions) with a 0–700 range, so after z-scoring
the silent majority collapses to one value and contacts become +20σ outliers —
squared error on that is dominated by rare spikes. Compressed, the same channel
spans 0–6.6 and peaks at +6.5σ.

Cartpole recovers physically correct parameters from MuJoCo data
(gravity ≈ 9.81, cart mass ≈ 1.01, pole half-length ≈ 0.517), and PushT
recovers the environment's exact controller gains (`k_p` = 100,
`k_v` = 20). Two caveats are real and deliberate:

- **PushT's block channels are the weak point.** The T-block is modelled
  as a disc, so block translation improves only ~35% over persistence and
  rotation ~3%. The agent channels are exact. This structural mismatch is
  documented in `PushTSolver` and is part of what the benchmark tests.
- **Visual-only PokeWorld is degenerate in raw coordinates.** Only `k/m`
  and `c/m` affect the trajectory, so the paper reports those identifiable
  dynamics coordinates for visual-only runs. Raw mass/stiffness/drag R² is
  reserved for the touch-observed condition; constant radii return `nan`.

## Two environment gotchas this pipeline handles

- PushT's state ends in the **agent's** velocity, not the block's
  (`_get_obs` = agent pos + block pos + block angle + *agent* velocity);
  the `vel_block` name in the env's `_set_state` is a misnomer.
- PushT reports `block.angle % 2π`, whose wrap jumps no smooth model can
  predict. Episodes are unwrapped per-episode in `data.py`; without this
  the angular channel's error is ~9× larger and swamps everything.

## Workshop-paper experiment protocol

`scripts/eval/overnight.py` is the canonical matrix. Each group maps to one
claim or necessary control; it no longer sweeps arbitrary scale knobs before
testing the method's defining choices.

| group | comparison | paper question |
|---|---|---|
| A | predictive vs PhysWM vs theta-oracle, 3 seeds | accurate-but-blind premise and primary recovery result |
| B | `probe_source=predicted` vs `encoded`, 3 seeds | is the shared action-conditioned latent necessary? |
| C | Path-A teacher vs dataset target; induced vs detached/post-hoc | what supplies the gain? |
| D | visual-only, scored in `k/m` and `c/m` coordinates | does the visual-encoder claim survive valid identifiability? |
| E | frozen DINOv2-small, 3 seeds | does the result hold for the paper encoder? |
| F | true/inferred/shuffled/nominal substitution and rollout | are recovered parameters functionally physical? |
| G | probe capacity and context length | robustness checks |

Every decodability run reports, from the **same checkpoints**, Path-A error
against the dataset, persistence error, Path-B fidelity to the model teacher,
Path-B error against the dataset, raw-parameter R², and identifiable-dynamics
R². This prevents prediction accuracy and parameter recovery from being
silently paired across different models. Theta predictions and supervised
features are averaged within each episode before R², so overlapping windows
do not masquerade as independent samples or overweight contact-rich episodes.
The runner writes per-run JSON/log/checkpoints plus `summary.json` and a
paper-ready `summary.md` with mean and sample standard deviation by group.

The primary claim is supported only if Path A beats persistence, the
predictive baseline is weak on parameter recovery, PhysWM improves over that
baseline across seeds, and inferred theta beats shuffled/nominal theta in
held-out rollouts. The theta-oracle is a ceiling, not a method result.

Episode-level theta uses a strict context/query split. With the default
8-frame window and `probe_context=4`, the probe reads causal predictive
latents through index 4 (no observation after frame 4). The physical loss fits
the completed context transitions 0–3; all reported Path-B fidelity uses the
held-out query transitions 4–6. Functional rollouts use a 16-frame window,
eight context transitions, and seven held-out query transitions. Unit tests
perturb every query outcome and verify that inferred theta is unchanged.
For the raw-parameter visual+tactile condition, windows must contain at least
one completed contact inside the context; the selector never inspects a query
outcome. The visual-only condition disables this hidden-modality filter.

## Running

```bash
export MUJOCO_GL=egl                      # dm_control rendering, headless

python scripts/smoke/run_smoke.py         # full smoke suite
python scripts/smoke/validate_solvers.py  # solver adequacy diagnostic
pytest tests/wm/test_physwm.py            # design invariants

python scripts/train/physwm.py bench=pokeworld
python scripts/train/physwm.py bench=pokeworld bench.probe_source=encoded # routing ablation
python scripts/train/physwm.py bench=pokeworld bench.tactile.enabled=false # visual-only
python scripts/train/physwm.py bench=cartpole trainer.max_epochs=50
python scripts/train/physwm.py bench=pusht loss.alpha=0.5 loss.beta=0.1
python scripts/train/physwm.py bench=pokeworld probe.mode=step wandb.enabled=true

# H100 80 GB: DINOv2-small/224px, BF16, 256 episodes, 50 epochs, W&B,
# and automatic full-state resume. Run the seeds sequentially on one GPU.
python scripts/train/physwm.py bench=cartpole hardware=h100 seed=0
python scripts/train/physwm.py --multirun bench=cartpole hardware=h100 seed=0,1,2

# RTX 4090 24 GB: BF16 batch 32 (measured ~3.7 GB peak of 24 GB).
python scripts/train/physwm.py bench=pokeworld hardware=rtx4090 seed=0
```

The H100 profile defaults to batch size 16 to leave safe memory headroom.
After a one-epoch profile on the rented host, try `loader.batch_size=32` if
peak allocated memory permits it. Each seed has a separate output directory,
and rerunning the same command resumes from its atomic `last.pt` checkpoint.
It also uses eight persistent data-loader workers; lower `loader.num_workers`
if the rented VM has fewer CPU cores or limited shared memory.

Each run writes `run_meta.json` alongside its checkpoints — commit sha and
dirty flag, exact command, host, torch/CUDA build, GPU name, and a
**fingerprint** of the experiment-defining config. `trainer.resume` refuses a
checkpoint whose fingerprint differs, so a sweep can never silently continue
another run's weights. `output_model_name` is seed-scoped by default; give
any other sweep dimension its own name.

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
