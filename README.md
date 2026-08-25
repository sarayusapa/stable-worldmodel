<h1 align="center">stable-worldmodel</h1>

<p align="center"><i>Probing physics from a general-purpose world model's own predictions — built on the stable-worldmodel platform.</i></p>

<p align="center">
  <a href="https://galilai-group.github.io/stable-worldmodel/"><img alt="Documentation" src="https://img.shields.io/badge/Docs-blue.svg"/></a>
  <a href="https://github.com/galilai-group/stable-worldmodel"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/galilai-group/stable-worldmodel/tests.yaml?label=Tests"/></a>
  <a href="https://pypi.python.org/pypi/stable-worldmodel/#history"><img alt="PyPI" src="https://img.shields.io/pypi/v/stable-worldmodel.svg"/></a>
  <a href="https://arxiv.org/abs/2605.21800v1" target="_blank" style="margin: 2px;"><img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2605.21800-b5212f?logo=arxiv" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"/></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"/></a>
  <a href="https://colab.research.google.com/github/galilai-group/stable-worldmodel/blob/main/scripts/notebooks/train_from_hf_buckets.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

<p align="center">
  <a href="#the-plan-physwm"><b>The Plan (PhysWM)</b></a> ·
  <a href="#running-physwm"><b>Running It</b></a> ·
  <a href="#the-platform-stable-worldmodel"><b>The Platform</b></a> ·
  <a href="#installation"><b>Installation</b></a> ·
  <a href="https://galilai-group.github.io/stable-worldmodel/"><b>Documentation</b></a> ·
  <a href="#citation"><b>Citation</b></a>
</p>

---

## The plan: PhysWM

This repo is home to an ongoing experiment, **PhysWM**, that asks a narrow
question: when a world model already predicts its own general next state,
can a small, physics-parameterized model be probed *out of the same
action-conditioned predictive latent* to explain that prediction — rather
than being fit directly to whatever the benchmark happened to record?

Two next-state predictions branch off one predictive latent `ẑ`:

```
pixels ─► encoder ─► z ─► predictor(z, a) ─► ẑ ─┬─► decoder ─────────────► s_A   (learned, ← s_next)
                                                 └─► probe ─► θ ─► solver ─► s_B   (physical, ← s_A.detach())
                                                                 (s_t, a_t, θ)

L = L_A + α·L_B + β·L_consistency          (β = 0 by default)
```

- **Path A (learned)** is the general-purpose next-state prediction: latent
  in, latent predicted forward, decoded to state. It is supervised on the
  dataset's actual `s_next` — it has to stay anchored to reality.
- **Path B (physical)** reads a low-capacity probe off the same
  action-conditioned predictive latent to
  emit `θ`, a vector of physics parameters (mass, gains, stiffness, ...),
  and runs it through a **frozen, differentiable, zero-parameter** physics
  solver to produce an independent next-state estimate.

**The ground-truth question, and why it isn't the benchmark label.** Once
both paths exist, what should Path B be compared against? Not itself,
trivially — the experiment is whether physics *explains* something, not
whether the solver can be made to output anything at all. That leaves two
real candidates: the benchmark's recorded `s_next`, or Path A's own
prediction. We use **Path A, detached, as Path B's target**.

The motivation is a hierarchy-of-representations claim, not just a
methodological preference: a world model can get plain next-state
prediction decent without ever internalizing the physics that produced
it — Path A can be a good high-level predictor while remaining a bad
physicist. Fitting Path B to the raw dataset label conflates those two
questions and mostly re-tests the first one, which Path A already answers,
usually better. Fitting Path B to Path A's prediction instead isolates the
second question: starting from what the high-level predictor already gets
right, can a low-capacity, physically-constrained probe recover the
compact physical parameterization *underneath* it? That's filling in the
hierarchy — using the general prediction the model is already decent at as
the training signal for the physics-aware latent structure it hasn't
learned on its own. The detach fixes Path A's output as the teacher value:
`L_B` cannot optimize the decoder through the target edge. Its gradient does,
deliberately, reach the shared predictor through the probe input, which is how
the physical loss induces structure in `ẑ`. Detaching that input is the
post-hoc-probe ablation.

`θ` is always a forward-pass output of the probe — never a free variable
fitted directly to a target — so gradients into the probe only ever arrive
by flowing through the frozen solver. Three solvers ship with it —
`pokeworld` (synthetic, ground-truth `θ` for an identifiability
certificate), `cartpole` (dm_control), and `pusht` — and each one's actual
ability to fit its environment is measured, not assumed, by
`scripts/smoke/validate_solvers.py`.

See [`stable_worldmodel/wm/physwm/README.md`](stable_worldmodel/wm/physwm/README.md)
for the full design rules, the invariants that enforce them (`tests/wm/test_physwm.py`),
and measured solver-adequacy results per benchmark.

## Running PhysWM

```bash
export MUJOCO_GL=egl                      # dm_control rendering, headless

pytest tests/wm/test_physwm.py            # design invariants
python scripts/smoke/run_smoke.py         # full smoke suite
python scripts/smoke/validate_solvers.py  # solver adequacy diagnostic

python scripts/train/physwm.py bench=pokeworld
python scripts/train/physwm.py bench=cartpole trainer.max_epochs=50
python scripts/train/physwm.py bench=pusht loss.alpha=0.5 loss.beta=0.1
python scripts/eval/overnight.py --dry-run  # inspect the workshop matrix

# Serious H100 profile: DINOv2-small at 224px, BF16, 256 episodes, 50 epochs
python scripts/train/physwm.py bench=cartpole hardware=h100 seed=0
```

Log every run — command, config, seed, hardware and results — in
[`progress.md`](progress.md); the paper's numbers are traced back to it.

## The platform: `stable-worldmodel`

PhysWM isn't a separate codebase bolted on the side — it's built as one of
the `stable_worldmodel.wm` model families and leans directly on the
platform this repo provides:

- **Environments** (`swm/*`) supply the live rollouts Path B's solver is
  checked against — `pusht`, `cartpole` (dm_control) — plus a synthetic
  `pokeworld` env for the identifiability certificate.
- **The dataset pipeline** (Lance / HDF5 / video, see below) supplies the
  offline transition windows PhysWM trains on, through the same
  `swm.data.load_dataset` autodetection every other baseline in this repo
  uses.
- **Shared building blocks** — the block-causal spatio-temporal
  predictor, the DINO backbone encoder — make Path A a drop-in
  general-purpose next-state predictor rather than a bespoke model.
- **Planning solvers** (CEM, MPPI, ...) are what a trained PhysWM
  checkpoint would ultimately be evaluated with under model-predictive
  control, the same as `DINO-WM`, `PLDM`, or any other baseline here.

Everything from here down — installation, data formats, the environment
suite, the planning solvers, the other baselines — is that shared
substrate: the infrastructure the plan above runs on, reusable by any
world model research built in this repo, including PhysWM's.

## Installation

From PyPI:

```bash
pip install 'stable-worldmodel[data]'    # recommended: base + Lance dataset I/O
pip install stable-worldmodel            # planning/model only, no dataset I/O
pip install 'stable-worldmodel[all]'     # + training, environments, and data formats
```

`[data]` pulls in the Lance stack (`lancedb`, `pylance`, `pyarrow`) that backs the
default dataset format. It is a separate extra so that consumers who only need the
solvers and world model — e.g. embedding `stable_worldmodel.planning` in a robotics
image — are not forced to install ~410 MB of native wheels they never import.

LeRobot dataset support is a separate opt-in extra (requires Python 3.12+): `pip install 'stable-worldmodel[lerobot]'`.

From source (development):

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel
uv venv --python=3.10 && source .venv/bin/activate
uv sync --extra all --group dev
```

Datasets and checkpoints are stored under `$STABLEWM_HOME` (defaults to `~/.stable_worldmodel/`). Override the variable to point at your preferred storage location.

> The library is in active development. APIs may change between minor versions.

## Quick Start

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

# 1. Collect a dataset
world = swm.World("swm/PushT-v1", num_envs=8)
world.set_policy(your_expert_policy)
world.collect("data/pusht_demo.lance", episodes=100, seed=0)

# 2. Load it and train your world model (format is autodetected)
dataset = swm.data.load_dataset("data/pusht_demo.lance", num_steps=16)
world_model = ...  # your model

# 3. Evaluate with model-predictive control
solver = CEMSolver(model=world_model, num_samples=300)
policy = WorldModelPolicy(
    solver=solver,
    config=PlanConfig(horizon=10, receding_horizon=5),
)

world.set_policy(policy)
results = world.evaluate(episodes=50)
print(f"Success Rate: {results['success_rate']:.1f}%")
```

Reference implementations are provided in [`scripts/train/`](scripts/train): [`lewm.py`](scripts/train/lewm.py) implements [LeWM](https://le-wm.github.io/), and [`prejepa.py`](scripts/train/prejepa.py) reproduces [DINO-WM](https://arxiv.org/abs/2411.04983). To train directly from HuggingFace object storage with no local dataset download, see the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galilai-group/stable-worldmodel/blob/main/scripts/notebooks/train_from_hf_buckets.ipynb) notebook.

<p align="center">
  <img src="docs/assets/lewm-gpu-usage.png" alt="GPU utilization comparison" width="60%">
  <br>
  <em>GPU utilization for LeWM trained with  Push-T LanceDB dataset on a H200 GPU.</em>
</p>

## Data Formats

Recording, loading, and conversion all go through a small **format registry**. Pick the backend that matches your trade-off, or [register your own](https://galilai-group.github.io/stable-worldmodel/api/dataset/#registering-a-custom-format).

| Format    | On-disk layout                                  | Best for                                       |
|-----------|-------------------------------------------------|------------------------------------------------|
| `lance`   | LanceDB table (episode-contiguous flat rows)    | default — append-friendly, fast indexed reads  |
| `hdf5`    | single `.h5` file (one dataset per column)      | portable single-file artifact                  |
| `folder`  | `.npz` columns + one JPEG per step              | inspection, partial-key streaming              |
| `video`   | `.npz` columns + one MP4 per episode (`decord`) | long episodes, compact image storage           |
| `lerobot` | `lerobot://<repo_id>` (read-only adapter)       | training/eval directly on LeRobot Hub datasets |

```python
world.collect("data/pusht.lance", episodes=100)                  # default: lance
world.collect("data/pusht_video", episodes=100, format="video")  # mp4 episodes

ds = swm.data.load_dataset("data/pusht.lance", num_steps=16)     # autodetect
swm.data.convert("data/pusht.lance", "data/pusht_video",
                 dest_format="video", fps=30)                    # one-shot migration
```

Every writer accepts a `mode` kwarg (`'append'` (default), `'overwrite'`, `'error'`); re-running `world.collect` extends the existing dataset rather than failing.

<details>
<summary><b>Throughput & storage benchmarks</b></summary>

Numbers below were produced by [`scripts/benchmark/compare_h5_lance.py`](scripts/benchmark/compare_h5_lance.py) and can be reproduced with it. Benchmarks use the [PushT dataset](https://huggingface.co/datasets/galilai-group/lewm-pusht) from the [LeWorldModel](https://le-wm.github.io/) paper.

## Throughput

| Format  | Source   | Cache    | samples/s | ms/step  |
|---------|----------|----------|-----------|----------|
| HDF5    | local    | no-cache |    1416.1 |     45.2 |
| HDF5    | local    | cached   |    1474.0 |     43.4 |
| LanceDB | local    | no-cache |    4814.8 |     13.3 |
| LanceDB | local    | cached   |    4431.3 |     14.4 |
| Video   | local    | -        |    1330.6 |     48.1 |
| LanceDB | s3       | no-cache |    3183.7 |     20.1 |
| LanceDB | s3       | cached   |    3253.2 |     19.7 |
| HDF5    | s3       | no-cache |       9.1 |   7032.5 |
| HDF5    | s3       | cached   |     756.5 |     84.6 |

## Storage size per format (local)

| Format  | Local size |
|---------|------------|
| HDF5    |   43.12 GB |
| LanceDB |   13.31 GB |
| Video   |  496.29 MB |

</details>

## Environments

<div align="center">

<table>
<tr>
<td align="center"><img src="docs/assets/ballincup.gif" width="120"/><br><img src="docs/assets/ballincup_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/cartpole.gif"  width="120"/><br><img src="docs/assets/cartpole_var.gif"  width="120"/></td>
<td align="center"><img src="docs/assets/cheetah.gif"   width="120"/><br><img src="docs/assets/cheetah_var.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/finger.gif"    width="120"/><br><img src="docs/assets/finger_var.gif"    width="120"/></td>
<td align="center"><img src="docs/assets/hopper.gif"    width="120"/><br><img src="docs/assets/hopper_var.gif"    width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/pendulum.gif"  width="120"/><br><img src="docs/assets/pendulum_var.gif"  width="120"/></td>
<td align="center"><img src="docs/assets/quadruped.gif" width="120"/><br><img src="docs/assets/quadruped_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/reacher.gif"   width="120"/><br><img src="docs/assets/reacher_var.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/walker.gif"    width="120"/><br><img src="docs/assets/walker_var.gif"    width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/cartpole_control.gif"     width="120"/><br><img src="docs/assets/cartpole_control_var.gif"     width="120"/></td>
<td align="center"><img src="docs/assets/mountain_car_control.gif" width="120"/><br><img src="docs/assets/mountain_car_control_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/acrobot_control.gif"      width="120"/><br><img src="docs/assets/acrobot_control_var.gif"      width="120"/></td>
<td align="center"><img src="docs/assets/pendulum_control.gif"     width="120"/><br><img src="docs/assets/pendulum_control_var.gif"     width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/cube.gif"    width="120"/><br><img src="docs/assets/cube_fov.gif"    width="120"/></td>
<td align="center"><img src="docs/assets/scene.gif"   width="120"/><br><img src="docs/assets/scene_fov.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/pusht.gif"   width="120"/><br><img src="docs/assets/pusht_fov.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/tworoom.gif" width="120"/><br><img src="docs/assets/tworoom_fov.gif" width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/fetch_reach.gif"        width="120"/><br><img src="docs/assets/fetch_reach_var.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/fetch_push.gif"         width="120"/><br><img src="docs/assets/fetch_push_var.gif"         width="120"/></td>
<td align="center"><img src="docs/assets/fetch_slide.gif"        width="120"/><br><img src="docs/assets/fetch_slide_var.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/fetch_pickandplace.gif" width="120"/><br><img src="docs/assets/fetch_pickandplace_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/craftax_classic.gif"    width="120"/><br><img src="docs/assets/craftax.gif"                width="120"/></td>
</tr>
</table>

<em>Top row: default appearance &nbsp;·&nbsp; Bottom row: visual factor of variation</em>

</div>

Environments are pulled from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), [Gymnasium classic control](https://gymnasium.farama.org/environments/classic_control/), [OGBench](https://github.com/seohongpark/ogbench), [Craftax](https://github.com/MichaelTMatthews/Craftax), the [Arcade Learning Environment](https://ale.farama.org/) (100+ Atari games), and classical world model benchmarks ([Two-Room](https://arxiv.org/abs/2411.04983), [PushT](https://arxiv.org/abs/2303.04137)). Most environments ship with a set of **factors of variation** — independently controllable visual and physical parameters (lighting, textures, dynamics, morphology) — that make it straightforward to evaluate zero-shot generalization to distribution shifts without any additional setup. Adding a new environment only requires conforming to the [Gymnasium](https://gymnasium.farama.org/) interface.

<details>
<summary><b>Full environment list</b></summary>

<div align="center">

| [Environment ID](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/envs) |  # FoV  |
|------------------------------|---------|
| swm/PushT-v1                 | 16      |
| swm/TwoRoom-v1               | 17      |
| swm/OGBCube-v0               | 11      |
| swm/OGBScene-v0              | 12      |
| swm/HumanoidDMControl-v0     | 7       |
| swm/CheetahDMControl-v0      | 7       |
| swm/HopperDMControl-v0       | 7       |
| swm/ReacherDMControl-v0      | 8       |
| swm/WalkerDMControl-v0       | 8       |
| swm/AcrobotDMControl-v0      | 8       |
| swm/PendulumDMControl-v0     | 6       |
| swm/CartpoleDMControl-v0     | 6       |
| swm/BallInCupDMControl-v0    | 9       |
| swm/FingerDMControl-v0       | 10      |
| swm/ManipulatorDMControl-v0  | 8       |
| swm/QuadrupedDMControl-v0    | 7       |
| swm/CartPoleControl-v1       | 10      |
| swm/MountainCarControl-v0    | 5       |
| swm/MountainCarContinuousControl-v0 | 4 |
| swm/AcrobotControl-v1        | 11      |
| swm/PendulumControl-v1       | 9       |
| swm/FetchReach-v3            | 8       |
| swm/FetchPush-v3             | 11      |
| swm/FetchSlide-v3            | 11      |
| swm/FetchPickAndPlace-v3     | 11      |
| swm/CraftaxClassicPixels-v1  | —       |
| swm/CraftaxClassicSymbolic-v1| —       |
| swm/CraftaxPixels-v1         | —       |
| swm/CraftaxSymbolic-v1       | —       |
| [ALE/* (100+ Atari games)](https://ale.farama.org/) | — |

</div>

</details>

## Solvers and Baselines

<div align="center">

| [Solver](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/solver) | Type            |
|---------------------------------------|-----------------|
| Cross-Entropy Method (CEM)            | Sampling        |
| Improved CEM (iCEM)                   | Sampling        |
| Model Predictive Path Integral (MPPI) | Sampling        |
| Predictive Sampling                   | Sampling        |
| Gradient Descent (SGD, Adam)          | Gradient        |
| Projected Gradient Descent (PGD)      | Gradient        |
| Augmented Lagrangian                  | Constrained Opt |

| [Baseline](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) | Type              |
|----------|-------------------|
| DINO-WM  | JEPA              |
| PhysWM   | JEPA + Physics    |
| PLDM     | JEPA              |
| LeWM     | JEPA              |
| GCBC     | Behaviour Cloning |
| GCIVL    | RL                |
| GCIQL    | RL                |

</div>

`PhysWM` is this repo's active project — see [The Plan](#the-plan-physwm) at
the top of this document for the architecture, the objective, and why its
physics path is fit to the learned path's prediction rather than the raw
benchmark label.

## Command-Line Interface

After installation, the `swm` command is available for inspecting/converting datasets, environments, and checkpoints without writing code:

```bash
swm datasets                                        # list cached datasets
swm inspect pusht_expert_train                      # inspect a specific dataset
swm envs                                            # list all registered environments
swm fovs PushT-v1                                   # show factors of variation for an environment
swm checkpoints                                     # list available model checkpoints
swm convert pusht_expert_train --dest-format video  # convert a dataset to another format
```

## Documentation

The full documentation lives at [galilai-group.github.io/stable-worldmodel](https://galilai-group.github.io/stable-worldmodel/), with API references, tutorials, and guides.

## Built on `stable-worldmodel`

- **[C-JEPA](https://hazel-heejeong-nam.github.io/cjepa/)**
- **[LeWM](https://le-wm.github.io/)**

## Citation

```bibtex
@misc{maes_lld2026swm,
  title  = {stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation},
  author = {Lucas Maes and Quentin Le Lidec and Luiz Facury and Nassim Massaudi and
            Ayush Chaurasia and Francesco Capuano and Richard Gao and Taj Gillin and
            Dan Haramati and Damien Scieur and Yann LeCun and Randall Balestriero},
  year   = {2026},
  eprint = {2605.21800},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2605.21800},
}
```

## Questions

Open an [issue](https://github.com/galilai-group/stable-worldmodel/issues) — happy to help.
