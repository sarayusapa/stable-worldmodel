"""PhysWM: a physics-grounded world model prototype.

Two next-state predictions from the same action-conditioned DINO-WM latent -- one learned
(latent prediction decoded to state), one physical (probe -> theta ->
frozen differentiable solver). The learned path is supervised on the
dataset's ground-truth next state; the physical path is supervised on the
learned path's own (detached) prediction, so the physics probe is fit to
explain the world model's general belief about the next state rather than
the raw benchmark label.
"""

from .build import (
    BENCHMARKS,
    build_datasets,
    build_encoder,
    build_episodes,
    build_physwm,
)
from .data import (
    EpisodeWindowDataset,
    PokeWorldSim,
    collect_stats,
    env_episodes,
    pokeworld_episodes,
    swm_dataset_episodes,
)
from .loss import physwm_loss, physwm_metrics, theta_r2
from .module import (
    DinoV2Encoder,
    DinoWMPredictor,
    StateDecoder,
    StateEncoder,
    ThetaProbe,
    TinyCNNEncoder,
)
from .physwm import Normalizer, PhysWM, assert_no_free_theta
from .solvers import (
    SOLVERS,
    CartpoleSolver,
    PhysicsSolver,
    PokeWorldSolver,
    PushTSolver,
    build_solver,
)

__all__ = [
    'BENCHMARKS',
    'SOLVERS',
    'CartpoleSolver',
    'DinoV2Encoder',
    'DinoWMPredictor',
    'EpisodeWindowDataset',
    'Normalizer',
    'PhysWM',
    'PhysicsSolver',
    'PokeWorldSim',
    'PokeWorldSolver',
    'PushTSolver',
    'StateDecoder',
    'StateEncoder',
    'ThetaProbe',
    'TinyCNNEncoder',
    'assert_no_free_theta',
    'build_datasets',
    'build_encoder',
    'build_episodes',
    'build_physwm',
    'build_solver',
    'collect_stats',
    'env_episodes',
    'physwm_loss',
    'physwm_metrics',
    'pokeworld_episodes',
    'swm_dataset_episodes',
    'theta_r2',
]
