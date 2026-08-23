"""Config-driven assembly of a :class:`PhysWM` and its data.

Every component is chosen by name from config, so encoder, predictor,
probe, solver and dataset are swappable without touching the training
loop. Wiring that depends on another module's shape (encoder width ->
predictor width, solver theta_dim -> probe output) is resolved here
rather than through fragile config interpolation.
"""

from stable_worldmodel.wm.lewm.module import Embedder

from .data import EpisodeWindowDataset, env_episodes, pokeworld_episodes
from .module import (
    DinoV2Encoder,
    DinoWMPredictor,
    StateDecoder,
    StateEncoder,
    ThetaProbe,
    TinyCNNEncoder,
)
from .physwm import PhysWM
from .solvers import build_solver

#: benchmark name -> (solver name, state_dim, action_dim)
BENCHMARKS = {
    # poked disc; ground-truth theta available -> identifiability certificate
    'pokeworld': {
        'solver': 'pokeworld',
        'state_dim': 4,
        'action_dim': 2,
        'has_theta_true': True,
    },
    # dm_control cartpole; state = [x, angle, x_dot, angle_dot]
    'cartpole': {
        'solver': 'cartpole',
        'state_dim': 4,
        'action_dim': 1,
        'env_id': 'swm/CartpoleDMControl-v0',
        'has_theta_true': False,
    },
    # PushT; state = [agent_xy, block_xy, block_angle, block_vel_xy]
    'pusht': {
        'solver': 'pusht',
        'state_dim': 7,
        'action_dim': 2,
        'env_id': 'swm/PushT-v1',
        'has_theta_true': False,
    },
}


def build_encoder(cfg, state_dim):
    """``cfg.name`` in ``{tiny_cnn, dinov2, state}``."""
    name = cfg['name']
    if name == 'tiny_cnn':
        return TinyCNNEncoder(
            image_size=cfg.get('image_size', 64),
            in_channels=cfg.get('in_channels', 3),
            embed_dim=cfg.get('embed_dim', 128),
            patch_size=cfg.get('patch_size', 8),
            width=cfg.get('width', 64),
        )
    if name == 'dinov2':
        return DinoV2Encoder(
            model_name=cfg.get('model_name', 'facebook/dinov2-small'),
            frozen=cfg.get('frozen', True),
            image_size=cfg.get('image_size', 224),
        )
    if name == 'state':
        return StateEncoder(
            state_dim=state_dim, embed_dim=cfg.get('embed_dim', 128)
        )
    raise KeyError(f'unknown encoder {name!r}')


def build_physwm(cfg, state_dim: int, action_dim: int) -> PhysWM:
    """Assemble the full model from a (dict-like) config."""
    enc_cfg = cfg['encoder']
    encoder = build_encoder(enc_cfg, state_dim)
    embed_dim = encoder.embed_dim
    num_tokens = encoder.num_tokens
    obs_key = 'state' if enc_cfg['name'] == 'state' else 'pixels'

    solver = build_solver(
        cfg['solver']['name'],
        dt=cfg['solver']['dt'],
        substeps=cfg['solver']['substeps'],
    )

    act_cfg = cfg['action_encoder']
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=act_cfg.get('smoothed_dim', action_dim),
        emb_dim=act_cfg.get('emb_dim', 32),
        mlp_scale=act_cfg.get('mlp_scale', 4),
    )

    pred_cfg = cfg['predictor']
    predictor = DinoWMPredictor(
        embed_dim=embed_dim,
        action_dim=act_cfg.get('emb_dim', 32),
        num_tokens=num_tokens,
        max_frames=pred_cfg.get('max_frames', 16),
        hidden_dim=pred_cfg.get('hidden_dim', 256),
        depth=pred_cfg.get('depth', 4),
        heads=pred_cfg.get('heads', 8),
        dim_head=pred_cfg.get('dim_head', 32),
        mlp_dim=pred_cfg.get('mlp_dim', 512),
        dropout=pred_cfg.get('dropout', 0.0),
    )

    dec_cfg = cfg['decoder']
    state_decoder = StateDecoder(
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        state_dim=state_dim,
        hidden_dim=dec_cfg.get('hidden_dim', 256),
        depth=dec_cfg.get('depth', 2),
        pool=dec_cfg.get('pool', 'mean'),
    )

    probe_cfg = cfg['probe']
    probe = ThetaProbe(
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        theta_dim=solver.theta_dim,
        hidden_dim=probe_cfg.get('hidden_dim', 0),
        mode=probe_cfg.get('mode', 'episode'),
        pool=probe_cfg.get('pool', 'mean'),
        dropout=probe_cfg.get('dropout', 0.0),
        detach_input=probe_cfg.get('detach_input', False),
        init_scale=probe_cfg.get('init_scale', 0.01),
    )
    probe.init_from_solver(solver)

    return PhysWM(
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        state_decoder=state_decoder,
        probe=probe,
        solver=solver,
        state_dim=state_dim,
        action_dim=action_dim,
        obs_key=obs_key,
        probe_frames=cfg.get('probe_frames', 'context'),
        solver_state_source=cfg.get('solver_state_source', 'gt'),
    )


def build_episodes(cfg, seed: int):
    """Build train/val episode lists for the configured benchmark."""
    name = cfg['benchmark']
    spec = BENCHMARKS[name]
    d = cfg['data']
    render = cfg['encoder']['name'] != 'state'

    if name == 'pokeworld':
        train, _ = pokeworld_episodes(
            num_episodes=d['num_episodes'],
            length=d['episode_length'],
            seed=seed,
            render=render,
            image_size=cfg['encoder'].get('image_size', 64),
            **d.get('sim', {}),
        )
        val, _ = pokeworld_episodes(
            num_episodes=max(1, d['num_episodes'] // 4),
            length=d['episode_length'],
            seed=seed + 10_000,
            render=render,
            image_size=cfg['encoder'].get('image_size', 64),
            **d.get('sim', {}),
        )
        return train, val

    kwargs = dict(
        env_id=spec['env_id'],
        length=d['episode_length'],
        frameskip=d.get('frameskip', 1),
        image_size=cfg['encoder'].get('image_size', 64),
        render=render,
    )
    train = env_episodes(num_episodes=d['num_episodes'], seed=seed, **kwargs)
    val = env_episodes(
        num_episodes=max(1, d['num_episodes'] // 4),
        seed=seed + 10_000,
        **kwargs,
    )
    return train, val


def build_datasets(cfg, seed: int):
    """Episodes -> windowed torch datasets."""
    train_eps, val_eps = build_episodes(cfg, seed)
    window = cfg['data']['window']
    stride = cfg['data'].get('stride', 1)
    return (
        EpisodeWindowDataset(train_eps, window=window, stride=stride),
        EpisodeWindowDataset(val_eps, window=window, stride=stride),
    )


__all__ = [
    'BENCHMARKS',
    'build_datasets',
    'build_encoder',
    'build_episodes',
    'build_physwm',
]
