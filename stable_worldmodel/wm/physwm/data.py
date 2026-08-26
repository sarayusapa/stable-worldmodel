"""Data sources for PhysWM.

Three interchangeable sources, all yielding the same batch contract so the
training loop never has to care which one it is looking at:

    {'pixels': (T, C, H, W)      # or absent for state-only runs
     'state':  (T, S),           # physical units
     'action': (T, A),           # physical units
     'theta_true': (K,)}         # only where ground truth exists

* :func:`pokeworld_episodes` -- the identifiability benchmark. A poked
  disc with per-episode random mass / contact stiffness / drag, so the
  **true theta is known** and the probe can be scored with an R^2
  certificate.
* :func:`env_episodes` -- rollouts from any registered ``swm/*`` gym env
  (PushT, dm_control cartpole) under a random or supplied policy.
* :func:`swm_dataset_episodes` -- adapter over ``swm.data.load_dataset``
  for pre-collected Lance/HDF5 datasets.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# PokeWorld: the identifiability benchmark
# ----------------------------------------------------------------------


class PokeWorldSim:
    """Poked disc with contact stiffness and drag; known ground-truth theta.

    Follows ``latent-world-model-identifiability``'s PokeWorld setting:
    a poker pushes an object subject to viscous drag, and we ask which
    physics parameters a latent world model actually recovers.

        m * v_dot = k * penetration * n_hat - c * |v|^(p-1) * v

    ``drag_exponent`` ``p`` defaults to 1 (linear drag), which the
    :class:`PokeWorldSolver` models exactly. Set ``p != 1`` to introduce
    deliberate structural mismatch between the data and the solver.

    Parameter ranges and the observability structure follow the reference
    PokeWorld (``latent-world-model-identifiability``): the three
    parameters span a deliberate observability spectrum -- drag is chiefly
    visual (glide decay), mass cross-modal (impulse/velocity coupling) and
    stiffness almost purely tactile (sub-step contact peaks).

    Making all three *individually* identifiable requires observing
    contact force, not just motion: from motion alone the dynamics depend
    only on ``k/m`` and ``c/m``. The state therefore carries a fifth
    channel, ``touch`` -- the PEAK contact-force magnitude over the
    sub-steps of the transition that led into this state. With touch
    observed, ``k`` follows from the contact peak, ``m`` from the ratio of
    contact impulse to velocity change, and ``c`` from decay during
    free glide. Rendering is unchanged and identical across theta: the
    physics is hidden from the pixels, exposed only through dynamics.
    """

    theta_names = (
        'mass',
        'contact_stiffness',
        'drag',
        'poker_radius',
        'object_radius',
    )

    def __init__(
        self,
        dt: float = 0.001,
        substeps: int = 20,
        world_size: float = 1.0,
        drag_exponent: float = 1.0,
        mass_range=(0.5, 3.0),
        stiffness_range=(500.0, 6000.0),
        drag_range=(0.5, 4.0),
        poker_radius: float = 0.10,
        object_radius: float = 0.10,
    ):
        self.dt = dt
        self.substeps = substeps
        self.world_size = world_size
        self.drag_exponent = drag_exponent
        self.mass_range = mass_range
        self.stiffness_range = stiffness_range
        self.drag_range = drag_range
        self.poker_radius = poker_radius
        self.object_radius = object_radius

    def sample_theta(self, rng) -> np.ndarray:
        lo_m, hi_m = self.mass_range
        lo_k, hi_k = self.stiffness_range
        lo_c, hi_c = self.drag_range
        return np.array(
            [
                rng.uniform(lo_m, hi_m),
                rng.uniform(lo_k, hi_k),
                rng.uniform(lo_c, hi_c),
                self.poker_radius,
                self.object_radius,
            ],
            dtype=np.float32,
        )

    def rollout(self, rng, length: int):
        """One episode. Returns ``(states, actions, theta)``.

        State is ``[x, y, vx, vy, touch]``. ``touch`` is
        ``log1p(peak contact force)`` over the sub-steps of the transition
        that produced this state, so ``states[0, 4]`` is 0 by construction
        (no preceding transition). The peak -- not an end-of-step sample --
        is what makes stiffness observable: with stiff contacts the force
        spikes and decays well inside one transition.

        The log compression is not cosmetic. Raw peak force is
        zero-inflated (contact fires in ~5% of transitions) with a range of
        0-700, so after z-scoring the silent majority collapses onto one
        value and the contact events become +20 sigma outliers. Squared
        error on that target is dominated by rare spikes and teaches
        nothing in between. ``log1p`` puts the channel in 0-6.6 and makes
        the contact events ordinary-sized, which is also how real tactile
        transduction behaves (compressive, Weber-Fechner-like).
        """
        theta = self.sample_theta(rng)
        m, k, c = theta[0], theta[1], theta[2]
        contact = self.poker_radius + self.object_radius

        pos = rng.uniform(-0.2, 0.2, size=2).astype(np.float64)
        vel = np.zeros(2)
        # the poker performs a smooth random walk so contact is frequent
        poker = pos + rng.uniform(-contact, contact, size=2)

        states = np.zeros((length, 5), dtype=np.float32)
        actions = np.zeros((length, 2), dtype=np.float32)
        drift = rng.normal(0, 1, size=2)
        touch = 0.0

        for t in range(length):
            states[t] = np.concatenate([pos, vel, [np.log1p(touch)]])
            drift = 0.85 * drift + 0.15 * rng.normal(0, 1, size=2)
            poker = np.clip(
                poker + 0.03 * drift, -self.world_size, self.world_size
            )
            actions[t] = poker

            touch = 0.0
            for _ in range(self.substeps):
                delta = pos - poker
                dist = max(np.linalg.norm(delta), 1e-6)
                overlap = max(contact - dist, 0.0)
                f_contact = k * overlap * (delta / dist)
                touch = max(touch, float(np.linalg.norm(f_contact)))
                speed = np.linalg.norm(vel)
                f = f_contact - c * (speed ** (self.drag_exponent - 1.0)) * vel
                vel = vel + self.dt * f / m
                pos = pos + self.dt * vel
                # elastic walls at the edge of the rendered field. Without
                # them a stiff contact launches the object off-screen and
                # the observation carries no information about the state at
                # all -- at k up to 6000 that happened in ~20% of frames.
                over = pos - np.clip(pos, -self.world_size, self.world_size)
                pos = pos - 2.0 * over
                vel = np.where(over != 0.0, -vel, vel)

        return states, actions, theta

    def render(self, states, actions, size: int = 64):
        """Render an episode to ``(T, 3, size, size)`` float32 in [0, 1].

        Deliberately simple: two soft discs on a blank field. The encoder
        only needs enough signal to localize object and poker.
        """
        T = states.shape[0]
        ax = np.linspace(-self.world_size, self.world_size, size)
        gy, gx = np.meshgrid(ax, ax, indexing='ij')
        img = np.zeros((T, 3, size, size), dtype=np.float32)

        def blob(cx, cy, radius):
            d2 = (gx - cx) ** 2 + (gy - cy) ** 2
            return np.exp(-d2 / (2 * (radius / 2) ** 2))

        for t in range(T):
            img[t, 0] = blob(states[t, 0], states[t, 1], self.object_radius)
            img[t, 2] = blob(actions[t, 0], actions[t, 1], self.poker_radius)
        return np.clip(img, 0.0, 1.0)


def pokeworld_episodes(
    num_episodes: int,
    length: int,
    seed: int = 0,
    render: bool = True,
    image_size: int = 64,
    **sim_kwargs,
):
    """Generate PokeWorld episodes with ground-truth theta."""
    sim = PokeWorldSim(**sim_kwargs)
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(num_episodes):
        states, actions, theta = sim.rollout(rng, length)
        ep = {
            'state': states,
            'action': actions,
            'theta_true': theta,
            # the tactile reading is also an OBSERVATION, so it is exposed
            # under its own key. The model must never index the `state`
            # tensor for inputs -- that array is the supervision target,
            # and reading it there is how target leakage starts.
            'touch': states[:, 4].copy(),
        }
        if render:
            ep['pixels'] = sim.render(states, actions, image_size)
        episodes.append(ep)
    return episodes, sim


# ----------------------------------------------------------------------
# Rollouts from registered swm gym environments
# ----------------------------------------------------------------------


def _cartpole_state(env):
    """dm_control cartpole state as ``[x, angle, x_dot, angle_dot]``.

    The wrapper exposes ``qpos``/``qvel`` in ``info``; dm_control's
    cartpole has ``qpos = [slider, hinge]`` and ``qvel`` likewise, which
    is exactly :class:`CartpoleSolver`'s state layout.
    """
    info = env.unwrapped.info
    return np.concatenate([info['qpos'], info['qvel']]).astype(np.float32)


def pusht_contact_policy(noise: float = 0.25, seed: int = 0):
    """Scripted PushT policy that actually pushes the block.

    Uniform random actions almost never bring the agent into contact, so
    the block stays still and the contact half of the physics is never
    exercised -- the data carries no signal about it. This policy steers
    the agent at the block (PushT actions are relative and scaled by 100,
    so ``a = (block - agent) / 100`` targets the block directly) with
    exploration noise on top.
    """
    rng = np.random.default_rng(seed)

    def policy(state):
        agent, block = state[0:2], state[2:4]
        a = (block - agent) / 100.0
        a = a + rng.normal(0, noise, size=2)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    return policy


def _resize(frame, size):
    """Resize an HxWx3 uint8/float frame to ``size`` x ``size``."""
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    import cv2

    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def pusht_randomized_episodes(
    num_episodes: int,
    length: int,
    seed: int = 0,
    frameskip: int = 1,
    image_size: int = 64,
    render: bool = True,
    kp_range=(40.0, 200.0),
    kv_range=(8.0, 40.0),
    friction_range=(0.2, 2.0),
    mass_range=(0.4, 3.0),
):
    """PushT with per-episode physics, giving it a ground-truth theta.

    Stock PushT fixes its dynamics, so every episode shares one theta and
    the shuffled-theta control is vacuous: permuting a constant changes
    nothing. Randomizing per episode makes identifiability measurable.

    ``k_p`` and ``k_v`` are recorded as ground truth because
    :class:`PushTSolver` reproduces the environment's PD agent law
    exactly, so their true values are known. Block friction and mass are
    also randomized -- they change the dynamics, which is what makes the
    shuffled control informative -- but they enter the solver only
    through effective quasi-static mobilities with no 1:1 counterpart, so
    no ground truth is claimed for them and those theta entries are left
    at their nominal constants (the R^2 helper drops constant columns).
    """
    import gymnasium as gym

    import stable_worldmodel  # noqa: F401  (registers the swm/* ids)

    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    rng = np.random.default_rng(seed)
    env = gym.make('swm/PushT-v1', resolution=image_size)
    policy = pusht_contact_policy(seed=seed)
    episodes = []

    for ep_i in range(num_episodes):
        obs, _ = env.reset(seed=int(seed + ep_i))
        u = env.unwrapped
        kp = float(rng.uniform(*kp_range))
        kv = float(rng.uniform(*kv_range))
        friction = float(rng.uniform(*friction_range))
        mass = float(rng.uniform(*mass_range))
        u.k_p, u.k_v = kp, kv
        for body in getattr(u.space, 'bodies', []):
            if body is getattr(u, 'agent', None):
                continue
            try:
                body.mass = mass
            except (AttributeError, AssertionError):
                pass
        for shape in getattr(u.space, 'shapes', []):
            try:
                shape.friction = friction
            except AttributeError:
                pass

        states, actions, frames = [], [], []
        for _ in range(length):
            st = np.asarray(obs['state'], dtype=np.float32)
            a = np.asarray(policy(st), dtype=np.float32).reshape(-1)
            states.append(st)
            actions.append(a)
            if render:
                frame = _resize(np.asarray(env.render()), image_size)
                frames.append(np.asarray(frame, dtype=np.float32) / 255.0)
            for _ in range(frameskip):
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    obs, _ = env.reset(seed=int(seed + ep_i))
                    break
        state_arr = np.stack(states)
        # PushT reports `block.angle % 2*pi`; unwrap so the angle channel
        # is the continuous one the solver integrates
        state_arr[:, 4] = np.unwrap(state_arr[:, 4])
        ep = {
            'state': state_arr.astype(np.float32),
            'action': np.stack(actions),
            # order matches PushTSolver.theta_names; only kp/kv vary
            'theta_true': np.array(
                [kp, kv, 15.0, 30.0, 400.0, 1.0, 1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            'physics': {'k_p': kp, 'k_v': kv,
                        'friction': friction, 'mass': mass},
        }
        if render and frames:
            ep['pixels'] = np.stack(frames).transpose(0, 3, 1, 2)
        episodes.append(ep)
    env.close()
    return episodes, None


def fetch_expert_policy(noise: float = 0.15, seed: int = 0):
    """Scripted Fetch-push policy that actually drives contact.

    Mirrors :func:`pusht_contact_policy`'s minimalism: uniform-random
    actions almost never bring the gripper into contact with the object,
    so the object stays still and the mass/friction physics are never
    exercised. This steers the gripper at the object (planar, gripper
    held closed throughout -- Push/Slide never need it open) with
    exploration noise on top.

    Uses Fetch's standard 25-d ``observation`` layout: ``grip_pos =
    obs[0:3]``, ``object_pos = obs[3:6]`` (stable across
    gymnasium-robotics Push/Slide/PickAndPlace).
    """
    rng = np.random.default_rng(seed)

    def policy(obs):
        grip = obs[0:3]
        obj = obs[3:6]
        a_xyz = (obj - grip) * 5.0
        a_xyz = a_xyz + rng.normal(0, noise, size=3)
        a_xyz[2] = 0.0  # stay planar: push/slide keep the gripper height fixed
        a = np.concatenate([a_xyz, [-1.0]])  # gripper held closed
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    return policy


_FETCH_CACHE_DIR = Path(
    os.environ.get('SWM_FETCH_CACHE_DIR', '/workspace/physwm-artifacts/fetch_episode_cache')
)


def _fetch_cache_key(num_episodes, length, seed, frameskip, image_size, render, env_id):
    return (
        f'{env_id.replace("/", "_")}_ep{num_episodes}_len{length}_seed{seed}'
        f'_fs{frameskip}_img{image_size}_render{int(render)}.pkl'
    )


def fetch_episodes(
    num_episodes: int,
    length: int,
    seed: int = 0,
    frameskip: int = 1,
    image_size: int = 64,
    render: bool = True,
    env_id: str = 'swm/FetchPush-v3',
):
    """Fetch Push/Slide episodes with per-episode ground-truth (mass, friction).

    Ground truth comes from :class:`FetchWrapper`'s own
    ``info['theta_true']`` (realized MuJoCo ``body_mass`` /
    ``geom_friction`` after each reset, not just the sampled target) --
    unlike :func:`pusht_randomized_episodes`, which has to reach into
    pymunk bodies/shapes directly because PushT has no such plumbing.

    State is the reduced 6-d planar projection
    :class:`~stable_worldmodel.wm.physwm.solvers.FetchPushSolver` expects:
    ``[gripper_x, gripper_y, object_x, object_y, object_vx, object_vy]``,
    read out of Fetch's standard 25-d ``observation`` layout.

    Collection is CPU-bound (MuJoCo rollout + software EGL rendering when
    no GPU render device is available -- common on shared pods) and can
    take much longer than the actual training that follows. Results are
    cached to disk keyed by every parameter that affects the data, so
    re-running with the same (seed, num_episodes, length, ...) -- as
    happens whenever several conditions share a seed/episode budget --
    pays the collection cost once, not once per condition.
    """
    import gymnasium as gym
    import pickle

    import stable_worldmodel  # noqa: F401  (registers the swm/* ids)

    cache_key = _fetch_cache_key(
        num_episodes, length, seed, frameskip, image_size, render, env_id
    )
    cache_path = _FETCH_CACHE_DIR / cache_key
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f), None

    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    render_mode = 'rgb_array' if render else None
    env = gym.make(env_id, render_mode=render_mode, resolution=image_size)
    policy = fetch_expert_policy(seed=seed)
    episodes = []

    for ep_i in range(num_episodes):
        _, info = env.reset(
            seed=int(seed + ep_i),
            options={'variation': ['block.mass', 'block.friction']},
        )
        theta = info.get('theta_true') or {'mass': float('nan'), 'friction': float('nan')}
        goal = np.asarray(info.get('goal_state'), dtype=np.float32)

        states, actions, frames = [], [], []
        for _ in range(length):
            raw = np.asarray(info['proprio'], dtype=np.float32)
            grip_xy = raw[0:2]
            obj_xy = raw[3:5]
            obj_vxy = raw[14:16]
            st = np.concatenate([grip_xy, obj_xy, obj_vxy]).astype(np.float32)
            a = policy(raw)
            states.append(st)
            actions.append(a)
            if render:
                frame = _resize(np.asarray(env.render()), image_size)
                frames.append(np.asarray(frame, dtype=np.float32) / 255.0)
            for _ in range(frameskip):
                _, _, term, trunc, info = env.step(a)
                if term or trunc:
                    _, info = env.reset(
                        seed=int(seed + ep_i),
                        options={'variation': ['block.mass', 'block.friction']},
                    )
                    theta = info.get('theta_true') or theta
                    break
        ep = {
            'state': np.stack(states).astype(np.float32),
            'action': np.stack(actions),
            # order matches FetchPushSolver.theta_names
            'theta_true': np.array(
                [theta['mass'], theta['friction']], dtype=np.float32
            ),
            # planar (x, y) goal position; z is dropped to match the
            # solver's 2-d object state, same convention as the state itself
            'goal': goal[:2] if goal.size >= 2 else goal,
        }
        if render and frames:
            ep['pixels'] = np.stack(frames).transpose(0, 3, 1, 2)
        episodes.append(ep)
    env.close()

    _FETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(episodes, f)
    tmp_path.replace(cache_path)  # atomic: no other process sees a partial file

    return episodes, None


def _cartpole_episodes_subprocess(
    num_episodes, length, seed, frameskip, image_size, render,
    batch_size=8, max_attempts=None, batch_timeout=900,
):
    """Collect cartpole episodes via repeated calls to
    ``_cartpole_batch_worker.py``, isolating the native-SIGABRT risk (see
    that script's docstring) in a throwaway subprocess per batch.

    Each attempt asks for up to ``batch_size`` new episodes at a fresh,
    never-before-tried seed range; whatever the worker managed to flush to
    disk before dying (if it died) is kept, and only the shortfall is
    retried. Seeds always advance rather than repeat, since a crash is a
    property of the random action sequence a seed draws, not something a
    bare retry at the same seed would avoid.
    """
    import pickle
    import shutil
    import subprocess
    import sys
    import tempfile

    worker = str(
        Path(__file__).resolve().parents[3]
        / 'scripts' / 'eval' / '_cartpole_batch_worker.py'
    )
    if max_attempts is None:
        max_attempts = max(20, (num_episodes // batch_size + 1) * 4)

    tmp_root = Path(tempfile.mkdtemp(prefix='cartpole_eps_'))
    collected = []
    next_seed = seed
    attempt = 0
    try:
        while len(collected) < num_episodes and attempt < max_attempts:
            attempt += 1
            want = min(batch_size, num_episodes - len(collected))
            out_dir = tmp_root / f'batch_{attempt}'
            cmd = [
                sys.executable, worker,
                '--start-seed', str(next_seed), '--count', str(want),
                '--length', str(length), '--frameskip', str(frameskip),
                '--image-size', str(image_size), '--out-dir', str(out_dir),
            ]
            if render:
                cmd.append('--render')
            try:
                subprocess.run(cmd, timeout=batch_timeout, capture_output=True)
            except subprocess.TimeoutExpired:
                pass
            for p in sorted(out_dir.glob('ep_*.pkl')) if out_dir.exists() else []:
                with open(p, 'rb') as f:
                    collected.append(pickle.load(f))
            next_seed += want * 1000  # clear of every seed just tried, hit or miss

        if len(collected) < num_episodes:
            print(
                f"[env_episodes] cartpole: only collected "
                f"{len(collected)}/{num_episodes} episodes after {attempt} "
                f"subprocess attempts (worker kept crashing); proceeding "
                f"with what we have"
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return collected[:num_episodes]


def env_episodes(
    env_id: str,
    num_episodes: int,
    length: int,
    seed: int = 0,
    frameskip: int = 1,
    image_size: int = 64,
    render: bool = True,
    policy=None,
):
    """Roll out a registered ``swm/*`` environment.

    ``frameskip`` holds each action for that many ``env.step`` calls,
    matching the dataset convention used elsewhere in the repo. The
    solver's ``dt * substeps`` must cover the same wall-clock interval as
    one transition, so changing ``frameskip`` means changing
    ``solver.substeps`` too.

    dm_control envs need a GL backend for rendering (``MUJOCO_GL=egl``);
    PushT needs a headless SDL driver, which is set below.
    """
    is_cartpole = 'Cartpole' in env_id
    is_pusht = 'PushT' in env_id

    if is_cartpole:
        # A single unstable action sequence anywhere in the rollout can
        # trip a native SIGABRT deep inside mj_step (see
        # _cartpole_batch_worker.py's docstring) -- not a Python
        # exception, so it cannot be caught here. Every cartpole episode
        # is therefore collected in a disposable subprocess instead of
        # in-process: a crash there costs one retried batch, not this
        # whole run.
        return _cartpole_episodes_subprocess(
            num_episodes, length, seed, frameskip, image_size, render
        )

    import gymnasium as gym

    import stable_worldmodel  # noqa: F401  (registers the swm/* ids)

    if is_pusht:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

    if policy is None and is_pusht:
        # random actions never make contact -> no block physics in the
        # data; drive at the block instead
        policy = pusht_contact_policy(seed=seed)

    make_kwargs = {'resolution': image_size} if is_pusht else {}
    env = gym.make(env_id, **make_kwargs)
    episodes = []

    for ep_i in range(num_episodes):
        obs, _ = env.reset(seed=int(seed + ep_i))
        states, actions, frames = [], [], []
        for _ in range(length):
            s = np.asarray(obs['state'], dtype=np.float32)
            if policy is not None:
                a = policy(s)
            else:
                a = env.action_space.sample()
            a = np.asarray(a, dtype=np.float32).reshape(-1)

            states.append(s)
            actions.append(a)
            if render:
                frame = env.render()
                frame = _resize(np.asarray(frame), image_size)
                frames.append(np.asarray(frame, dtype=np.float32) / 255.0)

            for _ in range(frameskip):
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    obs, _ = env.reset(seed=int(seed + ep_i))
                    break

        state_arr = np.stack(states)
        if is_pusht:
            # PushT reports `block.angle % 2*pi`, so the angle channel has
            # 2*pi jumps that no smooth solver (and no regression head) can
            # predict -- they would dominate the angular error. Unwrap it
            # per episode to recover the continuous angle the physics
            # actually follows.
            state_arr[:, 4] = np.unwrap(state_arr[:, 4])

        ep = {
            'state': state_arr.astype(np.float32),
            'action': np.stack(actions),
        }
        if render:
            ep['pixels'] = np.stack(frames).transpose(0, 3, 1, 2)
        episodes.append(ep)
    env.close()
    return episodes


def swm_dataset_episodes(name: str, keys=('state', 'action'), **kwargs):
    """Adapter over ``swm.data.load_dataset`` for pre-collected datasets.

    Used for full-scale runs on the repo's Lance/HDF5 datasets (e.g.
    ``pusht_expert_train.lance``); the synthetic sources above are what
    the smoke tests exercise.
    """
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(name, **kwargs)
    episodes = []
    for i in range(len(dataset)):
        item = dataset[i]
        ep = {
            k: np.asarray(item[k], dtype=np.float32) for k in keys if k in item
        }
        if 'pixels' in item:
            ep['pixels'] = np.asarray(item['pixels'], dtype=np.float32)
        episodes.append(ep)
    return episodes


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------


class EpisodeWindowDataset(Dataset):
    """Slice episodes into fixed-length windows.

    A window is the unit the model consumes: ``T`` frames giving ``T - 1``
    scored transitions. Windows never straddle an episode boundary, which
    matters because per-episode theta must be constant within a window.
    """

    def __init__(
        self,
        episodes,
        window: int,
        stride: int = 1,
        context: int | None = None,
        require_context_touch: bool = False,
    ):
        assert window >= 2, 'window must cover at least one transition'
        if require_context_touch:
            assert context is not None and 1 <= context < window - 1
        self.episodes = episodes
        self.window = window
        self.index = []
        for ep_i, ep in enumerate(episodes):
            n = ep['state'].shape[0]
            for start in range(0, n - window + 1, stride):
                if require_context_touch:
                    assert 'touch' in ep, (
                        'require_context_touch needs a touch observation'
                    )
                    # touch[t] summarizes transition t-1 -> t. Context K
                    # owns touch indices 1..K; K+1 is the first query
                    # outcome and must not be used for window selection.
                    touch = ep['touch'][start + 1:start + context + 1]
                    if not np.any(touch > 0):
                        continue
                self.index.append((ep_i, start))
        assert self.index, f'no windows: episodes shorter than window={window}'

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep_i, start = self.index[i]
        ep = self.episodes[ep_i]
        sl = slice(start, start + self.window)
        item = {
            'state': torch.from_numpy(ep['state'][sl]).float(),
            'action': torch.from_numpy(ep['action'][sl]).float(),
        }
        if 'pixels' in ep:
            item['pixels'] = torch.from_numpy(ep['pixels'][sl]).float()
        if 'theta_true' in ep:
            item['theta_true'] = torch.from_numpy(ep['theta_true']).float()
        if 'touch' in ep:
            item['touch'] = torch.from_numpy(ep['touch'][sl]).float()
        if 'goal' in ep:
            item['goal'] = torch.from_numpy(ep['goal']).float()
        item['episode'] = torch.tensor(ep_i)
        return item


def collect_stats(dataset, max_items: int = 512, seed: int = 0):
    """Gather ``(state, action)`` tensors for fitting the normalizers.

    The sample is drawn at RANDOM across the whole dataset, not from the
    front. Windows are laid out episode by episode, so taking the first
    ``max_items`` of them reads only the first few episodes -- at window 8
    and stride 1 a 48-step episode yields 41 windows, so 512 windows is
    about 12 episodes. PokeWorld redraws theta every episode over wide
    ranges, so those 12 episodes are a badly biased sample of the state
    distribution, and the bias grows as episodes are added (512 windows of
    a 1024-episode set is the first 1.2% of it).

    A mis-fitted normalizer is not cosmetic: every loss term is computed
    in normalized space, so it silently reweights the state dimensions
    against each other and distorts what the model is optimizing.
    """
    n = min(len(dataset), max_items)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    states = torch.stack([dataset[i]['state'] for i in idx])
    actions = torch.stack([dataset[i]['action'] for i in idx])
    return states, actions


__all__ = [
    'EpisodeWindowDataset',
    'PokeWorldSim',
    'collect_stats',
    'env_episodes',
    'fetch_episodes',
    'pokeworld_episodes',
    'pusht_randomized_episodes',
    'pusht_contact_policy',
    'swm_dataset_episodes',
]
