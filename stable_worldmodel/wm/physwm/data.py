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
    import gymnasium as gym

    import stable_worldmodel  # noqa: F401  (registers the swm/* ids)

    is_cartpole = 'Cartpole' in env_id
    is_pusht = 'PushT' in env_id
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
            s = (
                _cartpole_state(env)
                if is_cartpole
                else np.asarray(obs['state'], dtype=np.float32)
            )
            if policy is not None:
                a = policy(s)
            else:
                a = env.action_space.sample()
            a = np.asarray(a, dtype=np.float32).reshape(-1)

            states.append(s)
            actions.append(a)
            if render:
                frame = (
                    env.unwrapped.render(width=image_size, height=image_size)
                    if is_cartpole
                    else env.render()
                )
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
    'pokeworld_episodes',
    'pusht_contact_policy',
    'swm_dataset_episodes',
]
