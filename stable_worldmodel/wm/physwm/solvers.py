"""Frozen, differentiable physics solvers -- Path B of :class:`PhysWM`.

Every solver here is a *hardcoded* integrator. It owns **zero learnable
parameters** and is never touched by the optimizer. Its only job:

    (state_t, action_t, theta) -> state_{t+1}

where ``theta`` is the physics-parameter vector produced by a forward-pass
probe on the world-model latent. Gradients flow *through* the integrator
back into the probe -- that is the whole point of Path B.

Contract enforced by :class:`PhysicsSolver` (see :meth:`assert_frozen`):

* ``len(list(solver.parameters())) == 0`` -- zero learnable params.
* ``theta`` is always a forward **argument**, never module state. A solver
  therefore *cannot* be used to optimize theta as a free per-step variable;
  theta only ever exists as the output of a probe forward pass.
* Constants (integration step, parameter bounds) live in buffers, which
  ``.parameters()`` does not report and the optimizer never sees.

Parameter bounds
----------------
A probe emits an unconstrained ``raw`` vector. ``bound_theta`` squashes it
into the solver's physical range via ``lo + (hi - lo) * sigmoid(raw)``. This
keeps theta physically meaningful (positive masses, sane gravity) and keeps
gradients bounded. ``raw_for`` is the inverse, used to bias-initialize a
probe at a nominal parameter set.
"""

import torch
from torch import nn


def _expand_theta(theta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast ``theta`` to ``ref``'s leading (batch, time) shape.

    Accepts per-episode theta ``(B, K)`` or per-step theta ``(B, T, K)``
    and returns ``(B, T, K)`` so downstream math is shape-uniform.
    """
    lead = ref.shape[:-1]
    if theta.dim() == ref.dim() - 1:
        # per-episode: (B, K) -> (B, T, K)
        theta = theta.unsqueeze(1)
    if theta.shape[:-1] != lead:
        theta = theta.expand(*lead, theta.shape[-1])
    return theta


class PhysicsSolver(nn.Module):
    """Base class for frozen differentiable integrators.

    Subclasses declare ``state_names``/``action_dim``/``theta_names`` and
    implement :meth:`derivative` (continuous-time) or override
    :meth:`step` (custom discrete update).
    """

    state_names: tuple[str, ...] = ()
    theta_names: tuple[str, ...] = ()
    action_dim: int = 0

    def __init__(self, dt: float = 0.01, substeps: int = 10, **kwargs):
        super().__init__()
        assert substeps >= 1, 'substeps must be >= 1'
        self.dt = float(dt)
        self.substeps = int(substeps)
        # bounds live in buffers -> never reported by .parameters()
        lo, hi, nominal = self.default_bounds()
        self.register_buffer('theta_lo', torch.tensor(lo))
        self.register_buffer('theta_hi', torch.tensor(hi))
        self.register_buffer('theta_nominal', torch.tensor(nominal))
        self.assert_frozen()

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def theta_dim(self) -> int:
        return len(self.theta_names)

    def default_bounds(self):
        """Return ``(lo, hi, nominal)`` lists, one entry per theta name."""
        raise NotImplementedError

    def assert_frozen(self) -> None:
        """Hard guarantee: a solver has no learnable parameters."""
        n = sum(p.numel() for p in self.parameters())
        assert n == 0, (
            f'{type(self).__name__} must have zero learnable parameters, '
            f'found {n}. Physics constants belong in buffers.'
        )

    # ------------------------------------------------------------------
    # theta (de)parameterization
    # ------------------------------------------------------------------

    def bound_theta(self, raw: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained probe output into the physical range."""
        assert raw.shape[-1] == self.theta_dim, (
            f'expected last dim {self.theta_dim} '
            f'({", ".join(self.theta_names)}), got {raw.shape[-1]}'
        )
        lo = self.theta_lo.to(raw.dtype)
        hi = self.theta_hi.to(raw.dtype)
        return lo + (hi - lo) * torch.sigmoid(raw)

    def raw_for(self, theta: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`bound_theta` (used for bias init)."""
        lo, hi = self.theta_lo, self.theta_hi
        u = ((theta - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        return torch.log(u) - torch.log1p(-u)

    def nominal_raw(self) -> torch.Tensor:
        """``raw`` vector whose bounded image is the nominal parameters."""
        return self.raw_for(self.theta_nominal)

    def unpack(self, theta: torch.Tensor) -> dict:
        """Split a bounded theta tensor into a name -> tensor dict."""
        parts = theta.split(1, dim=-1)
        return dict(zip(self.theta_names, (p.squeeze(-1) for p in parts)))

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------

    def derivative(self, state, action, p):
        """Continuous-time state derivative. ``p`` is the unpacked dict."""
        raise NotImplementedError

    def step(self, state, action, p, dt):
        """One integrator substep. Default: explicit Euler on
        :meth:`derivative`. Subclasses with a semi-implicit or
        position-based update override this."""
        return state + dt * self.derivative(state, action, p)

    def forward(self, state, action, theta):
        """Integrate one *transition* (``substeps`` x ``dt``).

        Args:
            state: ``(B, T, state_dim)`` current states.
            action: ``(B, T, action_dim)`` actions held over the transition.
            theta: ``(B, K)`` per-episode or ``(B, T, K)`` per-step,
                already bounded by :meth:`bound_theta`.

        Returns:
            ``(B, T, state_dim)`` next-state predictions.
        """
        assert state.shape[-1] == self.state_dim, (
            f'state last dim must be {self.state_dim}, got {state.shape[-1]}'
        )
        theta = _expand_theta(theta, state)
        p = self.unpack(theta)
        dt = self.dt
        for _ in range(self.substeps):
            state = self.step(state, action, p, dt)
        return state


class CartpoleSolver(PhysicsSolver):
    """Classic cart-pole dynamics (dm_control ``cartpole`` benchmark).

    State ``[x, angle, x_dot, angle_dot]`` with ``angle`` measured from
    upright, matching dm_control's hinge convention (``qpos = [x, angle]``,
    ``qvel = [x_dot, angle_dot]``).

    The pole is a uniform rod of half-length ``l``; the ``4/3`` term is its
    moment of inertia about the hinge. Integration is semi-implicit Euler,
    which is stable for this system and differentiable everywhere.
    """

    state_names = ('x', 'angle', 'x_dot', 'angle_dot')
    action_dim = 1
    theta_names = (
        'cart_mass',
        'pole_mass',
        'pole_half_length',
        'gravity',
        'force_gain',
        'cart_damping',
    )

    def default_bounds(self):
        #        m_c  m_p   l     g     gain  damp
        lo = [0.10, 0.01, 0.05, 1.00, 1.00, 0.00]
        hi = [5.00, 2.00, 1.50, 20.0, 50.0, 5.00]
        nom = [1.00, 0.10, 0.50, 9.81, 10.0, 0.10]
        return lo, hi, nom

    def step(self, state, action, p, dt):
        x, th, xd, thd = state.unbind(-1)
        force = p['force_gain'] * action[..., 0]

        m_c, m_p, ell = p['cart_mass'], p['pole_mass'], p['pole_half_length']
        total = m_c + m_p
        cos, sin = torch.cos(th), torch.sin(th)

        temp = (
            force + m_p * ell * thd.pow(2) * sin - p['cart_damping'] * xd
        ) / total
        # denominator stays positive: m_p/total < 1 and cos^2 <= 1 => >= 1/3
        denom = ell * (4.0 / 3.0 - m_p * cos.pow(2) / total)
        thdd = (p['gravity'] * sin - cos * temp) / denom
        xdd = temp - m_p * ell * thdd * cos / total

        # semi-implicit (symplectic) Euler: velocities first, then positions
        xd = xd + dt * xdd
        thd = thd + dt * thdd
        x = x + dt * xd
        th = th + dt * thd
        return torch.stack([x, th, xd, thd], dim=-1)


class PokeWorldSolver(PhysicsSolver):
    """Poked point-object with contact stiffness and linear drag.

    Mirrors the ``latent-world-model-identifiability`` PokeWorld setting:
    a poker (commanded by the action) pushes a disc object that is subject
    to viscous drag.

        m * v_dot = k_contact * penetration * n_hat - c_drag * v

    State ``[x, y, vx, vy, touch]``; action ``[poker_x, poker_y]``.
    ``touch`` is ``log1p(peak contact force)`` over the sub-steps of the
    transition, matching :class:`PokeWorldSim`.

    Identifiability: from motion alone the dynamics depend only on
    ``k/m`` and ``c/m``, so mass, stiffness and drag would be recoverable
    only up to a 2-d manifold. The ``touch`` channel breaks that
    degeneracy the way the reference PokeWorld does -- the contact peak
    pins ``k``, the contact impulse against the resulting velocity change
    pins ``m``, and decay during free glide pins ``c``. The radii remain
    constant by construction, so their R^2 is undefined (reported `nan`).
    """

    state_names = ('x', 'y', 'vx', 'vy', 'touch')
    action_dim = 2
    #: arena half-width; must match ``PokeWorldSim.world_size``
    world_size = 1.0
    theta_names = (
        'mass',
        'contact_stiffness',
        'drag',
        'poker_radius',
        'object_radius',
    )

    def default_bounds(self):
        # brackets the reference sampling ranges (m 0.5-3, k 500-6000,
        # c 0.5-4) with headroom, so a bounded probe can reach them
        #      m     k        c      r_p   r_o
        lo = [0.10, 100.0, 0.00, 0.02, 0.02]
        hi = [6.00, 8000.0, 10.00, 0.50, 0.50]
        nom = [1.50, 1500.0, 2.00, 0.10, 0.10]
        return lo, hi, nom

    def _contact_force(self, pos, poker, p):
        delta = pos - poker
        dist = delta.norm(dim=-1).clamp_min(1e-6)
        # soft, one-sided contact: zero force outside the contact radius
        overlap = torch.relu(p['poker_radius'] + p['object_radius'] - dist)
        normal = delta / dist.unsqueeze(-1)
        return (
            p['contact_stiffness'].unsqueeze(-1)
            * overlap.unsqueeze(-1)
            * normal
        )

    def step(self, state, action, p, dt):
        pos, vel, touch = state[..., :2], state[..., 2:4], state[..., 4]
        poker = action[..., :2]

        f_contact = self._contact_force(pos, poker, p)
        f_drag = -p['drag'].unsqueeze(-1) * vel
        acc = (f_contact + f_drag) / p['mass'].unsqueeze(-1)

        vel = vel + dt * acc
        pos = pos + dt * vel
        # elastic walls, matching PokeWorldSim exactly (same order of
        # operations, same sub-step): reflect position, flip velocity
        w = self.world_size
        over = pos - pos.clamp(-w, w)
        pos = pos - 2.0 * over
        vel = torch.where(over != 0.0, -vel, vel)
        # running peak over the transition; forward() seeds it at zero
        touch = torch.maximum(touch, f_contact.norm(dim=-1))
        return torch.cat([pos, vel, touch.unsqueeze(-1)], dim=-1)

    def forward(self, state, action, theta):
        """Integrate one transition, reporting the PEAK contact force.

        The touch channel is an observable of the transition, not a
        dynamical state, so it is reset to zero before integrating and
        accumulated as a running max across sub-steps. The incoming
        ``state[..., 4]`` (last transition's peak) is deliberately
        ignored -- carrying it forward would make touch depend on history
        it has no physical dependence on.
        """
        state = torch.cat(
            [state[..., :4], torch.zeros_like(state[..., 4:5])], dim=-1
        )
        out = super().forward(state, action, theta)
        # report log1p(peak force), matching PokeWorldSim: the raw peak is
        # zero-inflated with a huge dynamic range and makes a poor target
        return torch.cat(
            [out[..., :4], torch.log1p(out[..., 4:5])], dim=-1
        )


class PushTSolver(PhysicsSolver):
    """Quasi-static planar pushing, matching the PushT environment.

    State is PushT's own 7-d observation::

        [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]

    Note the last two entries are the **agent's** velocity, not the
    block's (``PushT._get_obs`` returns ``agent.position + block.position
    + block.angle + agent.velocity``; the ``vel_block`` name in the env's
    ``_set_state`` is a misnomer). The block's velocity is therefore an
    unobserved quantity, which is exactly why the block is modelled
    quasi-statically below.

    Action is PushT's relative action in [-1, 1]^2, converted to an
    absolute target by ``target = agent_pos + action * action_scale``.
    Following the environment, the target is computed **once** per
    transition and held fixed across the integration substeps.

    Agent sub-dynamics reproduce the environment exactly -- PushT drives
    the agent with the PD law ``a = k_p (target - x) - k_v v`` and steps
    pymunk semi-implicitly -- so ``k_p`` and ``k_v`` are genuinely
    identifiable from the data.

    Block sub-dynamics are an **approximation**, documented on purpose:

    * The T-shaped block is treated as a disc of effective radius
      ``block_radius``; the true T geometry (three faces, varying moment
      arm) is not captured.
    * Contact is a soft penetration spring, and the resulting wrench maps
      to block motion through an isotropic limit-surface mobility
      (``mobility_lin``, ``mobility_ang``) -- the standard quasi-static
      planar-pushing model, in which the block has no momentum.
    * Rotation is generated by a **body-fixed centre-of-friction offset**
      (``com_offset_*``). This matters: a pure disc pushed along its
      centre line has a lever arm parallel to the contact force, so its
      torque is identically zero and ``block_angle`` can never be
      predicted at all. A T-block's centre of friction is offset from its
      geometric centre, so pushing through the centre still spins it --
      that offset is what makes the angular channel identifiable.

    The benchmark question is whether the probe can still recover a theta
    that explains observed transitions under this structural mismatch.
    """

    state_names = (
        'agent_x',
        'agent_y',
        'block_x',
        'block_y',
        'block_angle',
        'agent_vx',
        'agent_vy',
    )
    action_dim = 2
    theta_names = (
        'agent_kp',
        'agent_kv',
        'agent_radius',
        'block_radius',
        'contact_stiffness',
        'mobility_lin',
        'mobility_ang',
        'com_offset_x',
        'com_offset_y',
    )

    def __init__(
        self,
        dt: float = 0.01,
        substeps: int = 10,
        relative: bool = True,
        action_scale: float = 100.0,
    ):
        # env conventions, not physics to be identified -> constants
        self.relative = bool(relative)
        self.action_scale = float(action_scale)
        super().__init__(dt=dt, substeps=substeps)

    def default_bounds(self):
        #      k_p    k_v    r_a    r_b     k_c    mob_l   mob_a   cx    cy
        lo = [10.0, 1.00, 3.0, 10.0, 0.10, 0.0010, 1e-5, -60.0, -60.0]
        hi = [500.0, 100.0, 50.0, 120.0, 50.0, 1.0000, 1e-2, 60.0, 60.0]
        nom = [100.0, 20.00, 15.0, 40.0, 5.00, 0.0500, 5e-4, 0.0, 15.0]
        return lo, hi, nom

    def forward(self, state, action, theta):
        """Override: the PD target is fixed for the whole transition.

        PushT computes ``target = agent_pos + action * action_scale``
        once per ``env.step`` and then runs ``substeps`` physics steps
        against that fixed target. Recomputing it per substep would be a
        different controller, so we resolve it here and hand ``step`` an
        absolute target.
        """
        theta = _expand_theta(theta, state)
        p = self.unpack(theta)
        if self.relative:
            target = state[..., 0:2] + action[..., :2] * self.action_scale
        else:
            target = action[..., :2]
        for _ in range(self.substeps):
            state = self.step(state, target, p, self.dt)
        return state

    def step(self, state, target, p, dt):
        agent = state[..., 0:2]
        block = state[..., 2:4]
        angle = state[..., 4]
        agent_vel = state[..., 5:7]

        # --- agent: PushT's own PD law, semi-implicit (pymunk order)
        acc = (
            p['agent_kp'].unsqueeze(-1) * (target - agent)
            - p['agent_kv'].unsqueeze(-1) * agent_vel
        )
        agent_vel = agent_vel + dt * acc
        agent = agent + dt * agent_vel

        # --- contact: soft penetration spring between two discs
        delta = block - agent
        dist = delta.norm(dim=-1).clamp_min(1e-6)
        overlap = torch.relu(p['agent_radius'] + p['block_radius'] - dist)
        normal = delta / dist.unsqueeze(-1)
        force = (
            p['contact_stiffness'].unsqueeze(-1)
            * overlap.unsqueeze(-1)
            * normal
        )

        # contact point: on the agent's surface, facing the block
        contact = agent + normal * p['agent_radius'].unsqueeze(-1)

        # centre of friction: body-fixed offset rotated into world frame.
        # Without this offset the lever arm is parallel to the force and
        # the torque vanishes identically (see the class docstring).
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        off_x, off_y = p['com_offset_x'], p['com_offset_y']
        com = block + torch.stack(
            [
                cos_a * off_x - sin_a * off_y,
                sin_a * off_x + cos_a * off_y,
            ],
            dim=-1,
        )

        lever = contact - com
        torque = lever[..., 0] * force[..., 1] - lever[..., 1] * force[..., 0]
        omega = p['mobility_ang'] * torque

        # --- quasi-static limit surface: wrench -> block twist (no inertia)
        # the geometric centre both translates and swings about the COM
        radius = block - com
        swing = torch.stack(
            [-omega * radius[..., 1], omega * radius[..., 0]], dim=-1
        )
        block = block + dt * (p['mobility_lin'].unsqueeze(-1) * force + swing)
        angle = angle + dt * omega

        return torch.cat(
            [agent, block, angle.unsqueeze(-1), agent_vel], dim=-1
        )


SOLVERS = {
    'cartpole': CartpoleSolver,
    'pokeworld': PokeWorldSolver,
    'pusht': PushTSolver,
}


def build_solver(name: str, **kwargs) -> PhysicsSolver:
    """Instantiate a solver by name and freeze it.

    Freezing is belt-and-braces: solvers hold no parameters to begin with
    (asserted in ``__init__``), and ``requires_grad_(False)`` makes the
    intent explicit to anyone reading a checkpoint.
    """
    if name not in SOLVERS:
        raise KeyError(
            f'unknown solver {name!r}; available: {sorted(SOLVERS)}'
        )
    solver = SOLVERS[name](**kwargs)
    solver.requires_grad_(False)
    solver.assert_frozen()
    return solver


__all__ = [
    'SOLVERS',
    'CartpoleSolver',
    'PhysicsSolver',
    'PokeWorldSolver',
    'PushTSolver',
    'build_solver',
]
