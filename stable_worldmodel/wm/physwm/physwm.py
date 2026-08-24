"""PhysWM -- a physics-grounded world model with two next-state paths.

Both paths read the **same** encoder latent:

    Path A (learned):   pixels -> z -> predictor -> z_hat -> decoder -> s_A
    Path B (physical):  pixels -> z -> probe -> theta -> solver(s_t, a_t,
                        theta) -> s_B

Path A regresses the dataset's ground truth ``s_next``. Path B regresses
Path A's own (detached) prediction, not ``s_next`` directly: the
experiment is whether a physics parameterization can explain the world
model's *general* next-state belief, not whether the solver can re-derive
the raw benchmark label.

Design rules enforced here (and covered by ``tests/wm/test_physwm.py``):

1. **Two predictions, one latent.** Path A and Path B both branch off the
   encoder latent ``z``. Neither path is a post-hoc fit to the other.
2. **A regresses ground truth; B regresses A, and A never regresses B.**
   ``L_A`` regresses the dataset's ``s_next``. ``L_B`` regresses Path A's
   detached prediction, so fitting the physics path never pulls Path A's
   decoder toward the solver. The optional consistency term is symmetric
   and its weight is 0 by default -- see ``loss.py``.
3. **Theta is a forward-pass probe output.** It is produced by
   ``probe(z)`` on every forward pass, defaults to one vector per episode,
   and is never an ``nn.Parameter``. ``assert_no_free_theta`` makes this a
   checked invariant, not a convention.
4. **The solver is frozen, differentiable and has zero learnable
   parameters.** Gradients flow through the integrator into the probe --
   that is the only way the probe learns.

Units and normalization
-----------------------
The solver is a *physical* model, so it must see states and actions in
physical units. The networks want normalized inputs, and ``L_A``/``L_B``
must be commensurate. So: the batch carries physical units, the encoder
and action encoder consume normalized values, the solver consumes
physical values, and **all losses are computed in normalized state
space**. The normalizer lives inside the model (buffers) so a checkpoint
is self-contained.
"""

import torch
from einops import rearrange
from torch import nn


class Normalizer(nn.Module):
    """Z-score normalizer with buffers, fitted once from the dataset.

    Lives inside the model so that a checkpoint restores the exact
    normalization used at train time.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> 'Normalizer':
        """Fit from ``(..., dim)`` samples."""
        flat = x.reshape(-1, x.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(self.eps))
        self.fitted.fill_(True)
        return self

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, x):
        return x * self.std + self.mean


def assert_no_free_theta(model: nn.Module) -> None:
    """Invariant check: theta must never be a free optimizable variable.

    Theta is only ever the output of ``probe(z)``. If someone later adds an
    ``nn.Parameter`` named ``theta`` (a per-step free variable fitted
    directly to the targets), Path B stops being a probe of the latent and
    the whole experiment is void. Fail loudly instead.
    """
    for name, _ in model.named_parameters():
        leaf = name.split('.')[-1]
        assert 'theta' not in leaf.lower(), (
            f'parameter {name!r} looks like a free theta variable. Theta '
            'must be a forward-pass probe output, never optimized directly.'
        )


class PhysWM(nn.Module):
    """Physics-grounded world model with a learned and a physical path.

    Args:
        encoder: ``obs -> (B, N, D)`` tokens.
        action_encoder: ``(B, T, A) -> (B, T, A_emb)``.
        predictor: ``(z, act_emb) -> z_hat`` block-causal predictor.
        state_decoder: ``z -> normalized state`` (Path A head).
        probe: ``z -> raw theta`` (Path B head).
        solver: frozen differentiable integrator.
        obs_key: ``'pixels'`` or ``'state'`` -- what the encoder consumes.
        probe_frames: ``'context'`` pools the probe over the frames the
            solver steps *from* (``z[:, :-1]``); ``'all'`` pools over the
            whole window. ``'context'`` is the default so the probe never
            sees the target frame it is being scored against.
        solver_state_source: ``'gt'`` feeds the solver the dataset state
            (isolates theta estimation); ``'decoded'`` feeds it the
            decoder's own read-out of the current latent (fully latent
            grounded, harder).
    """

    def __init__(
        self,
        encoder,
        action_encoder,
        predictor,
        state_decoder,
        probe,
        solver,
        state_dim: int,
        action_dim: int,
        obs_key: str = 'pixels',
        probe_frames: str = 'context',
        solver_state_source: str = 'gt',
    ):
        super().__init__()
        assert probe_frames in ('context', 'all')
        assert solver_state_source in ('gt', 'decoded')
        assert solver.state_dim == state_dim, (
            f'solver expects state_dim {solver.state_dim}, model was given '
            f'{state_dim}'
        )
        assert solver.action_dim == action_dim, (
            f'solver expects action_dim {solver.action_dim}, model was '
            f'given {action_dim}'
        )

        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.state_decoder = state_decoder
        self.probe = probe
        self.solver = solver

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_key = obs_key
        self.probe_frames = probe_frames
        self.solver_state_source = solver_state_source

        self.state_norm = Normalizer(state_dim)
        self.action_norm = Normalizer(action_dim)

        # the solver is frozen and parameter-free; make it explicit
        self.solver.requires_grad_(False)
        self.solver.assert_frozen()
        assert_no_free_theta(self)

    # ------------------------------------------------------------------

    def fit_normalizers(self, state, action) -> None:
        self.state_norm.fit(state)
        self.action_norm.fit(action)

    def encode(self, batch):
        """Encode the observation window into ``(B, T, N, D)`` tokens."""
        obs = batch[self.obs_key]
        b, t = obs.shape[:2]
        flat = rearrange(obs, 'b t ... -> (b t) ...')
        if self.obs_key == 'state':
            flat = self.state_norm.norm(flat)
        z = self.encoder(flat)
        return rearrange(z, '(b t) n d -> b t n d', b=b, t=t)

    def predict_latent(self, z, action):
        """Roll the predictor one step: output ``t`` predicts frame ``t+1``."""
        act_emb = self.action_encoder(self.action_norm.norm(action))
        return self.predictor(z, act_emb)

    # ------------------------------------------------------------------

    def forward(self, batch):
        """Run both paths over a window and return everything for the loss.

        Expects ``batch`` with physical-unit ``state`` ``(B, T, S)``,
        ``action`` ``(B, T, A)`` and the observation under ``obs_key``.

        For a window of ``T`` frames we score the ``T - 1`` transitions
        ``t -> t + 1``. Everything returned is aligned on those
        transitions, and all state tensors are in **normalized** space so
        the two paths are directly comparable.
        """
        state, action = batch['state'], batch['action']
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        T = state.shape[1]
        assert T >= 2, 'need at least 2 frames to score a transition'

        z = self.encode(batch)

        # ---- Path A: learned latent prediction, decoded to state --------
        z_hat = self.predict_latent(z, action)
        # output at t is the prediction of frame t+1 -> drop the last
        state_a = self.state_decoder(z_hat[:, :-1])

        # ---- Path B: probe -> theta -> frozen physics solver ------------
        z_probe = z[:, :-1] if self.probe_frames == 'context' else z
        theta_raw = self.probe(z_probe)
        theta = self.solver.bound_theta(theta_raw)

        if self.solver_state_source == 'gt':
            s_cur = state[:, :-1]
        else:
            s_cur = self.state_norm.unnorm(self.state_decoder(z[:, :-1]))
        state_b_phys = self.solver(s_cur, action[:, :-1], theta)
        state_b = self.state_norm.norm(state_b_phys)

        # ---- dataset ground truth (supervision target for Path A only) --
        # Path B's target is state_a (detached), assembled in loss.py --
        # both paths are returned here and physwm_loss picks the right
        # target for each.
        target = self.state_norm.norm(state[:, 1:])

        return {
            'state_a': state_a,  # Path A prediction (normalized)
            'state_b': state_b,  # Path B prediction (normalized)
            'target': target,  # dataset s_next (normalized); Path A's target
            'theta': theta,  # bounded physics params
            'theta_raw': theta_raw,
            'z': z,
            'z_hat': z_hat,
            'state_a_phys': self.state_norm.unnorm(state_a),
            'state_b_phys': state_b_phys,
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_theta(self, batch):
        """Convenience: return the probe's bounded theta for a batch."""
        z = self.encode(batch)
        z_probe = z[:, :-1] if self.probe_frames == 'context' else z
        return self.solver.bound_theta(self.probe(z_probe))

    def param_report(self) -> dict:
        """Learnable-parameter counts per submodule (logged at startup)."""

        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            'encoder': count(self.encoder),
            'action_encoder': count(self.action_encoder),
            'predictor': count(self.predictor),
            'state_decoder': count(self.state_decoder),
            'probe': count(self.probe),
            'solver': count(self.solver),  # must be 0
            'total': count(self),
        }


__all__ = ['Normalizer', 'PhysWM', 'assert_no_free_theta']
