"""PhysWM -- a physics-grounded world model with two next-state paths.

Both paths read the **same action-conditioned predictive latent**:

    pixels -> z -> predictor(z, a) -> z_hat
                                      |-- decoder -> s_A
                                      `-- probe -> theta -> solver -> s_B

Path A regresses the dataset's ground truth ``s_next``. Path B regresses
Path A's own (detached) prediction, not ``s_next`` directly: the
experiment is whether a physics parameterization can explain the world
model's *general* next-state belief, not whether the solver can re-derive
the raw benchmark label.

Design rules enforced here (and covered by ``tests/wm/test_physwm.py``):

1. **Two predictions, one action-conditioned latent.** Path A and Path B
   both branch off ``z_hat = predictor(z, a)``. Reading the pre-action
   encoder output is retained only as a named ablation.
2. **A regresses ground truth; B regresses A through a stopped target.**
   ``L_A`` regresses the dataset's ``s_next``. ``L_B`` regresses Path A's
   detached prediction, so fitting the physics path never pulls Path A's
   decoder toward the solver. The optional consistency term is symmetric
   and its weight is 0 by default -- see ``loss.py``.
3. **Theta is a forward-pass probe output.** It is produced by
   ``probe(z_hat)`` on every forward pass, defaults to one vector per episode,
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
        probe: ``z_hat -> raw theta`` (Path B head).
        solver: frozen differentiable integrator.
        obs_key: ``'pixels'`` or ``'state'`` -- what the encoder consumes.
        probe_frames: ``'context'`` pools the probe over the frames the
            solver steps *from* (``z_hat[:, :-1]``); ``'all'`` pools over the
            whole window. ``'context'`` is the default so the probe never
            sees the target frame it is being scored against.
        probe_source: ``'predicted'`` (default) reads the same
            action-conditioned ``z_hat`` that Path A decodes. ``'encoded'``
            reads the pre-action visual code ``z`` and exists only for the
            critical routing ablation.
        probe_context: query start index for strict context/query evaluation.
            With ``K``, the episode probe reads causal predictive latents
            through index ``K`` (which sees observations only through frame
            ``K``), and Path B is scored on transitions ``K:``. ``None``
            retains the legacy all-transition behavior.
        physics_loss_scope: transitions used to train Path B. ``'context'``
            fits persistent physics to completed identification transitions;
            ``'query'`` and ``'all'`` are ablations. Reporting always uses
            the held-out query when ``probe_context`` is set.
        solver_state_source: ``'gt'`` feeds the solver the dataset state
            (isolates theta estimation); ``'decoded'`` feeds it the
            decoder's own read-out of the current latent (fully latent
            grounded, harder).
        tactile_index: index of a state channel to feed the encoder as an
            extra token, or ``None``. PokeWorld uses 4 (``touch``).

            This is what makes stiffness identifiable *from the model's
            input*. The pixels are identical across theta by construction,
            and motion alone only determines ``k/m`` and ``c/m`` -- so a
            probe reading pixels can never resolve mass, stiffness and drag
            individually, no matter how long it trains. A target it cannot
            observe teaches nothing. The tactile channel is a separate
            modality (proprioceptive, not visual), so adding it does not
            leak theta into the rendering.

            Only frame ``t``'s tactile reading enters ``z[t]``, and the
            probe pools ``z_hat[:, :-1]`` under ``probe_frames='context'``, so
            the target frame's touch is never visible to the probe.
        loss_state_indices: state channels scored by the two objectives.
            Defaults to all channels. Visual-only PokeWorld excludes the
            unobserved tactile output while retaining it in the solver state.
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
        probe_source: str = 'predicted',
        probe_context: int | None = None,
        physics_loss_scope: str = 'all',
        solver_state_source: str = 'gt',
        tactile_index: int | None = None,
        tactile_embed_dim: int | None = None,
        loss_state_indices: tuple[int, ...] | list[int] | None = None,
    ):
        super().__init__()
        assert probe_frames in ('context', 'all')
        assert probe_source in ('predicted', 'encoded')
        assert probe_context is None or probe_context >= 1
        assert physics_loss_scope in ('context', 'query', 'all')
        if physics_loss_scope == 'context':
            assert probe_context is not None
        if probe_context is not None:
            assert probe.mode == 'episode', (
                'probe_context requires one persistent episode-level theta'
            )
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
        self.probe_source = probe_source
        self.probe_context = probe_context
        self.physics_loss_scope = physics_loss_scope
        self.solver_state_source = solver_state_source

        indices = (
            tuple(range(state_dim))
            if loss_state_indices is None
            else tuple(loss_state_indices)
        )
        assert indices and len(set(indices)) == len(indices)
        assert all(0 <= i < state_dim for i in indices)
        loss_mask = torch.zeros(state_dim, dtype=torch.bool)
        loss_mask[list(indices)] = True
        # Derived from config; non-persistent keeps older checkpoints loadable.
        self.register_buffer('state_loss_mask', loss_mask, persistent=False)

        self.tactile_index = tactile_index
        if tactile_index is not None:
            assert 0 <= tactile_index < state_dim, (
                f'tactile_index {tactile_index} outside state_dim {state_dim}'
            )
            assert tactile_embed_dim is not None, (
                'tactile_embed_dim is required when tactile_index is set'
            )
            self.tactile_embed = nn.Linear(1, tactile_embed_dim)

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
        z = rearrange(z, '(b t) n d -> b t n d', b=b, t=t)
        if self.tactile_index is not None:
            i = self.tactile_index
            # read the OBSERVATION key, never `batch['state']` -- that is
            # the supervision target and must not be a model input
            touch = batch['touch'].unsqueeze(-1)
            touch = (touch - self.state_norm.mean[i]) / self.state_norm.std[i]
            # one extra token per frame, carrying that frame's reading
            tok = self.tactile_embed(touch).unsqueeze(2)
            z = torch.cat([z, tok], dim=2)
        return z

    def predict_latent(self, z, action):
        """Roll the predictor one step: output ``t`` predicts frame ``t+1``."""
        act_emb = self.action_encoder(self.action_norm.norm(action))
        return self.predictor(z, act_emb)

    def select_probe_latent(self, z, z_hat):
        """Select the latent used by Path B and apply temporal alignment.

        The paper method uses ``predicted``: exactly the action-conditioned
        latent decoded by Path A. ``encoded`` is the pre-action routing used
        by the earlier implementation and is kept as a falsifying ablation.
        """
        source = z_hat if self.probe_source == 'predicted' else z
        if self.probe_context is not None:
            # z_hat[K] is causal: it sees frames <=K and action K, but not
            # the query outcome at K+1. Earlier action-conditioned latents
            # provide additional completed-history features to the pool.
            assert self.probe_context < source.shape[1] - 1, (
                f'probe_context={self.probe_context} leaves no query '
                f'transition in a {source.shape[1]}-frame window'
            )
            return source[:, :self.probe_context + 1]
        return source[:, :-1] if self.probe_frames == 'context' else source

    def physics_time_masks(self, frames, device):
        """Return disjoint training and held-out evaluation masks."""
        all_steps = torch.ones(frames - 1, dtype=torch.bool, device=device)
        if self.probe_context is None:
            return all_steps, all_steps
        context = torch.arange(frames - 1, device=device) < self.probe_context
        query = ~context
        loss = {
            'context': context,
            'query': query,
            'all': all_steps,
        }[self.physics_loss_scope]
        return loss, query

    def probe_latent(self, batch):
        """Return the latent Path B reads for an input batch."""
        z = self.encode(batch)
        z_hat = self.predict_latent(z, batch['action'])
        return self.select_probe_latent(z, z_hat)

    def probe_features(self, batch):
        """Return the exact pooled features consumed by ``ThetaProbe``."""
        return self.probe.features(self.probe_latent(batch))

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
        z_probe = self.select_probe_latent(z, z_hat)
        theta_raw = self.probe(z_probe)
        theta = self.solver.bound_theta(theta_raw)
        # cached by ThetaProbe.forward when probe.quantizer is enabled;
        # None (the default) means "no VQ term to add" for physwm_loss
        vq_loss = getattr(self.probe, 'last_vq_loss', None)

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

        physics_loss_mask, physics_eval_mask = self.physics_time_masks(
            T, state.device
        )
        return {
            'state_a': state_a,  # Path A prediction (normalized)
            'state_b': state_b,  # Path B prediction (normalized)
            'target': target,  # dataset s_next (normalized); Path A's target
            'theta': theta,  # bounded physics params
            'theta_raw': theta_raw,
            'z': z,
            'z_hat': z_hat,
            'z_probe': z_probe,
            'state_a_phys': self.state_norm.unnorm(state_a),
            'state_b_phys': state_b_phys,
            'state_loss_mask': self.state_loss_mask,
            'physics_loss_mask': physics_loss_mask,
            'physics_eval_mask': physics_eval_mask,
            'vq_loss': vq_loss,
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_theta(self, batch):
        """Convenience: return the probe's bounded theta for a batch."""
        return self.solver.bound_theta(self.probe(self.probe_latent(batch)))

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
