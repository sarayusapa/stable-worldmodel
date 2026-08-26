"""Swappable building blocks for :class:`PhysWM`.

Each block is behind a small, explicit interface so it can be replaced from
config without touching the training loop:

* **Encoder** ``pixels -> (B, N, D)`` patch tokens. ``TinyCNNEncoder`` (no
  downloads, for smoke tests), ``DinoV2Encoder`` (the real DINO-WM
  backbone, frozen), ``StateEncoder`` (state-input ablation).
* **Predictor** ``(z, a) -> z_hat`` block-causal spatio-temporal ViT, the
  DINO-WM predictor.
* **StateDecoder** ``z_hat -> s`` -- Path A's read-out head.
* **ThetaProbe** ``z_hat -> raw theta`` -- Path B's read-out head, deliberately
  low capacity.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.lewm.module import FeedForward

# ----------------------------------------------------------------------
# Encoders:  pixels (B, C, H, W) -> tokens (B, N, D)
# ----------------------------------------------------------------------


from .vq import VectorQuantizer

class TinyCNNEncoder(nn.Module):
    """Small conv encoder producing a patch-token grid.

    Dependency-free stand-in for DINOv2 so smoke tests run without
    downloading weights. Same output contract: ``(B, N, D)`` tokens.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: int = 8,
        width: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f'image_size {image_size} must be divisible by '
            f'patch_size {patch_size}'
        )
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.grid = image_size // patch_size
        self.num_tokens = self.grid**2

        layers, c_in, stride_left = [], in_channels, patch_size
        while stride_left > 1:
            layers += [
                nn.Conv2d(c_in, width, 3, stride=2, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            ]
            c_in, stride_left = width, stride_left // 2
        layers.append(nn.Conv2d(c_in, embed_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pixels):
        z = self.net(pixels)
        return rearrange(z, 'b d h w -> b (h w) d')


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 patch-token encoder -- the DINO-WM backbone.

    Kept frozen by default: DINO-WM's premise is that the visual
    representation is fixed and only the predictor is learned.
    """

    def __init__(
        self,
        model_name: str = 'facebook/dinov2-small',
        frozen: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                'DinoV2Encoder needs `transformers`. Install the train '
                'extra, or use TinyCNNEncoder for smoke tests.'
            ) from exc
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size
        self.image_size = image_size
        patch = self.backbone.config.patch_size
        self.num_tokens = (image_size // patch) ** 2
        self.frozen = frozen
        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:  # a frozen backbone always stays in eval mode
            self.backbone.eval()
        return self

    def forward(self, pixels):
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.backbone(pixels, interpolate_pos_encoding=True)
        # drop the CLS token: DINO-WM predicts over patch tokens
        return out.last_hidden_state[:, 1:]


class StateEncoder(nn.Module):
    """Encode a raw state vector into a single token.

    Ablation path: lets the pipeline run without pixels, isolating the
    two-path objective from the visual representation.
    """

    def __init__(self, state_dim: int, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, state):
        return self.net(state).unsqueeze(1)


# ----------------------------------------------------------------------
# Predictor: block-causal spatio-temporal transformer (DINO-WM)
# ----------------------------------------------------------------------


class MaskedAttention(nn.Module):
    """Multi-head attention accepting an explicit boolean mask."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, attn_mask=None):
        x = self.norm(x)
        q, k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class STBlock(nn.Module):
    """Pre-norm transformer block with a maskable attention."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MaskedAttention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(x, attn_mask=attn_mask)
        return x + self.mlp(x)


class DinoWMPredictor(nn.Module):
    """DINO-WM predictor over patch tokens with block-causal attention.

    Input ``z`` is ``(B, T, N, D)`` patch tokens and ``act_emb`` is
    ``(B, T, A)``. The action embedding is concatenated onto every patch of
    its frame, tokens get separate spatial and temporal position
    embeddings, and attention is **block-causal**: a token in frame ``t``
    attends to every patch of frames ``<= t`` and to nothing later. Output
    at frame ``t`` is the prediction of frame ``t + 1``.

    With ``N == 1`` this degenerates to a pooled-latent predictor, so the
    same module covers both the patch-token and pooled regimes.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        num_tokens: int,
        max_frames: int = 16,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.max_frames = max_frames

        self.in_proj = nn.Linear(embed_dim + action_dim, hidden_dim)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, num_tokens, hidden_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_frames, 1, hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                STBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def block_causal_mask(self, T, N, device):
        """``(T*N, T*N)`` bool mask; True = attention allowed."""
        frame = torch.arange(T, device=device).repeat_interleave(N)
        return frame.unsqueeze(0) <= frame.unsqueeze(1)

    def forward(self, z, act_emb):
        B, T, N, _ = z.shape
        assert T <= self.max_frames, (
            f'{T} frames exceeds max_frames={self.max_frames}'
        )
        assert N == self.num_tokens, (
            f'expected {self.num_tokens} tokens per frame, got {N}'
        )

        # action conditioning: broadcast each frame's action onto its patches
        a = act_emb.unsqueeze(2).expand(B, T, N, act_emb.shape[-1])
        x = self.in_proj(torch.cat([z, a], dim=-1))
        x = x + self.spatial_pos + self.temporal_pos[:, :T]

        mask = self.block_causal_mask(T, N, z.device)
        x = rearrange(x, 'b t n d -> b (t n) d')
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.out_proj(self.norm(x))
        x = rearrange(x, 'b (t n) d -> b t n d', t=T, n=N)
        # residual: predict the *delta* to the current latent
        return z + x


# ----------------------------------------------------------------------
# Read-out heads
# ----------------------------------------------------------------------


def _pool_tokens(z, mode):
    """``(B, T, N, D)`` -> ``(B, T, D')`` patch pooling."""
    if mode == 'mean':
        return z.mean(dim=2)
    if mode == 'max':
        return z.amax(dim=2)
    if mode == 'flatten':
        return rearrange(z, 'b t n d -> b t (n d)')
    raise ValueError(f'unknown pool mode {mode!r}')


def _pooled_dim(embed_dim, num_tokens, mode):
    return embed_dim * num_tokens if mode == 'flatten' else embed_dim


class StateDecoder(nn.Module):
    """Path A read-out: predicted latent -> predicted next state."""

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int,
        state_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        pool: str = 'mean',
    ):
        super().__init__()
        self.pool = pool
        dim = _pooled_dim(embed_dim, num_tokens, pool)
        layers, d = [], dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(_pool_tokens(z, self.pool))


class ThetaProbe(nn.Module):
    """Path B read-out: latent -> **raw** physics parameters.

    Deliberately low capacity. ``hidden_dim = 0`` gives a single linear
    map, which is the honest "is theta linearly decodable from the latent"
    probe; a small hidden layer is available via config.

    ``mode``:
      * ``'episode'`` (default) -- pool over patches **and time** to emit
        one theta per sequence. Physics parameters are properties of the
        episode, not of the timestep, so this is the right default and it
        makes theta strictly a function of the whole observed window.
      * ``'step'`` -- one theta per timestep.

    Whatever the mode, theta is **always** the output of this forward pass.
    It is never an ``nn.Parameter`` and never a free per-step variable that
    the optimizer could fit directly to the targets -- that would let
    Path B cheat by memorizing the answer instead of reading the latent.
    See ``assert_no_free_theta`` in ``physwm.py``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int,
        theta_dim: int,
        hidden_dim: int = 0,
        mode: str = 'episode',
        pool: str = 'mean',
        dropout: float = 0.0,
        detach_input: bool = False,
        init_scale: float = 0.01,
        tactile_tokens: int = 0,
        quantize: bool = False,
        num_codes: int = 64,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        assert mode in ('episode', 'step'), f'bad probe mode {mode!r}'
        assert 0 <= tactile_tokens < num_tokens, (
            f'tactile_tokens {tactile_tokens} outside num_tokens {num_tokens}'
        )
        self.mode = mode
        self.pool = pool
        self.detach_input = detach_input
        self.tactile_tokens = tactile_tokens
        # Tactile tokens get their OWN pathway instead of being averaged in
        # with the observation tokens. Pooling all N together would give the
        # tactile reading a 1/N share of a linear probe's input -- with 16
        # pixel tokens that is ~6%, and it is the only token carrying the
        # information that separates stiffness from mass and drag. Keeping
        # it separate makes it half the input while leaving the probe
        # low-capacity.
        dim = _pooled_dim(embed_dim, num_tokens - tactile_tokens, pool)
        dim += embed_dim * tactile_tokens

        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, theta_dim),
            )
        else:
            self.net = nn.Sequential(nn.Linear(dim, theta_dim))

        # start near a nominal, physically sensible theta
        head = self.net[-1]
        nn.init.normal_(head.weight, std=init_scale)
        nn.init.zeros_(head.bias)

        self.quantizer = (
            VectorQuantizer(
                theta_dim, num_codes=num_codes, commitment_beta=commitment_beta
            )
            if quantize
            else None
        )
        self.last_vq_loss = None
        self.last_vq_indices = None

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_from_solver(self, solver) -> None:
        """Bias-init so the initial theta equals the solver's nominal set.

        For the quantized variant, seed EVERY code near the solver's
        nominal raw-theta scale (not just index 0): ``VectorQuantizer``'s
        own default init is a tiny ``uniform(-1/K, 1/K)``, which has no
        way to know the actual scale ``theta_raw`` operates at. Left
        uncorrected, the encoder's output (unconstrained until
        ``bound_theta``) drifts far outside that tiny range early in
        training, distances to every code but the nearest explode
        together, and the codebook collapses to a single code chasing a
        moving target (confirmed empirically: default init collapsed to
        1/8 active codes with ``loss_vq`` growing, not shrinking, over
        training). Seeding every code in the same neighborhood as
        ``nominal_raw()`` gives the argmin genuine choices from step one.
        """
        with torch.no_grad():
            self.net[-1].bias.copy_(solver.nominal_raw())
            if self.quantizer is not None:
                nominal = solver.nominal_raw()
                noise = 0.1 * torch.randn_like(
                    self.quantizer.codebook.weight
                )
                self.quantizer.codebook.weight.copy_(
                    nominal.unsqueeze(0) + noise
                )

    def features(self, z):
        """Return the exact low-capacity feature consumed by the probe.

        Keeping this operation public makes the paper's supervised
        decodability control use precisely the same representation and
        pooling as the unsupervised probe.  In particular, tactile tokens
        retain their dedicated pathway instead of being accidentally
        averaged into the visual tokens by evaluation code.
        """
        if self.detach_input:
            z = z.detach()
        if self.tactile_tokens:
            k = self.tactile_tokens
            obs, tac = z[:, :, :-k], z[:, :, -k:]
            h = torch.cat(
                [
                    _pool_tokens(obs, self.pool),
                    rearrange(tac, 'b t n d -> b t (n d)'),
                ],
                dim=-1,
            )
        else:
            h = _pool_tokens(z, self.pool)
        if self.mode == 'episode':
            h = h.mean(dim=1)
        return h

    def forward(self, z):
        """``(B, T, N, D)`` -> ``(B, K)`` (episode) or ``(B, T, K)``.

        Any tactile tokens are the LAST ``tactile_tokens`` entries along the
        token axis (see ``PhysWM.encode``); they bypass pooling.

        If quantization is enabled, the returned tensor is the STRAIGHT-
        THROUGH quantized theta (gradients pass through unchanged to the
        pre-quantization value); the commitment+codebook loss is cached on
        ``self.last_vq_loss`` for the caller to add into the total
        objective (see ``PhysWM.forward`` / ``physwm_loss``).
        """
        raw = self.net(self.features(z))
        if self.quantizer is not None:
            raw, vq_loss, indices = self.quantizer(raw)
            self.last_vq_loss = vq_loss
            self.last_vq_indices = indices
        else:
            self.last_vq_loss = None
        return raw


__all__ = [
    'DinoV2Encoder',
    'DinoWMPredictor',
    'MaskedAttention',
    'STBlock',
    'StateDecoder',
    'StateEncoder',
    'ThetaProbe',
    'TinyCNNEncoder',
]
