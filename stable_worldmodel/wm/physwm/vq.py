"""Vector-quantized theta bottleneck -- a design-space ablation.

Inserted inside :class:`~stable_worldmodel.wm.physwm.module.ThetaProbe`,
between the linear/MLP head and the frozen solver: does forcing theta
onto a learned discrete codebook regularize the self-distilled
representation? Continuous is the default; this is the alternative arm
of that ablation.

Standard VQ-VAE recipe (codebook + nearest-neighbour lookup +
straight-through estimator + commitment loss), scoped tightly to theta:
the codebook lives in raw (pre-``bound_theta``) space, so the solver and
its frozen-parameter invariants (see ``assert_no_free_theta`` in
``physwm.py``) are completely unaware this exists.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Nearest-code lookup with a straight-through gradient and commitment loss.

    NOTE: the codebook is a genuine learnable ``nn.Parameter`` (via
    ``nn.Embedding``) -- that is fine, it is part of the probe, not a
    free per-step theta. Just don't name it anything containing
    ``'theta'``: ``assert_no_free_theta`` rejects any parameter name
    that does, as a blanket check against exactly that kind of leak.
    """

    def __init__(
        self, dim: int, num_codes: int = 64, commitment_beta: float = 0.25
    ):
        super().__init__()
        self.num_codes = num_codes
        self.commitment_beta = commitment_beta
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes
        )

    def forward(self, raw: torch.Tensor):
        """``raw``: ``(..., dim)`` -> ``(quantized, loss, indices)``.

        ``quantized`` has the same shape as ``raw``. Gradients flow
        through it straight back to ``raw`` (straight-through estimator),
        so whatever produced ``raw`` still trains as if quantization
        were the identity; ``loss`` is the codebook + commitment terms
        the caller must add into the total objective for the codebook to
        train at all.
        """
        shape = raw.shape
        flat = raw.reshape(-1, shape[-1])

        # (N, num_codes) squared distance to every code
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        quantized_flat = self.codebook(indices)

        # codebook loss pulls codes toward the encoder output; commitment
        # loss pulls the encoder output toward its assigned code (scaled
        # down by commitment_beta so the encoder isn't over-constrained)
        codebook_loss = F.mse_loss(quantized_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized_flat.detach())
        loss = codebook_loss + self.commitment_beta * commitment_loss

        # straight-through: forward uses the quantized value, backward
        # copies the gradient straight through to `flat` unchanged
        quantized_flat = flat + (quantized_flat - flat).detach()

        quantized = quantized_flat.reshape(shape)
        indices = indices.reshape(shape[:-1])
        return quantized, loss, indices

    @torch.no_grad()
    def code_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Histogram of code usage -- check the codebook isn't collapsed."""
        return torch.bincount(indices.reshape(-1), minlength=self.num_codes)


__all__ = ['VectorQuantizer']
