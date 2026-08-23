"""The PhysWM objective.

    L = L_A + alpha * L_B + beta * L_consistency

    L_A            = || state_A          - s_next ||^2
    L_B            = || solver(theta)    - s_next ||^2
    L_consistency  = || state_A          - solver(theta) ||^2      (beta = 0)

Both ``L_A`` and ``L_B`` regress the **dataset ground truth** ``s_next``.
Path B is *never* supervised on Path A's output: that would turn the
physics path into a distillation of the learned path and destroy the
point of the experiment (an independent physical estimate of the same
transition). ``L_consistency`` is off by default (``beta = 0``) and, when
enabled, is symmetric -- it pulls the two paths together rather than
pointing one at the other.

All terms are computed in normalized state space (see ``physwm.py``) so
they are commensurate and ``alpha``/``beta`` are scale-free.
"""

import torch

_DETACH_MODES = ('none', 'a', 'b')


def physwm_loss(
    out: dict,
    alpha: float = 1.0,
    beta: float = 0.0,
    consistency_detach: str = 'none',
) -> dict:
    """Compute the composite objective and its diagnostics.

    Args:
        out: the dict returned by :meth:`PhysWM.forward`.
        alpha: weight on the physics path ``L_B``.
        beta: weight on the consistency term. **0 by default.**
        consistency_detach:
            ``'none'`` (default) -- symmetric, gradients reach both paths.
            ``'b'`` -- detach Path B, so consistency only pulls A toward B.
            ``'a'`` -- detach Path A. NOTE this makes the consistency term
            a supervision of B on A, which violates the design rule above.
            It exists only as an explicit, opt-in ablation and is inert
            while ``beta == 0``.

    Returns:
        dict with ``loss`` plus every component, detached for logging.
    """
    assert consistency_detach in _DETACH_MODES, (
        f'consistency_detach must be one of {_DETACH_MODES}'
    )

    a, b, target = out['state_a'], out['state_b'], out['target']

    # --- both paths regress ground truth; B never regresses A -----------
    loss_a = (a - target).pow(2).mean()
    loss_b = (b - target).pow(2).mean()

    a_c = a.detach() if consistency_detach == 'a' else a
    b_c = b.detach() if consistency_detach == 'b' else b
    loss_consistency = (a_c - b_c).pow(2).mean()

    loss = loss_a + alpha * loss_b + beta * loss_consistency

    return {
        'loss': loss,
        'loss_a': loss_a,
        'loss_b': loss_b,
        'loss_consistency': loss_consistency,
    }


@torch.no_grad()
def physwm_metrics(out: dict, solver=None) -> dict:
    """Diagnostics logged alongside the loss.

    * per-path RMSE in physical units (interpretable, unlike the
      normalized loss);
    * theta summary statistics per named parameter.

    The identifiability certificate (:func:`theta_r2`) is deliberately
    NOT computed here: R^2 is only meaningful over a sample in which the
    true parameter actually varies, and a single minibatch usually holds
    windows from just one or two episodes. Accumulate theta over a whole
    epoch and call :func:`theta_r2` once instead.
    """
    metrics = {}
    tgt = out['target']
    metrics['rmse_a'] = (out['state_a'] - tgt).pow(2).mean().sqrt()
    metrics['rmse_b'] = (out['state_b'] - tgt).pow(2).mean().sqrt()

    theta = out['theta']
    flat = theta.reshape(-1, theta.shape[-1])
    if solver is not None:
        for i, name in enumerate(solver.theta_names):
            metrics[f'theta/{name}_mean'] = flat[:, i].mean()
            metrics[f'theta/{name}_std'] = flat[:, i].std(unbiased=False)

    return metrics


@torch.no_grad()
def theta_r2(theta_pred, theta_true, names, min_rel_std=1e-6) -> dict:
    """Per-parameter coefficient of determination R^2.

    This is the identifiability certificate: how much of the variation in
    each *true* physics parameter the probe's estimate actually explains.

    Must be called on a sample spanning **many episodes**. R^2 compares
    against the variance of the true parameter, so if that parameter is
    constant over the sample the denominator collapses and the score
    explodes to a meaningless large negative number. Two guards:

    * parameters that are constant by construction (PokeWorld's fixed
      radii) or constant within the sample yield ``nan``, not a huge
      negative value -- R^2 is genuinely undefined there;
    * callers should accumulate over a whole epoch rather than scoring
      per minibatch.

    Parameters that only enter the dynamics as a ratio (PokeWorld's mass
    vs stiffness vs drag) are expected to score near zero individually
    even with a perfect probe -- that is a property of the system, and
    surfacing it is the point of the certificate.
    """
    pred = theta_pred.reshape(-1, theta_pred.shape[-1]).float()
    true = theta_true.reshape(-1, theta_true.shape[-1]).float()
    assert pred.shape == true.shape, (
        f'theta shape mismatch: pred {pred.shape} vs true {true.shape}'
    )
    ss_res = (true - pred).pow(2).sum(0)
    centered = true - true.mean(0)
    ss_tot = centered.pow(2).sum(0)

    # undefined when the true parameter does not vary over the sample
    scale = true.abs().mean(0).clamp_min(1e-12)
    varies = true.std(0) > min_rel_std * scale
    r2 = torch.where(
        varies,
        1.0 - ss_res / ss_tot.clamp_min(1e-12),
        torch.full_like(ss_res, float('nan')),
    )
    return {f'r2/{n}': r2[i] for i, n in enumerate(names)}


__all__ = ['physwm_loss', 'physwm_metrics', 'theta_r2']
