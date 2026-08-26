"""The PhysWM objective.

    L = L_A + alpha * L_B + beta * L_consistency

    L_A            = || state_A          - s_next   ||^2
    L_B            = || solver(theta)    - state_A' ||^2   (state_A' = state_A.detach())
    L_consistency  = || state_A          - solver(theta) ||^2      (beta = 0)

``L_A`` regresses the **dataset ground truth** ``s_next`` -- Path A is the
model's general-purpose next-state prediction and must stay anchored to
reality. ``L_B`` regresses **Path A's own prediction**, detached, instead
of the dataset label: the experiment is whether a physics parameterization
can explain what the world model already believes happens next, not
whether the solver can re-derive the raw benchmark label directly. The
detach is what keeps this a "ground truth" in the loss sense -- the target
for ``L_B`` carries no gradient of its own, so fitting the physics path
never pulls Path A's decoder toward the solver. ``L_consistency`` is off by
default (``beta = 0``) and, when enabled, is symmetric -- it pulls the two
paths together rather than pointing one at the other.

All terms are computed in normalized state space (see ``physwm.py``) so
they are commensurate and ``alpha``/``beta`` are scale-free.
"""

import torch

_DETACH_MODES = ('none', 'a', 'b')
_PHYSICS_TARGETS = ('path_a', 'dataset')


def physwm_loss(
    out: dict,
    alpha: float = 1.0,
    beta: float = 0.0,
    consistency_detach: str = 'none',
    physics_target: str = 'path_a',
    vq_beta: float = 1.0,
) -> dict:
    """Compute the composite objective and its diagnostics.

    Args:
        out: the dict returned by :meth:`PhysWM.forward`.
        alpha: weight on the physics path ``L_B``.
        beta: weight on the consistency term. **0 by default.**
        consistency_detach:
            ``'none'`` (default) -- symmetric, gradients reach both paths.
            ``'b'`` -- detach Path B, so consistency only pulls A toward B.
            ``'a'`` -- detach Path A. Redundant with ``L_B`` (both become
            ``(a.detach() - b)^2``); kept only so the consistency term can
            be swept independently of ``alpha``. Inert while ``beta == 0``.
        physics_target:
            ``'path_a'`` (default) is the workshop method: Path B learns
            from Path A's stopped-gradient, high-signal prediction.
            ``'dataset'`` directly targets ``s_next`` and exists only as
            the target-source ablation.

    Returns:
        dict with ``loss`` plus every component, detached for logging.
    """
    assert consistency_detach in _DETACH_MODES, (
        f'consistency_detach must be one of {_DETACH_MODES}'
    )
    assert physics_target in _PHYSICS_TARGETS, (
        f'physics_target must be one of {_PHYSICS_TARGETS}'
    )

    mask = out.get('state_loss_mask')
    a, b, target = out['state_a'], out['state_b'], out['target']
    if mask is not None:
        a, b, target = a[..., mask], b[..., mask], target[..., mask]

    # --- A regresses the dataset ground truth ----------------------------
    loss_a = (a - target).pow(2).mean()
    time_mask = out.get('physics_loss_mask')
    if time_mask is not None:
        a_phys = a[:, time_mask]
        b_phys = b[:, time_mask]
        target_phys = target[:, time_mask]
    else:
        a_phys, b_phys, target_phys = a, b, target
    # --- B regresses Path A's (detached) prediction, not the benchmark ---
    # label: the physics probe is fit to explain the WM's general belief
    # about the next state. Detaching blocks the teacher/decoder edge; the
    # shared predictor is still shaped deliberately through B's probe input.
    teacher = (
        a_phys.detach() if physics_target == 'path_a' else target_phys
    )
    loss_b = (b_phys - teacher).pow(2).mean()

    a_c = a_phys.detach() if consistency_detach == 'a' else a_phys
    b_c = b_phys.detach() if consistency_detach == 'b' else b_phys
    loss_consistency = (a_c - b_c).pow(2).mean()

    vq_loss = out.get('vq_loss')
    loss = loss_a + alpha * loss_b + beta * loss_consistency
    if vq_loss is not None:
        loss = loss + vq_beta * vq_loss

    return {
        'loss': loss,
        'loss_a': loss_a,
        'loss_b': loss_b,
        'loss_consistency': loss_consistency,
        'loss_vq': (
            vq_loss.detach() if vq_loss is not None else torch.tensor(0.0)
        ),
    }


@torch.no_grad()
def physwm_metrics(out: dict, solver=None) -> dict:
    """Diagnostics logged alongside the loss.

    * per-path RMSE in normalized state units;
    * theta summary statistics per named parameter.

    ``rmse_a`` uses all transitions. Path-B metrics use the held-out query
    mask when a context/query split is configured: ``rmse_b`` is fidelity
    to Path A and ``rmse_b_vs_dataset`` is accuracy against real ``s_next``.
    They are evaluation diagnostics, not necessarily the transitions used
    by the physical training loss.

    The identifiability certificate (:func:`theta_r2`) is deliberately
    NOT computed here: R^2 is only meaningful over a sample in which the
    true parameter actually varies, and a single minibatch usually holds
    windows from just one or two episodes. Accumulate theta over a whole
    epoch and call :func:`theta_r2` once instead.
    """
    metrics = {}
    mask = out.get('state_loss_mask')
    tgt, a, b = out['target'], out['state_a'], out['state_b']
    if mask is not None:
        tgt, a, b = tgt[..., mask], a[..., mask], b[..., mask]
    metrics['rmse_a'] = (a - tgt).pow(2).mean().sqrt()
    time_mask = out.get('physics_eval_mask')
    if time_mask is not None:
        tgt, a, b = tgt[:, time_mask], a[:, time_mask], b[:, time_mask]
    metrics['rmse_b'] = (b - a).pow(2).mean().sqrt()
    metrics['rmse_b_vs_teacher'] = metrics['rmse_b']
    metrics['rmse_b_vs_dataset'] = (b - tgt).pow(2).mean().sqrt()

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
