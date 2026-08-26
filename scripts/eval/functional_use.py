"""Are the recovered parameters *used*, or merely decodable?

A parameter can be readable off a latent and still be inert -- the model
might carry it without the prediction depending on it. Decodability
(``scripts/eval/decodability.py``) cannot tell those apart, because it
only ever reads. This script writes: it intervenes on theta and measures
whether the prediction moves.

Three interventions, all on a trained model, all on held-out episodes:

1. **Sensitivity.** Scale each theta component by ``1 +- eps`` in turn and
   measure the induced change in Path B's predicted next state. A
   component with ~zero sensitivity is inert: the solver ignores it, so
   recovering it would mean nothing.
2. **Substitution.** Replace the probe's theta with (a) the episode's
   TRUE theta, (b) another episode's theta (shuffled), (c) the solver's
   nominal theta. If the probe's theta is functionally right, its error
   should sit near TRUE and clearly below SHUFFLED. If SHUFFLED is just
   as good, theta is not carrying episode-specific physics.
3. **Multi-horizon rollout.** Same substitution, but rolling the frozen
   solver forward H steps from a single true state. Wrong parameters
   compound: the gap should widen with horizon. A gap that stays flat
   means the prediction is dominated by the initial state, not by theta.

Errors are reported against the dataset's true next state, in scaled
units (per-dimension std), so they are comparable across state dims.

Run:
    MUJOCO_GL=egl python scripts/eval/functional_use.py
    MUJOCO_GL=egl python scripts/eval/functional_use.py --epochs 30 --horizon 16
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# sibling script in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decodability import bench_cfg, train


def scaled_rmse(pred, target, scale, mask=None):
    if mask is not None:
        pred, target, scale = pred[..., mask], target[..., mask], scale[mask]
    return float(((pred - target) / scale).pow(2).mean().sqrt())


@torch.no_grad()
def collect(model, loader, device):
    """Everything the interventions need, on held-out windows."""
    model.eval()
    S, A, SN, TH_P, TH_T, EP, G = [], [], [], [], [], [], []
    has_goal = None
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        if 'theta_true' not in batch:
            raise SystemExit('benchmark has no ground-truth theta')
        if has_goal is None:
            has_goal = 'goal' in batch
        out = model(batch)
        th = out['theta']
        if th.dim() == 3:
            th = th.mean(1)
        time_mask = out.get('physics_eval_mask')
        s = batch['state'][:, :-1]
        a = batch['action'][:, :-1]
        s_next = batch['state'][:, 1:]
        if time_mask is not None:
            s, a, s_next = (
                s[:, time_mask], a[:, time_mask], s_next[:, time_mask]
            )
        S.append(s)
        A.append(a)
        SN.append(s_next)
        TH_P.append(th)
        TH_T.append(batch['theta_true'])
        EP.append(batch['episode'])
        if has_goal:
            G.append(batch['goal'])
    s, a, sn = torch.cat(S), torch.cat(A), torch.cat(SN)
    th_p, th_t, episode = torch.cat(TH_P), torch.cat(TH_T), torch.cat(EP)
    g = torch.cat(G) if has_goal else None
    rows = []
    for ep in episode.unique(sorted=True):
        keep = (episode == ep).nonzero(as_tuple=False).flatten()
        first = keep[0]
        row = [s[first], a[first], sn[first], th_p[keep].mean(0), th_t[first]]
        if has_goal:
            row.append(g[first])
        rows.append(tuple(row))
    n = len(rows[0])
    out = tuple(torch.stack([row[i] for row in rows]) for i in range(n))
    return out if has_goal else out + (None,)


def sensitivity(solver, s, a, theta, names, eps=0.1, mask=None):
    """d(prediction) / d(scaling each theta component)."""
    base = solver(s, a, theta.unsqueeze(1).expand(-1, s.shape[1], -1))
    scale = base.reshape(-1, base.shape[-1]).std(0).clamp_min(1e-6)
    out = {}
    for i, n in enumerate(names):
        bumped = theta.clone()
        bumped[:, i] = bumped[:, i] * (1.0 + eps)
        pred = solver(s, a, bumped.unsqueeze(1).expand(-1, s.shape[1], -1))
        out[n] = scaled_rmse(pred, base, scale, mask)
    return out


def theta_variation(theta_true, names, rel_tol=1e-6):
    """Per-parameter spread across episodes.

    PokeWorld fixes the two radii by construction, so they are identical
    in every episode. Shuffling theta between episodes cannot perturb
    them, and the substitution gap is therefore generated entirely by the
    parameters that do vary. Reporting sensitivity without that context
    reads as if the method were chasing inert parameters, so mark them.
    """
    std = theta_true.float().std(0)
    mean = theta_true.float().mean(0).abs()
    rel = std / mean.clamp_min(1e-12)
    return {
        n: {'std': float(std[i]), 'mean': float(theta_true[:, i].float().mean()),
            'varies': bool(rel[i] > rel_tol)}
        for i, n in enumerate(names)
    }


def substitution(
    solver, s, a, s_next, theta_probe, theta_true, scale, mask=None
):
    """Path B error under each theta source, one step."""
    def err(th):
        return scaled_rmse(
            solver(s, a, th.unsqueeze(1).expand(-1, s.shape[1], -1)),
            s_next, scale, mask,
        )

    perm = torch.randperm(len(theta_true), device=theta_true.device)
    nominal = solver.theta_nominal.to(theta_true.device)
    return {
        'probe': err(theta_probe),
        'true': err(theta_true),
        'shuffled': err(theta_true[perm]),
        'nominal': err(nominal.unsqueeze(0).expand(len(theta_true), -1)),
    }


def rollout(solver, s0, a, theta, horizon):
    """Roll the frozen solver forward from a single true state."""
    s = s0
    preds = []
    for t in range(horizon):
        s = solver(s, a[:, t:t + 1], theta.unsqueeze(1)).squeeze(1)
        preds.append(s)
        s = s.unsqueeze(1)
    return torch.stack(preds, 1)


def success_rate(solver, s0, a, theta, goal, obj_idx, threshold, horizon):
    """Fraction of episodes whose object position ever comes within
    ``threshold`` of ``goal`` during an H-step FROZEN-SOLVER rollout under
    ``theta``. This is the solver-space analog of the real env's
    ``is_success`` -- a substituted theta can only be evaluated inside the
    differentiable solver, not the real MuJoCo sim, so "success" here
    means "the physics this theta implies would have reached the goal",
    not "the real robot reached it".
    """
    horizon = min(horizon, a.shape[1])  # same guard as multi_horizon()
    preds = rollout(solver, s0, a, theta, horizon)  # (B, horizon, state_dim)
    obj_traj = preds[..., obj_idx]  # (B, horizon, len(obj_idx))
    dist = (obj_traj - goal.unsqueeze(1)).norm(dim=-1)  # (B, horizon)
    reached = dist.min(dim=1).values < threshold
    return float(reached.float().mean())


def task_success(solver, s0, a, theta_probe, theta_true, goal, obj_idx, threshold, horizon):
    """Same true/probe/shuffled/nominal comparison as ``substitution()``,
    scored as success rate instead of prediction error -- turns "is theta
    functionally used" into "does using it reach the goal".
    """
    perm = torch.randperm(len(theta_true), device=theta_true.device)
    nominal = solver.theta_nominal.to(theta_true.device)
    sources = {
        'probe': theta_probe,
        'true': theta_true,
        'shuffled': theta_true[perm],
        'nominal': nominal.unsqueeze(0).expand(len(theta_true), -1),
    }
    return {
        k: success_rate(solver, s0, a, th, goal, obj_idx, threshold, horizon)
        for k, th in sources.items()
    }


def multi_horizon(
    solver, s, a, s_next, theta_probe, theta_true, scale, horizon,
    mask=None,
):
    horizon = min(horizon, a.shape[1])
    s0 = s[:, :1]
    target = s_next[:, :horizon]
    perm = torch.randperm(len(theta_true), device=theta_true.device)
    nominal = solver.theta_nominal.to(theta_true.device)
    sources = {
        'probe': theta_probe,
        'true': theta_true,
        'shuffled': theta_true[perm],
        'nominal': nominal.unsqueeze(0).expand(len(theta_true), -1),
    }
    curves = {}
    for name, th in sources.items():
        pred = rollout(solver, s0, a, th, horizon)
        curves[name] = [
            scaled_rmse(
                pred[:, :h + 1], target[:, :h + 1], scale, mask
            )
            for h in range(horizon)
        ]
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--length', type=int, default=48)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--encoder', default='tiny_cnn', choices=['tiny_cnn', 'dinov2'])
    ap.add_argument('--probe-hidden', type=int, default=0)
    ap.add_argument('--probe-source', default='predicted',
                    choices=['predicted', 'encoded'])
    ap.add_argument('--detach-probe-input', action='store_true')
    ap.add_argument('--physics-target', default='path_a',
                    choices=['path_a', 'dataset'])
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--no-tactile', action='store_true')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--theta-supervision', type=float, default=0.0)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument(
        '--benchmark', default='pokeworld',
        choices=['pokeworld', 'pusht_rand', 'fetch_push'],
    )
    ap.add_argument(
        '--goal-threshold', type=float, default=0.05,
        help='distance (real units) within which the object counts as '
             'having reached the goal in a solver rollout -- 0.05 matches '
             "Gymnasium-Robotics Fetch's own is_success threshold.",
    )
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    cfg = bench_cfg(
        args.episodes,
        args.length,
        encoder=args.encoder,
        probe_hidden=args.probe_hidden,
        probe_source=args.probe_source,
        detach_probe_input=args.detach_probe_input,
        window=args.window,
        tactile=not args.no_tactile,
        benchmark=args.benchmark,
    )
    model, tl, vl = train(
        cfg,
        args.alpha,
        args.epochs,
        args.seed,
        device,
        batch_size=args.batch_size,
        theta_supervision=args.theta_supervision,
        physics_target=args.physics_target,
        amp=not args.no_amp,
    )
    solver = model.solver
    names = list(solver.theta_names)

    s, a, s_next, th_p, th_t, goal = collect(model, vl, device)
    scale = s.reshape(-1, s.shape[-1]).std(0).clamp_min(1e-6)
    mask = model.state_loss_mask

    # task-success layer: only meaningful where a goal exists AND the
    # solver's state includes an explicit object position (Fetch), not
    # PokeWorld/pusht_rand -- skip cleanly rather than guessing indices
    success = None
    if goal is not None and hasattr(solver, 'state_names') and (
        'object_x' in solver.state_names and 'object_y' in solver.state_names
    ):
        obj_idx = [
            solver.state_names.index('object_x'),
            solver.state_names.index('object_y'),
        ]
        success = task_success(
            solver, s[:, :1], a, th_p, th_t, goal, obj_idx,
            args.goal_threshold, args.horizon,
        )

    sens = sensitivity(solver, s, a, th_p, names, eps=args.eps, mask=mask)
    variation = theta_variation(th_t, names)
    sub = substitution(
        solver, s, a, s_next, th_p, th_t, scale, mask=mask
    )
    curves = multi_horizon(
        solver, s, a, s_next, th_p, th_t, scale, args.horizon, mask=mask
    )

    w = max(len(n) for n in names) + 1
    print(f'\nFunctional use of theta -- PokeWorld, {args.episodes} episodes '
          f'x {args.length}, {args.epochs} epochs, seed {args.seed}')

    print(f'\n1. SENSITIVITY: prediction shift from scaling theta by '
          f'{1 + args.eps:g}x')
    print('   (near zero => the solver ignores that component; recovering '
          'it is meaningless)')
    for n in names:
        tag = '' if variation[n]['varies'] else '   [constant by construction]'
        print(f'   {n:<{w}} {sens[n]:>10.5f}{tag}')
    live = [n for n in names if variation[n]['varies']]
    if len(live) < len(names):
        print(f'\n   only {len(live)} of {len(names)} parameters vary across '
              'episodes; the substitution\n   gap below is generated by '
              f'{", ".join(live)} alone.')

    print('\n2. SUBSTITUTION: one-step Path B error vs true s_next '
          '(scaled RMSE, lower better)')
    for k in ('true', 'probe', 'nominal', 'shuffled'):
        print(f'   {k:<{w}} {sub[k]:>10.5f}')
    gap = sub['shuffled'] - sub['true']
    got = sub['shuffled'] - sub['probe']
    frac = got / gap if abs(gap) > 1e-12 else float('nan')
    print(f'\n   shuffled-minus-true gap : {gap:>10.5f}   '
          '(how much episode-specific theta is worth at all)')
    print(f'   probe closes            : {frac:>10.1%} of it')

    print(f'\n3. MULTI-HORIZON rollout error (scaled RMSE, horizon 1..'
          f'{len(curves["true"])})')
    hdr = ''.join(f'{h + 1:>9}' for h in range(len(curves['true'])))
    print(f'   {"source":<{w}}{hdr}')
    for k in ('true', 'probe', 'nominal', 'shuffled'):
        row = ''.join(f'{v:>9.4f}' for v in curves[k])
        print(f'   {k:<{w}}{row}')
    print('\n   a probe curve tracking `true` and separating from `shuffled` '
          'as the\n   horizon grows is the evidence that theta is '
          'functionally used.')

    if success is not None:
        print(f'\n4. TASK SUCCESS: fraction of episodes reaching the goal '
              f'in a {args.horizon}-step solver rollout (threshold '
              f'{args.goal_threshold})')
        for k in ('true', 'probe', 'nominal', 'shuffled'):
            print(f'   {k:<{w}} {success[k]:>10.1%}')
        print('\n   this scores the SAME true/probe/shuffled/nominal '
              'comparison as (2)/(3) by task outcome\n   instead of '
              'prediction error -- a probe rate near `true` and clearly '
              'above `shuffled`\n   means the recovered physics is worth '
              'something for the actual task, not just accurate\n   in an '
              'RMSE sense.')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'meta': {'encoder': args.encoder,
                      'probe_hidden': args.probe_hidden,
                      'probe_source': args.probe_source,
                      'detach_probe_input': args.detach_probe_input,
                      'physics_target': args.physics_target,
                      'window': args.window,
                      'tactile': not args.no_tactile,
                      'windows': {
                          'train': len(tl.dataset), 'val': len(vl.dataset),
                      },
                      'epochs': args.epochs, 'episodes': args.episodes,
                      'length': args.length, 'alpha': args.alpha,
                      'seed': args.seed,
                      'theta_supervision': args.theta_supervision},
             'sensitivity': sens, 'theta_variation': variation,
             'substitution': sub,
             'multi_horizon': curves,
             'task_success': success}, indent=2))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
