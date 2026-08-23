"""Solver adequacy check: can each frozen solver fit its environment?

This is a **diagnostic**, not part of the training pipeline. It optimizes
a *free* theta directly against real transitions to measure the best a
solver could possibly do on an environment -- an oracle upper bound for
Path B. The training pipeline deliberately forbids this: there, theta is
always ``probe(latent)``, never a free variable fitted to the targets
(see ``assert_no_free_theta``). The two are kept apart on purpose, and
this file is the only place a free theta is ever created.

Reported per benchmark:

* ``persistence`` -- RMSE of predicting ``s_next = s`` (the trivial
  baseline any useful solver must beat).
* ``nominal``     -- RMSE of the solver at its hand-set nominal theta.
* ``fitted``      -- RMSE after fitting theta freely (the oracle bound).

Run:  MUJOCO_GL=egl python scripts/smoke/validate_solvers.py
"""

import argparse

import numpy as np
import torch

from stable_worldmodel.wm.physwm import (
    BENCHMARKS,
    build_solver,
    env_episodes,
    pokeworld_episodes,
)


def gather_transitions(bench: str, episodes: int, length: int, seed: int):
    """Return ``(s_t, a_t, s_next)`` tensors of shape ``(N, 1, D)``."""
    if bench == 'pokeworld':
        eps, _ = pokeworld_episodes(episodes, length, seed=seed, render=False)
    else:
        eps = env_episodes(
            BENCHMARKS[bench]['env_id'],
            num_episodes=episodes,
            length=length,
            seed=seed,
            render=False,
        )
    s = np.concatenate([e['state'][:-1] for e in eps])
    a = np.concatenate([e['action'][:-1] for e in eps])
    s_next = np.concatenate([e['state'][1:] for e in eps])
    # episode id per transition: theta is a per-EPISODE quantity, so the
    # oracle must fit one theta per episode (PokeWorld draws a different
    # true theta for every episode). Fitting a single global theta would
    # understate what the solver can do.
    ep_id = np.concatenate(
        [np.full(e['state'].shape[0] - 1, i) for i, e in enumerate(eps)]
    )
    def to(x):
        return torch.from_numpy(x).float().unsqueeze(1)

    return to(s), to(a), to(s_next), torch.from_numpy(ep_id).long()


def rmse(pred, target, scale):
    """Per-dimension RMSE, scaled so dimensions are comparable."""
    err = ((pred - target) / scale).pow(2).mean(dim=(0, 1)).sqrt()
    return err


def fit_theta(solver, s, a, s_next, scale, ep_id, steps=600, lr=0.05, seed=0):
    """Fit a FREE per-episode theta by gradient descent -- diagnostic only.

    One theta per episode, mirroring the model's per-episode probe. This
    is the ONLY place in the project where theta is a free optimizable
    variable, and it exists purely to bound what the solver could achieve.
    """
    torch.manual_seed(seed)
    n_ep = int(ep_id.max().item()) + 1
    raw = solver.nominal_raw().clone().repeat(n_ep, 1).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        theta = solver.bound_theta(raw)[ep_id]
        pred = solver(s, a, theta)
        loss = ((pred - s_next) / scale).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        theta = solver.bound_theta(raw)[ep_id]
        return theta, solver(s, a, theta)


def report(bench, args):
    spec = BENCHMARKS[bench]
    s, a, s_next, ep_id = gather_transitions(
        bench, args.episodes, args.length, args.seed
    )
    solver = build_solver(
        spec['solver'], dt=args.dt[bench], substeps=args.substeps[bench]
    )
    names = solver.state_names
    scale = s.reshape(-1, s.shape[-1]).std(0).clamp_min(1e-6)

    print(f'\n=== {bench} ===')
    print(f'transitions: {s.shape[0]}   state dims: {list(names)}')

    base = rmse(s, s_next, scale)
    nominal_theta = solver.theta_nominal.unsqueeze(0).expand(s.shape[0], -1)
    nom = rmse(solver(s, a, nominal_theta), s_next, scale)
    theta, pred = fit_theta(
        solver,
        s,
        a,
        s_next,
        scale,
        ep_id,
        steps=args.steps,
        seed=args.seed,
    )
    fit = rmse(pred, s_next, scale)

    width = max(len(n) for n in names) + 1
    print(f'{"dim":<{width}} {"persist":>10} {"nominal":>10} {"fitted":>10}')
    for i, n in enumerate(names):
        print(f'{n:<{width}} {base[i]:>10.4f} {nom[i]:>10.4f} {fit[i]:>10.4f}')
    print(
        f'{"MEAN":<{width}} {base.mean():>10.4f} '
        f'{nom.mean():>10.4f} {fit.mean():>10.4f}'
    )
    print('fitted theta (episode 0):')
    for i, n in enumerate(solver.theta_names):
        print(f'  {n:<20} {theta[0, i].item():>12.5f}')
    return {
        'persistence': base.mean().item(),
        'nominal': nom.mean().item(),
        'fitted': fit.mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--benchmarks', nargs='*', default=list(BENCHMARKS), type=str
    )
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--length', type=int, default=64)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    # must match the transition interval of each environment
    args.dt = {'pokeworld': 0.01, 'cartpole': 0.01, 'pusht': 0.01}
    args.substeps = {'pokeworld': 10, 'cartpole': 2, 'pusht': 10}

    results = {b: report(b, args) for b in args.benchmarks}

    print('\n=== summary (scaled RMSE, lower is better) ===')
    for b, r in results.items():
        verdict = (
            'solver beats persistence'
            if r['fitted'] < r['persistence']
            else 'SOLVER DOES NOT BEAT PERSISTENCE'
        )
        print(
            f'{b:<12} persist {r["persistence"]:.4f}  '
            f'nominal {r["nominal"]:.4f}  fitted {r["fitted"]:.4f}  '
            f'-> {verdict}'
        )


if __name__ == '__main__':
    main()
