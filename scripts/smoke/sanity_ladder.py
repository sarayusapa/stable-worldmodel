"""Preconditions for testing the PhysWM hypothesis, cheapest first.

Every rung here corresponds to a real bug that silently invalidated a
result during development. The hypothesis -- can a low-capacity probe
induce physical parameters from a world model's own prediction? -- is
only testable if every rung passes. A negative theta R^2 means nothing
if the encoder cannot see the object, and that is exactly what happened:
an entire overnight sweep measured a pipeline whose observations were
blank 17% of the time.

Run this before any experiment whose result you intend to believe.

    MUJOCO_GL=egl python scripts/smoke/sanity_ladder.py
    MUJOCO_GL=egl python scripts/smoke/sanity_ladder.py --quick

Rungs:

1. **Solver is exact.** Given true theta, the frozen solver reproduces
   the simulator. If it does not, Path B is measuring solver error, not
   physics. (Caught: the walls had to be added to BOTH sides.)
2. **Observations are informative.** The object is inside the rendered
   field and visible. (Caught: raising stiffness to the reference range
   launched it off-screen in 21% of frames.)
3. **Normalizer is calibrated.** Fitted stats generalize to held-out
   episodes. Every loss term lives in normalized space, so a bad fit
   silently reweights the state dimensions. (Caught: `collect_stats`
   sampled the first ~12 episodes.)
4. **Theta is identifiable at all.** The oracle certificate: free theta
   fitted to true transitions. This is the ceiling for any probe.
   (Caught: without a tactile channel the ceiling is ~0, so no probe
   could ever have succeeded.)
5. **The encoder can perceive position.** A supervised read-out of the
   current state from the latent. (Caught: mean-pooling patch tokens is
   permutation-invariant and averages position away.)
6. **Path A beats trivial baselines.** The world model actually predicts.
   Without this there is no "accurate but blind" gap to explain, and
   Path B is chasing a target that carries no dynamics.
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'eval'))

from stable_worldmodel.wm.physwm import (  # noqa: E402
    build_datasets,
    build_physwm,
    build_solver,
    collect_stats,
    pokeworld_episodes,
)
from stable_worldmodel.wm.physwm.loss import theta_r2  # noqa: E402
from stable_worldmodel.wm.physwm.module import _pool_tokens  # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f'  {"PASS" if ok else "FAIL"}  {name}\n        {detail}', flush=True)
    return ok


# ----------------------------------------------------------------------


def rung1_solver_exact(episodes, length):
    eps, _ = pokeworld_episodes(episodes, length, seed=0, render=False)
    to = lambda x: torch.from_numpy(np.stack(x)).float()  # noqa: E731
    s = to([e['state'][:-1] for e in eps])
    a = to([e['action'][:-1] for e in eps])
    sn = to([e['state'][1:] for e in eps])
    th = to([e['theta_true'] for e in eps])
    solver = build_solver('pokeworld', dt=0.001, substeps=20)
    err = float((solver(s, a, th) - sn).abs().max())
    return check(
        'solver reproduces the simulator given true theta',
        err < 1e-3,
        f'max abs error {err:.2e} (tolerance 1e-3)',
    )


def rung2_observations_informative(episodes, length):
    eps, _ = pokeworld_episodes(episodes, length, seed=0, render=True)
    S = np.concatenate([e['state'] for e in eps])
    P = np.concatenate([e['pixels'] for e in eps])
    off = float((np.abs(S[:, :2]) > 1.0).any(1).mean())
    blank = float((P[:, 0].max(axis=(1, 2)) < 0.05).mean())
    return check(
        'object stays in frame and is visible',
        off < 0.01 and blank < 0.01,
        f'off-screen {off:.2%}, invisible {blank:.2%} (both must be <1%)',
    )


def rung3_normalizer_calibrated(episodes, length):
    cfg = _cfg(episodes, length)
    tr, va = build_datasets(cfg, seed=0)
    st, _ = collect_stats(tr)
    mu = st.reshape(-1, st.shape[-1]).mean(0)
    sd = st.reshape(-1, st.shape[-1]).std(0).clamp_min(1e-6)
    V = torch.from_numpy(
        np.stack([e['state'] for e in va.episodes])
    ).float().reshape(-1, st.shape[-1])
    z = (V - mu) / sd
    rmse = float(z.pow(2).mean().sqrt())
    return check(
        'normalizer generalizes to held-out episodes',
        rmse < 1.6,
        f'RMSE of predicting 0 on val = {rmse:.3f} (want ~1.0, fail >1.6)',
    )


def rung4_theta_identifiable(episodes, length, steps):
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    from validate_solvers import fit_theta

    eps, _ = pokeworld_episodes(episodes, length, seed=0, render=False)
    tt = torch.from_numpy(np.stack([e['theta_true'] for e in eps])).float()
    s = np.concatenate([e['state'][:-1] for e in eps])
    a = np.concatenate([e['action'][:-1] for e in eps])
    sn = np.concatenate([e['state'][1:] for e in eps])
    ep = torch.from_numpy(np.concatenate(
        [np.full(e['state'].shape[0] - 1, i) for i, e in enumerate(eps)]
    )).long()
    to = lambda x: torch.from_numpy(x).float().unsqueeze(1)  # noqa: E731
    s, a, sn = to(s), to(a), to(sn)
    solver = build_solver('pokeworld', dt=0.001, substeps=20)
    scale = s.reshape(-1, s.shape[-1]).std(0).clamp_min(1e-6)
    th, _ = fit_theta(solver, s, a, sn, scale, ep, steps=steps, seed=0)
    first = torch.tensor(
        [(ep == i).nonzero()[0].item() for i in range(episodes)]
    )
    r2 = theta_r2(th[first].squeeze(1), tt, solver.theta_names)
    vals = {n: float(r2[f'r2/{n}'])
            for n in ('mass', 'contact_stiffness', 'drag')}
    worst = min(vals.values())
    return check(
        'theta is identifiable from true transitions (oracle ceiling)',
        worst > 0.3,
        ' '.join(f'{k} {v:.3f}' for k, v in vals.items())
        + f'  (worst {worst:.3f}, need >0.3)',
    )


def rung5_encoder_perceives(episodes, length, epochs, device):
    from torch.utils.data import DataLoader

    cfg = _cfg(episodes, length)
    tr, va = build_datasets(cfg, seed=0)
    m = build_physwm(cfg, 5, 2)
    st, ac = collect_stats(tr)
    m.fit_normalizers(st, ac)
    m.to(device)
    n_tok = m.encoder.num_tokens
    head = torch.nn.Linear(m.encoder.embed_dim * n_tok, 5).to(device)
    opt = torch.optim.AdamW(
        list(m.encoder.parameters()) + list(head.parameters()), lr=3e-4
    )
    for _ in range(epochs):
        m.train()
        for b in DataLoader(tr, batch_size=64, shuffle=True):
            b = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in b.items()}
            z = m.encode(b)[:, :, :-1]      # drop the tactile token
            pred = head(_pool_tokens(z, 'flatten'))
            loss = (pred - m.state_norm.norm(b['state'])).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    m.eval()
    se = torch.zeros(5, device=device)
    n = 0
    with torch.no_grad():
        for b in DataLoader(va, batch_size=64):
            b = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in b.items()}
            z = m.encode(b)[:, :, :-1]
            t = m.state_norm.norm(b['state'])
            se += (head(_pool_tokens(z, 'flatten')) - t).pow(2).sum((0, 1))
            n += t.shape[0] * t.shape[1]
    xy = float(((se[0] + se[1]) / (2 * n)).sqrt())
    # velocity is NOT checked: this read-out sees one frame at a time and
    # velocity needs temporal differencing, so it is unmeasurable here.
    return check(
        'encoder can perceive object position from pixels',
        xy < 0.35,
        f'position RMSE {xy:.4f} normalized (1.0 = predicting the mean)',
    )


def rung6_path_a_predicts(episodes, length, epochs, device):
    from decodability import bench_cfg, train
    from prediction_quality import measure

    cfg = bench_cfg(episodes, length)
    m, _, vl = train(cfg, 0.0, epochs, 0, device)   # alpha=0: Path A alone
    r = measure(m, vl, device)
    return check(
        'Path A beats the trivial baselines',
        r['path_a'] < r['persistence'],
        f"path_a {r['path_a']:.4f} vs persistence {r['persistence']:.4f} "
        f"vs predict_mean {r['predict_mean']:.4f}",
    )


def _cfg(episodes, length):
    from decodability import bench_cfg
    return bench_cfg(episodes, length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    q = args.quick
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    print('\nPhysWM sanity ladder — preconditions for a believable result\n')
    rung1_solver_exact(16 if q else 64, 48)
    rung2_observations_informative(16 if q else 64, 48)
    rung3_normalizer_calibrated(64 if q else 256, 48)
    rung4_theta_identifiable(32 if q else 64, 48, 400 if q else 2000)
    rung5_encoder_perceives(64 if q else 128, 48, 8 if q else 30, dev)
    rung6_path_a_predicts(64 if q else 256, 48, 10 if q else 60, dev)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print('\n' + '=' * 66)
    for n, ok, _ in RESULTS:
        print(f'{"PASS" if ok else "FAIL"}  {n}')
    print('=' * 66)
    if failed:
        print(f'\n{len(failed)} rung(s) FAILED — results above this rung are '
              f'not interpretable:\n  ' + '\n  '.join(failed))
        return 1
    print('\nall rungs pass — the hypothesis is testable')
    return 0


if __name__ == '__main__':
    sys.exit(main())
