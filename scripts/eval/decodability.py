"""Is theta in the latent at all, and does the physics path put it there?

This is the control condition behind the claim that the physics path
"recovers the blind parameters where the predictive objective alone does
not". It separates two very different questions:

* **Decodability** -- can a *supervised* read-out recover theta from a
  frozen latent? This is an upper bound on what any probe could extract,
  and it is the number that tells you whether the encoder+predictor
  encoded the physics at all.
* **Induction** -- does PhysWM's own probe, which never sees a theta
  label, land near that bound?

Four conditions are available on the same data, same seed, same
architecture:

1. ``predictive`` -- trained with ``alpha = 0``. Path B is built but
   contributes nothing to the loss, so the latent is shaped by
   next-state prediction alone. This is the baseline the abstract's
   claim rests on.
2. ``physwm``     -- trained with ``alpha > 0``: the physics path shapes
   the shared action-conditioned predictive latent.
3. ``state_target`` -- replaces the model's prediction with the dataset
   next state as Path B's target. This is a target-source ablation.
4. ``theta_oracle`` -- adds direct theta-label supervision. This is not the
   method; it is a diagnostic ceiling.

For 1 and 2 we report both the supervised decodability of the frozen
latent AND the model's own unsupervised probe, so "the physics is in
there but the probe can't find it" is distinguishable from "the physics
was never encoded".

Only PokeWorld has ground-truth theta, so only PokeWorld is supported.

Run:
    MUJOCO_GL=egl python scripts/eval/decodability.py
    MUJOCO_GL=egl python scripts/eval/decodability.py --epochs 30 --episodes 128
"""

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stable_worldmodel.wm.physwm import (
    BENCHMARKS,
    build_datasets,
    build_physwm,
    collect_stats,
    physwm_loss,
    theta_r2,
)


def encoder_cfg(name: str) -> dict:
    """Named encoder presets. ``dinov2`` is frozen, so only the predictor,
    decoder and probe train on top of it."""
    if name == 'tiny_cnn':
        return {
            'name': 'tiny_cnn',
            'image_size': 64,
            'patch_size': 16,
            'embed_dim': 128,
            'width': 64,
        }
    if name == 'dinov2':
        return {
            'name': 'dinov2',
            'model_name': 'facebook/dinov2-small',
            'frozen': True,
            'image_size': 224,
            'patch_size': 14,
        }
    raise KeyError(f'unknown encoder preset {name!r}')


def bench_cfg(
    episodes: int,
    length: int,
    image_size: int = 64,
    encoder: str = 'tiny_cnn',
    probe_hidden: int = 0,
    probe_source: str = 'predicted',
    detach_probe_input: bool = False,
    window: int = 8,
    tactile: bool = True,
    benchmark: str = 'pokeworld',
    quantize_theta: bool = False,
    num_codes: int = 64,
) -> dict:
    """Bench config, inlined so this script is standalone."""
    return {
        'benchmark': benchmark,
        'probe_source': probe_source,
        'probe_context': max(1, window // 2),
        'physics_loss_scope': 'context',
        'probe_frames': 'context',
        'solver_state_source': 'gt',
        'tactile': {'enabled': tactile},
        'encoder': encoder_cfg(encoder),
        'action_encoder': {'emb_dim': 32},
        'predictor': {
            'hidden_dim': 256,
            'depth': 4,
            'heads': 8,
            'dim_head': 32,
            'mlp_dim': 512,
            'dropout': 0.0,
            'max_frames': max(16, window),
        },
        # flatten: mean-pooling patches averages away position, and
        # PokeWorld's state IS position (see bench/pokeworld.yaml)
        'decoder': {
            'hidden_dim': 256,
            'depth': 2,
            'pool': 'mean' if encoder == 'dinov2' else 'flatten',
        },
        'probe': {
            'hidden_dim': probe_hidden,
            'mode': 'episode',
            'pool': 'mean',
            'dropout': 0.0,
            'detach_input': detach_probe_input,
            'init_scale': 0.01,
            'quantize': quantize_theta,
            'num_codes': num_codes,
        },
        # dt/substeps must match each environment's integration, taken
        # from scripts/train/config/bench/<benchmark>.yaml
        'solver': dict(
            {'pokeworld': {'dt': 0.001, 'substeps': 20},
             'pusht': {'dt': 0.01, 'substeps': 10},
             'pusht_rand': {'dt': 0.01, 'substeps': 10},
             'fetch_push': {'dt': 0.01, 'substeps': 10},
             'cartpole': {'dt': 0.01, 'substeps': 2}}[benchmark],
            name=BENCHMARKS[benchmark]['solver'],
        ),
        'data': {
            'num_episodes': episodes,
            'episode_length': length,
            'window': window,
            'stride': 1,
            'require_context_touch': (
                tactile and BENCHMARKS[benchmark].get('tactile_index') is not None
            ),
            'sim': {},
        },
    }


def train(
    cfg,
    alpha,
    epochs,
    seed,
    device,
    lr=3e-4,
    batch_size=32,
    theta_supervision=0.0,
    physics_target='path_a',
    amp=True,
):
    """Train one condition and return the model plus its loaders.

    ``theta_supervision > 0`` adds an explicit regression of the probe's
    theta onto the ground-truth theta. This is NOT part of PhysWM -- it is
    the diagnostic ceiling: if even direct supervision cannot make theta
    recoverable from the latent, the failure is in the encoder or the
    signal, not in the two-path objective.
    """
    torch.manual_seed(seed)
    train_set, val_set = build_datasets(cfg, seed=seed)
    spec = BENCHMARKS[cfg['benchmark']]
    model = build_physwm(cfg, spec['state_dim'], spec['action_dim'])
    states, actions = collect_stats(train_set)
    model.fit_normalizers(states, actions)
    model.to(device)

    tl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    use_amp = amp and device.type == 'cuda'
    theta_lo = model.solver.theta_lo.to(device)
    theta_hi = model.solver.theta_hi.to(device)

    for _ in range(epochs):
        model.train()
        for batch in tl:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            ctx = (
                torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with ctx:
                out = model(batch)
                loss = physwm_loss(
                    out,
                    alpha=alpha,
                    beta=0.0,
                    physics_target=physics_target,
                )['loss']
                if theta_supervision > 0:
                    th = out['theta']
                    if th.dim() == 3:
                        th = th.mean(1)
                    # compare in the bounded box's own units so no single
                    # parameter's scale dominates the regression
                    span = (theta_hi - theta_lo).clamp_min(1e-6)
                    tgt = batch['theta_true']
                    loss = loss + theta_supervision * (
                        ((th - tgt) / span).pow(2).mean()
                    )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, tl, vl


@torch.no_grad()
def pooled_latents(model, loader, device):
    """Frozen latents pooled exactly the way ThetaProbe pools them."""
    model.eval()
    Z, TH, EP = [], [], []
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        if 'theta_true' not in batch:
            raise SystemExit('benchmark has no ground-truth theta')
        h = model.probe_features(batch)
        if h.dim() == 3:  # step-mode probe: one feature per transition
            h = h.mean(dim=1)
        Z.append(h.float().cpu())
        TH.append(batch['theta_true'].float().cpu())
        EP.append(batch['episode'].cpu())
    return average_by_episode(torch.cat(Z), torch.cat(TH), torch.cat(EP))


def average_by_episode(values, theta_true, episode):
    """Return one mean representation/prediction and one label per episode."""
    value_ep, theta_ep = [], []
    for ep in episode.unique(sorted=True):
        keep = episode == ep
        value_ep.append(values[keep].mean(0))
        theta_ep.append(theta_true[keep][0])
    return torch.stack(value_ep), torch.stack(theta_ep)


def ridge_probe(z_tr, th_tr, z_va, th_va, ridge=1e-3):
    """Closed-form supervised linear read-out: the decodability ceiling.

    Standardized inputs, ridge-regularized so a wide latent cannot simply
    memorize the training episodes.
    """
    mu, sd = z_tr.mean(0, keepdim=True), z_tr.std(0, keepdim=True).clamp_min(1e-6)
    a = torch.cat([(z_tr - mu) / sd, torch.ones(len(z_tr), 1)], 1).double()
    b = torch.cat([(z_va - mu) / sd, torch.ones(len(z_va), 1)], 1).double()
    tm, ts = th_tr.mean(0, keepdim=True), th_tr.std(0, keepdim=True).clamp_min(1e-6)
    y = ((th_tr - tm) / ts).double()

    eye = torch.eye(a.shape[1], dtype=torch.float64)
    eye[-1, -1] = 0.0  # never penalize the bias
    w = torch.linalg.solve(a.T @ a + ridge * len(a) * eye, a.T @ y)
    return (b @ w).float() * ts + tm


@torch.no_grad()
def probe_theta(model, loader, device):
    """The model's OWN unsupervised probe output, per window."""
    model.eval()
    P, T, EP = [], [], []
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = model(batch)
        th = out['theta']
        if th.dim() == 3:
            th = th.mean(1)
        P.append(th.float().cpu())
        T.append(batch['theta_true'].float().cpu())
        EP.append(batch['episode'].cpu())
    return average_by_episode(torch.cat(P), torch.cat(T), torch.cat(EP))


@torch.no_grad()
def prediction_metrics(model, loader, device):
    """Premise and fidelity metrics from the same theta-evaluation model."""
    model.eval()
    sums = {
        'path_a_vs_dataset': 0.0,
        'path_a_query_vs_dataset': 0.0,
        'path_b_vs_teacher': 0.0,
        'path_b_vs_dataset': 0.0,
        'persistence_vs_dataset': 0.0,
        'persistence_query_vs_dataset': 0.0,
    }
    n_a = n_b = 0
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = model(batch)
        target = out['target']
        current = model.state_norm.norm(batch['state'][:, :-1])
        mask = out.get('state_loss_mask')
        state_a, state_b = out['state_a'], out['state_b']
        if mask is not None:
            target = target[..., mask]
            current = current[..., mask]
            state_a = state_a[..., mask]
            state_b = state_b[..., mask]
        state_a_phys, state_b_phys, target_phys = state_a, state_b, target
        current_phys = current
        time_mask = out.get('physics_eval_mask')
        if time_mask is not None:
            state_a_phys = state_a[:, time_mask]
            state_b_phys = state_b[:, time_mask]
            target_phys = target[:, time_mask]
            current_phys = current[:, time_mask]
        sums['path_a_vs_dataset'] += (
            state_a - target
        ).pow(2).sum().item()
        sums['path_b_vs_teacher'] += (
            state_b_phys - state_a_phys
        ).pow(2).sum().item()
        sums['path_a_query_vs_dataset'] += (
            state_a_phys - target_phys
        ).pow(2).sum().item()
        sums['path_b_vs_dataset'] += (
            state_b_phys - target_phys
        ).pow(2).sum().item()
        sums['persistence_vs_dataset'] += (
            current - target
        ).pow(2).sum().item()
        sums['persistence_query_vs_dataset'] += (
            current_phys - target_phys
        ).pow(2).sum().item()
        n_a += target.numel()
        n_b += target_phys.numel()
    return {
        'path_a_vs_dataset': (sums['path_a_vs_dataset'] / n_a) ** 0.5,
        'path_a_query_vs_dataset': (
            sums['path_a_query_vs_dataset'] / n_b
        ) ** 0.5,
        'path_b_vs_teacher': (sums['path_b_vs_teacher'] / n_b) ** 0.5,
        'path_b_vs_dataset': (sums['path_b_vs_dataset'] / n_b) ** 0.5,
        'persistence_vs_dataset': (
            sums['persistence_vs_dataset'] / n_a
        ) ** 0.5,
        'persistence_query_vs_dataset': (
            sums['persistence_query_vs_dataset'] / n_b
        ) ** 0.5,
    }


def fmt(r2, names):
    return {n: float(r2[f'r2/{n}']) for n in names}


def dynamics_coordinates(theta, names):
    """Map raw theta to the coordinates identifiable from trajectories.

    PokeWorld motion determines ``k/m`` and ``c/m``; the two radii enter
    only through their sum. Raw mass/stiffness/drag are separately
    identifiable only when touch is observed. Reporting both coordinate
    systems keeps the visual-only control scientifically valid.
    """
    index = {name: i for i, name in enumerate(names)}
    required = {
        'mass', 'contact_stiffness', 'drag',
        'poker_radius', 'object_radius',
    }
    if not required <= set(index):
        return theta, names
    mass = theta[..., index['mass']].clamp_min(1e-8)
    return torch.stack(
        [
            theta[..., index['contact_stiffness']] / mass,
            theta[..., index['drag']] / mass,
            theta[..., index['poker_radius']]
            + theta[..., index['object_radius']],
        ],
        dim=-1,
    ), ['stiffness_over_mass', 'drag_over_mass', 'contact_radius']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--length', type=int, default=48)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--encoder', default='tiny_cnn', choices=['tiny_cnn', 'dinov2'])
    ap.add_argument('--probe-hidden', type=int, default=0)
    ap.add_argument('--probe-source', default='predicted',
                    choices=['predicted', 'encoded'])
    ap.add_argument('--detach-probe-input', action='store_true')
    ap.add_argument(
        '--benchmark', default='pokeworld',
        choices=['pokeworld', 'pusht_rand', 'fetch_push'],
        help='benchmarks carrying a ground-truth theta. `pusht_rand` is '
             'PushT with per-episode physics; only k_p/k_v have a true '
             'value, the remaining theta entries are constant and are '
             'dropped from the R2 report. `fetch_push` is real Fetch '
             'Push with per-episode (mass, friction) -- both have a '
             'true value (the 2-parameter case, no stiffness analog).',
    )
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--no-tactile', action='store_true')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument(
        '--quantize-theta', action='store_true',
        help='route theta through a learned VQ codebook instead of a '
             'continuous linear/MLP head (design-space ablation).',
    )
    ap.add_argument('--num-codes', type=int, default=64)
    ap.add_argument(
        '--theta-weight', type=float, default=100.0,
        help='weight on the theta-supervision term in the `theta_oracle` '
             'condition. Must be large: the term is (err/span)^2 averaged '
             'over 5 params, which lands near 0.02 while the main loss is '
             '~1.4, so a weight of 1 supervises with ~1.5%% of the gradient '
             'and is not a ceiling at all.',
    )
    ap.add_argument(
        '--conditions',
        default='predictive,physwm',
        help='comma-separated: predictive, physwm, state_target, '
             'theta_oracle',
    )
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None, help='write results as JSON here')
    ap.add_argument(
        '--save-dir',
        default=None,
        help='save each trained condition\'s weights + config here',
    )
    args = ap.parse_args()

    # 4090 throughput: TF32 for the fp32 matmuls the autocast leaves alone
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
        quantize_theta=args.quantize_theta,
        num_codes=args.num_codes,
    )
    names = None
    results = {}
    dynamics_results = {}
    prediction = {}
    data_windows = None

    spec = {
        # (physics weight, theta-label weight, physics target)
        'predictive': (0.0, 0.0, 'path_a'),
        'physwm': (args.alpha, 0.0, 'path_a'),
        'state_target': (args.alpha, 0.0, 'dataset'),
        'theta_oracle': (args.alpha, args.theta_weight, 'path_a'),
        # compatibility with historical commands; new sweeps use the
        # unambiguous theta_oracle name.
        'supervised': (args.alpha, args.theta_weight, 'path_a'),
    }
    for label in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        if label not in spec:
            raise SystemExit(f'unknown condition {label!r}; choose from '
                             f'{", ".join(spec)}')
        alpha, sup, physics_target = spec[label]
        model, tl, vl = train(
            cfg,
            alpha,
            args.epochs,
            args.seed,
            device,
            batch_size=args.batch_size,
            theta_supervision=sup,
            physics_target=physics_target,
            amp=not args.no_amp,
        )
        names = list(model.solver.theta_names)
        data_windows = {'train': len(tl.dataset), 'val': len(vl.dataset)}

        z_tr, th_tr = pooled_latents(model, tl, device)
        z_va, th_va = pooled_latents(model, vl, device)
        pred = ridge_probe(z_tr, th_tr, z_va, th_va)
        results[f'{label}/decodable'] = fmt(
            theta_r2(pred, th_va, names), names
        )
        dyn_tr, dyn_names = dynamics_coordinates(th_tr, names)
        dyn_va, _ = dynamics_coordinates(th_va, names)
        dyn_pred = ridge_probe(z_tr, dyn_tr, z_va, dyn_va)
        dynamics_results[f'{label}/decodable'] = fmt(
            theta_r2(dyn_pred, dyn_va, dyn_names), dyn_names
        )

        p_th, p_true = probe_theta(model, vl, device)
        results[f'{label}/own_probe'] = fmt(
            theta_r2(p_th, p_true, names), names
        )
        p_dyn, _ = dynamics_coordinates(p_th, names)
        t_dyn, _ = dynamics_coordinates(p_true, names)
        dynamics_results[f'{label}/own_probe'] = fmt(
            theta_r2(p_dyn, t_dyn, dyn_names), dyn_names
        )
        prediction[label] = prediction_metrics(model, vl, device)

        if args.save_dir:
            d = Path(args.save_dir)
            d.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), d / f'{label}_weights.pt')
            (d / f'{label}_config.json').write_text(
                json.dumps(
                    {'cfg': cfg, 'alpha': alpha, 'theta_supervision': sup,
                     'physics_target': physics_target,
                     'epochs': args.epochs, 'seed': args.seed,
                     'episodes': args.episodes, 'length': args.length,
                     'encoder': args.encoder,
                     'probe_hidden': args.probe_hidden},
                    indent=2, default=str,
                )
            )

    width = max(len(n) for n in names) + 1
    print(f'\nPokeWorld theta recovery -- val R^2 '
          f'({args.episodes} episodes x {args.length}, '
          f'{args.epochs} epochs, seed {args.seed}, '
          f'encoder {args.encoder}, probe source {args.probe_source}, '
          f'probe_hidden {args.probe_hidden})\n')
    print(f'  windows after observability filter: '
          f'{data_windows["train"]} train / {data_windows["val"]} val')
    print('  decodable = supervised ridge read-out on the FROZEN latent')
    print('              (upper bound: is the physics encoded at all?)')
    print('  own probe = the model\'s unsupervised probe through the solver\n')
    cols = list(results)
    col_width = max(22, max(map(len, cols)) + 2)
    print(f'{"param":<{width}}'
          + ''.join(f'{c:>{col_width}}' for c in cols))
    for n in names:
        row = f'{n:<{width}}'
        for c in cols:
            v = results[c][n]
            row += (
                f'{"nan":>{col_width}}'
                if math.isnan(v)
                else f'{v:>{col_width}.4f}'
            )
        print(row)

    print('\nDynamics-coordinate recovery -- val R^2')
    dyn_cols = list(dynamics_results)
    dyn_width = max(len(n) for n in dyn_names) + 1
    dyn_col_width = max(22, max(map(len, dyn_cols)) + 2)
    print(f'{"coordinate":<{dyn_width}}'
          + ''.join(f'{c:>{dyn_col_width}}' for c in dyn_cols))
    for n in dyn_names:
        row = f'{n:<{dyn_width}}'
        for c in dyn_cols:
            v = dynamics_results[c][n]
            row += (
                f'{"nan":>{dyn_col_width}}'
                if math.isnan(v)
                else f'{v:>{dyn_col_width}.4f}'
            )
        print(row)

    print('\nPrediction/fidelity RMSE on the SAME held-out checkpoints')
    metric_names = list(next(iter(prediction.values())))
    print(f'{"metric":<28}' + ''.join(f'{c:>16}' for c in prediction))
    for metric in metric_names:
        print(f'{metric:<28}' + ''.join(
            f'{prediction[c][metric]:>16.4f}' for c in prediction
        ))

    print('\nreading it: if `predictive/decodable` is near zero the '
          'predictive\nobjective never encoded that parameter -- which is '
          'the claim. If it is\nhigh but `predictive/own_probe` is low, the '
          'physics was already there\nand only the read-out was missing.')

    if args.out:
        payload = {
            'meta': {
                'encoder': args.encoder,
                'probe_hidden': args.probe_hidden,
                'probe_source': args.probe_source,
                'detach_probe_input': args.detach_probe_input,
                'window': args.window,
                'tactile': not args.no_tactile,
                'epochs': args.epochs,
                'episodes': args.episodes,
                'length': args.length,
                'alpha': args.alpha,
                'seed': args.seed,
                'batch_size': args.batch_size,
                'conditions': args.conditions,
                'amp': not args.no_amp,
                'windows': data_windows,
            },
            'results': results,
            'dynamics_results': dynamics_results,
            'prediction': prediction,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
