"""Label-free identifiability across benchmarks of differing solver fidelity.

PokeWorld's solver matches its data-generating process exactly and it ships
ground-truth theta. PushT and Cartpole do not: their theta has no recorded
true value, and PushT's solver is a deliberate structural approximation (a
T-block treated as a disc with an isotropic limit-surface mobility). That
makes the R^2 certificate unavailable and forces the question that actually
matters for deployment: does the recovered theta *explain observed
transitions* better than a theta that carries no episode-specific
information?

Three label-free measures, all against the dataset's own next state:

* ``probe``    -- the theta the model inferred for this episode.
* ``shuffled`` -- another episode's inferred theta. If this scores as well,
  theta carries nothing episode-specific.
* ``nominal``  -- the solver's default theta, a constant. The bar any
  per-episode inference must clear.

We also report Path A against a persistence baseline, because a model that
predicts worse than "nothing moves" cannot support any claim about what its
latent encodes.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decodability import bench_cfg, train


def scaled_rmse(pred, target, scale, mask=None):
    d = (pred - target) / scale
    if mask is not None:
        d = d[..., mask]
    return float(d.pow(2).mean().sqrt())


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    S, A, SN, TH, EP = [], [], [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        th = out['theta']
        if th.dim() == 3:
            th = th.mean(1)
        state = batch['state']
        S.append(state[:, :-1]); A.append(batch['action'][:, :-1])
        SN.append(state[:, 1:]); TH.append(th)
        EP.append(batch.get('episode', torch.zeros(len(th), device=device)))
    return (torch.cat(S), torch.cat(A), torch.cat(SN),
            torch.cat(TH), torch.cat(EP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='pokeworld',
                    choices=['pokeworld', 'pusht', 'pusht_rand', 'cartpole'])
    ap.add_argument('--physics-target', default='path_a',
                    choices=['path_a', 'dataset'])
    ap.add_argument('--encoder', default='tiny_cnn')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--episodes', type=int, default=512)
    ap.add_argument('--length', type=int, default=48)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--probe-hidden', type=int, default=0)
    ap.add_argument('--probe-source', default='predicted')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = bench_cfg(episodes=args.episodes, length=args.length,
                    encoder=args.encoder, probe_hidden=args.probe_hidden,
                    probe_source=args.probe_source, window=args.window,
                    benchmark=args.benchmark)
    model, tl, vl = train(cfg, args.alpha, args.epochs, args.seed, device,
                          batch_size=args.batch_size,
                          physics_target=args.physics_target)
    solver = model.solver
    names = list(solver.theta_names)
    s, a, s_next, theta, episode = collect(model, vl, device)
    scale = s.reshape(-1, s.shape[-1]).std(0).clamp_min(1e-6)
    mask = model.state_loss_mask

    def err(th):
        return scaled_rmse(solver(s, a, th.unsqueeze(1).expand(-1, s.shape[1], -1)),
                           s_next, scale, mask)

    g = torch.Generator(device='cpu').manual_seed(args.seed)
    perm = torch.randperm(len(theta), generator=g).to(theta.device)
    nominal = solver.theta_nominal.to(theta.device)
    sub = {'probe': err(theta), 'shuffled': err(theta[perm]),
           'nominal': err(nominal.unsqueeze(0).expand(len(theta), -1))}

    # sensitivity of the solver's output to each theta component
    base = solver(s, a, theta.unsqueeze(1).expand(-1, s.shape[1], -1))
    sscale = base.reshape(-1, base.shape[-1]).std(0).clamp_min(1e-6)
    sens = {}
    for i, n in enumerate(names):
        bumped = theta.clone(); bumped[:, i] *= (1.0 + args.eps)
        sens[n] = scaled_rmse(
            solver(s, a, bumped.unsqueeze(1).expand(-1, s.shape[1], -1)),
            base, sscale, mask)

    # how much of theta's spread is between episodes vs within them
    spread = {n: float(theta[:, i].std()) for i, n in enumerate(names)}

    # Path A vs persistence, in the same units
    with torch.no_grad():
        pa, pers, n = 0.0, 0.0, 0
        for batch in vl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch)
            tgt = out['target']
            m = out.get('state_loss_mask')
            A_ = out['state_a']
            cur = model.state_norm.norm(batch['state'][:, :-1])
            if m is not None:
                A_, tgt, cur = A_[..., m], tgt[..., m], cur[..., m]
            pa += float((A_ - tgt).pow(2).mean()) * len(tgt)
            pers += float((cur - tgt).pow(2).mean()) * len(tgt)
            n += len(tgt)
        pred = {'path_a': (pa / n) ** .5, 'persistence': (pers / n) ** .5}

    gap = sub['shuffled'] - sub['nominal']
    payload = {
        'meta': {'benchmark': args.benchmark, 'physics_target': args.physics_target,
                 'encoder': args.encoder, 'epochs': args.epochs,
                 'episodes': args.episodes, 'length': args.length,
                 'window': args.window, 'seed': args.seed,
                 'theta_names': names,
                 'windows': {'train': len(tl.dataset), 'val': len(vl.dataset)}},
        'substitution': sub,
        'beats_nominal': bool(sub['probe'] < sub['nominal']),
        'beats_shuffled': bool(sub['probe'] < sub['shuffled']),
        'shuffled_minus_probe': sub['shuffled'] - sub['probe'],
        'sensitivity': sens,
        'theta_spread': spread,
        'prediction': pred,
    }
    w = max(len(n) for n in names) + 1
    print(f"\n{args.benchmark} / target={args.physics_target} / seed={args.seed}")
    print(f"  windows: train={len(tl.dataset)} val={len(vl.dataset)}")
    print(f"  Path A {pred['path_a']:.4f} vs persistence {pred['persistence']:.4f}")
    print("  substitution (scaled RMSE vs observed s_next, lower better):")
    for k in ('probe', 'nominal', 'shuffled'):
        print(f"    {k:<10} {sub[k]:.5f}")
    print(f"  probe beats nominal: {payload['beats_nominal']}   "
          f"beats shuffled: {payload['beats_shuffled']}")
    print("  sensitivity:")
    for n in names:
        print(f"    {n:<{w}} {sens[n]:.5f}   (theta sd {spread[n]:.4g})")
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
