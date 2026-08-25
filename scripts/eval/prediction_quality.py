"""Does Path A actually predict the future accurately?

The premise the whole project rests on is that a world model predicts
future states *accurately* while remaining blind to the physics that
produced them. Showing blindness is only half of it: a model that
predicts badly and is also blind to physics demonstrates nothing, because
there is no "accurate but blind" gap to explain.

This script measures the half that was never measured. Path A's next-state
error is compared against two baselines on the same held-out data, in the
same normalized units:

* ``persistence`` -- predict ``s_next = s``. The trivial baseline any
  useful predictor must beat.
* ``mean``        -- predict the per-dimension training mean. Scores ~1.0
  by construction in z-scored units.

A Path A error above ``persistence`` means the model is not a working
world model on this benchmark, and no statement about what its latent
does or does not encode can support the premise.

Run:
    MUJOCO_GL=egl python scripts/eval/prediction_quality.py --epochs 60 --episodes 128
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decodability import bench_cfg, train


@torch.no_grad()
def measure(model, loader, device):
    model.eval()
    se_a = se_b = se_persist = se_mean = n = 0.0
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = model(batch)
        tgt = out['target']                       # normalized s_next
        cur = model.state_norm.norm(batch['state'][:, :-1])   # normalized s
        k = tgt.numel()
        se_a += (out['state_a'] - tgt).pow(2).sum().item()
        se_b += (out['state_b'] - tgt).pow(2).sum().item()
        se_persist += (cur - tgt).pow(2).sum().item()
        # z-scored targets have mean ~0, so "predict the mean" is predict 0
        se_mean += tgt.pow(2).sum().item()
        n += k
    return {
        'path_a': (se_a / n) ** 0.5,
        'path_b': (se_b / n) ** 0.5,
        'persistence': (se_persist / n) ** 0.5,
        'predict_mean': (se_mean / n) ** 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--length', type=int, default=48)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--encoder', default='tiny_cnn',
                    choices=['tiny_cnn', 'dinov2'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    cfg = bench_cfg(args.episodes, args.length, encoder=args.encoder)

    results = {}
    for label, alpha in (('predictive', 0.0), ('physwm', args.alpha)):
        model, _, vl = train(
            cfg, alpha, args.epochs, args.seed, device,
            batch_size=args.batch_size,
        )
        results[label] = measure(model, vl, device)

    print(f'\nNext-state prediction quality — PokeWorld, {args.episodes} '
          f'episodes x {args.length}, {args.epochs} epochs, '
          f'encoder {args.encoder}, seed {args.seed}')
    print('RMSE in normalized (z-scored) state units, val split. '
          'Lower is better.\n')
    rows = ['path_a', 'path_b', 'persistence', 'predict_mean']
    w = 14
    print(f'{"":<{w}}' + ''.join(f'{c:>14}' for c in results))
    for r in rows:
        print(f'{r:<{w}}' + ''.join(f'{results[c][r]:>14.4f}' for c in results))

    print('\nverdict:')
    for c, v in results.items():
        beats = v['path_a'] < v['persistence']
        print(f'  {c:<12} Path A {"BEATS" if beats else "LOSES TO"} '
              f'persistence ({v["path_a"]:.4f} vs {v["persistence"]:.4f})')
    print('\nIf Path A does not beat persistence, the model is not a working '
          'world\nmodel here, and "accurate but blind" has not been shown.')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'meta': vars(args), 'results': results}, indent=2, default=str))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
