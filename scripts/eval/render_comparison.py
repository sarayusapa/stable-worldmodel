#!/usr/bin/env python3
"""GT vs base-WM (predictive) vs PhysWM: rendered frame + trajectory/error comparison.

Loads (or trains, for cartpole, which has no saved checkpoints) the
`predictive` (alpha=0, no physics grounding) and `physwm` (alpha=1,
physics-grounded) conditions, runs Path A forward on real held-out
windows (which already carry camera-rendered pixels), and picks the
window where the two conditions' one-step prediction error diverges the
most so the comparison actually shows a difference. Saves one labeled
PNG: the real rendered frame, a physical-coordinate trajectory overlay
(true vs predictive vs physwm), and a per-timestep error heatmap.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/workspace/stable-worldmodel/scripts/eval')
from decodability import bench_cfg, train, build_datasets, BENCHMARKS  # noqa: E402
from stable_worldmodel.wm.physwm import build_physwm  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


OBJ_IDX = {
    'fetch_push': (2, 3, 'object'),
    'pusht_rand': (2, 3, 'block'),
}

TITLE = {
    'fetch_push': 'Fetch Push',
    'pusht_rand': 'PushT (randomized physics)',
    'cartpole': 'Cartpole',
}


def cartpole_tip_xy(state):
    x = state[..., 0]
    angle = state[..., 1]
    half_len = 0.6
    tip_x = x + half_len * np.sin(angle)
    tip_y = half_len * np.cos(angle)
    return np.stack([tip_x, tip_y], axis=-1)


def obj_xy(benchmark, state):
    if benchmark == 'cartpole':
        return cartpole_tip_xy(state)
    i, j, _ = OBJ_IDX[benchmark]
    return state[..., [i, j]]


def load_ckpt(benchmark, ckpt_dir, label, device):
    ckpt_dir = Path(ckpt_dir)
    w = ckpt_dir / f'{label}_weights.pt'
    c = ckpt_dir / f'{label}_config.json'
    saved = json.loads(c.read_text())
    cfg = saved['cfg']
    spec = BENCHMARKS[benchmark]
    m = build_physwm(cfg, spec['state_dim'], spec['action_dim'])
    m.load_state_dict(torch.load(w, map_location=device))
    m.to(device).eval()
    return m, cfg


@torch.no_grad()
def path_a_denorm(model, batch, device):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(b)
    return model.state_norm.unnorm(out['state_a']).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True, choices=['fetch_push', 'pusht_rand', 'cartpole'])
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--length', type=int, default=48)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt-dir', default=None)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.ckpt_dir is not None:
        m_pred, cfg = load_ckpt(args.benchmark, args.ckpt_dir, 'predictive', device)
        m_phys, _ = load_ckpt(args.benchmark, args.ckpt_dir, 'physwm', device)
        _, val_set = build_datasets(cfg, seed=args.seed)
        vl = DataLoader(val_set, batch_size=32, shuffle=False)
        print(f'[{args.benchmark}] loaded checkpoints from {args.ckpt_dir}')
    else:
        cfg = bench_cfg(episodes=args.episodes, length=args.length, encoder='tiny_cnn',
                         window=args.window, benchmark=args.benchmark)
        m_pred, _, vl = train(cfg, alpha=0.0, epochs=40, seed=args.seed, device=device, batch_size=32)
        m_phys, _, vl2 = train(cfg, alpha=1.0, epochs=40, seed=args.seed, device=device, batch_size=32)
        print(f'[{args.benchmark}] trained predictive + physwm fresh')

    best = None  # (gap, batch, pred_arr, phys_arr, b_idx)
    for batch in vl:
        if 'pixels' not in batch:
            continue
        pred_arr = path_a_denorm(m_pred, batch, device)   # (B, T-1, S)
        phys_arr = path_a_denorm(m_phys, batch, device)
        true = batch['state'][:, 1:].numpy()               # (B, T-1, S)
        scale = batch['state'].numpy().reshape(-1, true.shape[-1]).std(0).clip(min=1e-6)
        pred_err = np.sqrt((((pred_arr - true) / scale) ** 2).mean(axis=(1, 2)))
        phys_err = np.sqrt((((phys_arr - true) / scale) ** 2).mean(axis=(1, 2)))
        gap = np.abs(phys_err - pred_err)
        b_idx = int(gap.argmax())
        if best is None or gap[b_idx] > best[0]:
            best = (float(gap[b_idx]), batch, pred_arr, phys_arr, b_idx)

    assert best is not None, 'no batch had pixels -- cannot render'
    gap, batch, pred_arr, phys_arr, b_idx = best
    print(f'[{args.benchmark}] most differentiating window: batch item {b_idx}, '
          f'|physwm_err - predictive_err| = {gap:.4f}')

    pixels = batch['pixels'][b_idx].numpy()          # (T, C, H, W)
    true_state = batch['state'][b_idx].numpy()        # (T, S)
    pred_state = pred_arr[b_idx]                       # (T-1, S)
    phys_state = phys_arr[b_idx]                        # (T-1, S)

    frame = pixels[-1].transpose(1, 2, 0)               # (H, W, C)
    frame = np.clip(frame, 0, 1)

    true_xy = obj_xy(args.benchmark, true_state[1:])    # (T-1, 2)
    pred_xy = obj_xy(args.benchmark, pred_state)
    phys_xy = obj_xy(args.benchmark, phys_state)

    scale = true_state.std(0).clip(min=1e-6)
    pred_err_t = np.sqrt((((pred_state - true_state[1:]) / scale) ** 2).mean(-1))
    phys_err_t = np.sqrt((((phys_state - true_state[1:]) / scale) ** 2).mean(-1))

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(frame)
    ax0.set_title(f'{TITLE[args.benchmark]} -- real rendered frame\n(last frame of the most-differentiating window)')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(true_xy[:, 0], true_xy[:, 1], 'o-', color='black', label='ground truth', linewidth=2)
    ax1.plot(pred_xy[:, 0], pred_xy[:, 1], 's--', color='tab:red', label='base WM (predictive, no physics)', alpha=0.85)
    ax1.plot(phys_xy[:, 0], phys_xy[:, 1], '^--', color='tab:blue', label='PhysWM (physics-grounded)', alpha=0.85)
    ax1.set_title(f'object trajectory, one-step Path A rollout (gap={gap:.3f} scaled-RMSE)')
    ax1.legend(fontsize=8, loc='best')
    ax1.set_aspect('equal', adjustable='datalim')

    ax2 = fig.add_subplot(gs[1, 1])
    heat = np.stack([pred_err_t, phys_err_t])
    im = ax2.imshow(heat, aspect='auto', cmap='inferno')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['base WM', 'PhysWM'])
    ax2.set_xlabel('timestep within window')
    ax2.set_title('per-timestep scaled-RMSE error map (brighter = worse)')
    fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)

    fig.suptitle(f'{TITLE[args.benchmark]}: ground truth vs base World Model vs PhysWM', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.benchmark}_gt_vs_basewm_vs_physwm.png'
    fig.savefig(out_path, dpi=150)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
