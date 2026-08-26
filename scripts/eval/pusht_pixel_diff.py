#!/usr/bin/env python3
"""PushT: real pixel-diff error maps -- render predicted states through the
actual simulator and subtract from the true rendered frame."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/workspace/stable-worldmodel/scripts/eval')
from decodability import bench_cfg, build_datasets, BENCHMARKS  # noqa: E402
from stable_worldmodel.wm.physwm import build_physwm  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

CKPT_DIR = Path('/workspace/physwm-artifacts/runs/pusht-cartpole-matrix/models/pusht_rand_decod_seed0')
OUT_DIR = Path('/workspace/physwm-artifacts/comparison_figures')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ckpt(label):
    saved = json.loads((CKPT_DIR / f'{label}_config.json').read_text())
    cfg = saved['cfg']
    spec = BENCHMARKS['pusht_rand']
    m = build_physwm(cfg, spec['state_dim'], spec['action_dim'])
    m.load_state_dict(torch.load(CKPT_DIR / f'{label}_weights.pt', map_location=device))
    m.to(device).eval()
    return m, cfg


@torch.no_grad()
def path_a(model, batch):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(b)
    return model.state_norm.unnorm(out['state_a']).cpu().numpy()


def main():
    m_pred, cfg = load_ckpt('predictive')
    m_phys, _ = load_ckpt('physwm')
    cfg = dict(cfg)
    cfg['episodes'] = 32  # small + fast, just for this figure -- model itself is unaffected
    _, val_set = build_datasets(cfg, seed=0)
    vl = DataLoader(val_set, batch_size=32, shuffle=False)

    best = None
    for batch in vl:
        if 'pixels' not in batch:
            continue
        pred_arr = path_a(m_pred, batch)
        phys_arr = path_a(m_phys, batch)
        true = batch['state'][:, 1:].numpy()
        scale = batch['state'].numpy().reshape(-1, true.shape[-1]).std(0).clip(min=1e-6)
        pred_err = np.sqrt((((pred_arr - true) / scale) ** 2).mean(axis=(1, 2)))
        phys_err = np.sqrt((((phys_arr - true) / scale) ** 2).mean(axis=(1, 2)))
        gap = np.abs(phys_err - pred_err)
        b_idx = int(gap.argmax())
        if best is None or gap[b_idx] > best[0]:
            best = (float(gap[b_idx]), batch, pred_arr, phys_arr, b_idx)

    gap, batch, pred_arr, phys_arr, b_idx = best
    print(f'most-differentiating window: gap={gap:.3f}')
    pixels = batch['pixels'][b_idx].numpy()       # (T, C, H, W)
    true_state = batch['state'][b_idx].numpy()     # (T, S)
    pred_state = pred_arr[b_idx]                    # (T-1, S)
    phys_state = phys_arr[b_idx]

    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    env = gym.make('swm/PushT-v1', resolution=64)

    def render_state(state7):
        env.reset(seed=0, options={'state': np.asarray(state7, dtype=np.float64)})
        frame = env.render()
        return np.asarray(frame, dtype=np.float32) / 255.0

    T = true_state.shape[0] - 1
    gt_frames = np.stack([pixels[t + 1].transpose(1, 2, 0) for t in range(T)])
    pred_frames = np.stack([render_state(pred_state[t]) for t in range(T)])
    phys_frames = np.stack([render_state(phys_state[t]) for t in range(T)])
    env.close()

    phys_diff = np.abs(gt_frames - phys_frames).sum(-1)   # (T, H, W)
    vmax = max(phys_diff.max(), 1e-6)

    ncols = min(T, 8)
    fig, axes = plt.subplots(3, ncols, figsize=(2.1 * ncols, 6.6))
    for c in range(ncols):
        t = int(round(c * (T - 1) / max(ncols - 1, 1)))
        axes[0, c].imshow(np.clip(gt_frames[t], 0, 1)); axes[0, c].axis('off')
        axes[1, c].imshow(np.clip(phys_frames[t], 0, 1)); axes[1, c].axis('off')
        im = axes[2, c].imshow(phys_diff[t], cmap='inferno', vmin=0, vmax=vmax); axes[2, c].axis('off')
        axes[0, c].set_title(f't={t}', fontsize=9)

    axes[0, 0].set_ylabel('ground truth', fontsize=10)
    axes[1, 0].set_ylabel('PhysWM (re-rendered)', fontsize=10)
    axes[2, 0].set_ylabel('|diff| error map', fontsize=10)
    for r, label in enumerate(['ground truth\n(real render)', 'PhysWM\n(re-rendered)', '|GT - PhysWM|\nerror map']):
        axes[r, 0].axis('on')
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center')

    fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.015, pad=0.01,
                 label='|GT pixel - predicted pixel| (summed over RGB)')
    fig.suptitle('PushT: ground truth vs PhysWM re-rendered predicted state, timestamp-by-timestamp pixel-diff error maps',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0, 1, 0.94])
    out_path = OUT_DIR / 'pusht_rand_pixel_diff_error_maps.png'
    fig.savefig(out_path, dpi=150)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
