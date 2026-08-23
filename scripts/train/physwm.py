"""Train PhysWM: a physics-grounded world model with two next-state paths.

    Path A (learned)   z -> predictor -> z_hat -> decoder      -> s_A
    Path B (physical)  z -> probe -> theta -> frozen solver    -> s_B

    L = L_A + alpha * L_B + beta * L_consistency

Both paths are supervised on the dataset's ground-truth ``s_next``; Path B
is never supervised on Path A. See ``stable_worldmodel/wm/physwm/`` for the
design rules and the invariants that enforce them.

Run:

    python scripts/train/physwm.py bench=pokeworld
    python scripts/train/physwm.py bench=cartpole trainer.max_epochs=50
    python scripts/train/physwm.py bench=pusht loss.alpha=0.5 loss.beta=0.1

Note on framework: unlike the other scripts in this directory, this one
uses a plain PyTorch loop rather than lightning + stable-pretraining. The
two-path objective, the theta diagnostics and the frozen-solver
invariants are easier to keep explicit here, and it keeps the prototype
runnable without the training extra. Config, seeding, W&B logging and
checkpointing follow the same conventions as the rest of the repo.
"""

import json
import logging
import math
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import stable_worldmodel as swm
from stable_worldmodel.wm.physwm import (
    BENCHMARKS,
    build_datasets,
    build_physwm,
    collect_stats,
    physwm_loss,
    physwm_metrics,
    theta_r2,
)

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def run_epoch(model, loader, cfg, device, optimizer=None, scheduler=None):
    """One pass. ``optimizer=None`` -> evaluation."""
    train = optimizer is not None
    model.train(train)
    totals, count = {}, 0
    # theta is accumulated over the WHOLE epoch: the identifiability R^2
    # is only meaningful across many episodes, never within one minibatch
    theta_pred, theta_true = [], []

    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(train):
            out = model(batch)
            losses = physwm_loss(
                out,
                alpha=cfg.loss.alpha,
                beta=cfg.loss.beta,
                consistency_detach=cfg.loss.consistency_detach,
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            if cfg.trainer.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.trainer.grad_clip
                )
            optimizer.step()
            if scheduler is not None:
                scheduler()

        metrics = physwm_metrics(out, solver=model.solver)
        if 'theta_true' in batch:
            theta = out['theta']
            if theta.dim() == 3:  # per-step probe -> average over time
                theta = theta.mean(dim=1)
            theta_pred.append(theta.detach().cpu())
            theta_true.append(batch['theta_true'].detach().cpu())
        merged = {**{k: v.detach() for k, v in losses.items()}, **metrics}
        bs = batch['state'].shape[0]
        for k, v in merged.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bs
        count += bs

    stats = {k: v / max(1, count) for k, v in totals.items()}
    if theta_pred:
        stats.update(
            {
                k: float(v)
                for k, v in theta_r2(
                    torch.cat(theta_pred),
                    torch.cat(theta_true),
                    model.solver.theta_names,
                ).items()
            }
        )
    return stats


@hydra.main(version_base=None, config_path='./config', config_name='physwm')
def run(cfg):
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(
        cfg.device
        if cfg.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    log.info(f'device: {device}')

    spec = BENCHMARKS[cfg.bench.benchmark]
    model_cfg = OmegaConf.to_container(cfg.bench, resolve=True)

    # ---------------- data ----------------
    train_set, val_set = build_datasets(model_cfg, seed=cfg.seed)
    log.info(f'windows: train={len(train_set)} val={len(val_set)}')

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        drop_last=len(train_set) > cfg.loader.batch_size,
        generator=gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
    )

    # ---------------- model ----------------
    model = build_physwm(model_cfg, spec['state_dim'], spec['action_dim'])
    states, actions = collect_stats(train_set, cfg.norm_samples)
    model.fit_normalizers(states, actions)
    model.to(device)

    report = model.param_report()
    log.info(f'parameters: {report}')
    assert report['solver'] == 0, 'solver must stay frozen and param-free'

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.wd
    )
    total_steps = cfg.trainer.max_epochs * max(1, len(train_loader))
    warmup = max(1, int(cfg.trainer.warmup_frac * total_steps))
    state = {'step': 0}

    def scheduler():
        state['step'] += 1
        lr = lr_at(state['step'], total_steps, warmup, cfg.optimizer.lr)
        for group in optimizer.param_groups:
            group['lr'] = lr

    # ---------------- logging ----------------
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
        cfg.output_model_name,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    # config.json is written in the format ``swm.wm.load_pretrained``
    # expects, so a checkpoint round-trips through the repo's loader.
    ckpt_config = {
        '_target_': 'stable_worldmodel.wm.physwm.build.build_physwm',
        'cfg': model_cfg,
        'state_dim': spec['state_dim'],
        'action_dim': spec['action_dim'],
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(ckpt_config, f, indent=2)

    # ---------------- train ----------------
    best = float('inf')
    history = []
    for epoch in range(cfg.trainer.max_epochs):
        train_stats = run_epoch(
            model, train_loader, cfg, device, optimizer, scheduler
        )
        val_stats = run_epoch(model, val_loader, cfg, device)

        row = {
            'epoch': epoch,
            **{f'train/{k}': v for k, v in train_stats.items()},
            **{f'val/{k}': v for k, v in val_stats.items()},
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        # where ground-truth physics params exist (PokeWorld), show the
        # identifiability certificate: how much of each true parameter the
        # probe's theta actually explains
        r2 = {
            k.split('/', 1)[1]: v
            for k, v in val_stats.items()
            if k.startswith('r2/') and not math.isnan(v)
        }
        r2_str = (
            '  | R2 ' + ' '.join(f'{k}={v:+.2f}' for k, v in r2.items())
            if r2
            else ''
        )
        log.info(
            f'epoch {epoch:3d} | '
            f'train loss {train_stats["loss"]:.5f} '
            f'(A {train_stats["loss_a"]:.5f} B {train_stats["loss_b"]:.5f}) '
            f'| val loss {val_stats["loss"]:.5f} '
            f'(A {val_stats["loss_a"]:.5f} B {val_stats["loss_b"]:.5f})'
            f'{r2_str}'
        )
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)

        is_best = val_stats['loss'] < best
        if is_best:
            best = val_stats['loss']
        if is_best or (epoch + 1) % cfg.trainer.ckpt_every == 0:
            torch.save(
                model.state_dict(),
                run_dir / ('weights.pt' if is_best else f'ep{epoch + 1}.pt'),
            )

    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump({'best_val_loss': best, **history[-1]}, f, indent=2)
    log.info(f'best val loss {best:.6f}; artifacts in {run_dir}')

    if wandb_run is not None:
        wandb_run.finish()
    return best


if __name__ == '__main__':
    sys.exit(0 if run() is not None else 0)
