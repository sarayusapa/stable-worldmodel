"""Train PhysWM: a physics-grounded world model with two next-state paths.

    z_hat = predictor(z, action)
    Path A (learned)   z_hat -> decoder                         -> s_A
    Path B (physical)  z_hat -> probe -> theta -> frozen solver -> s_B

    L = L_A + alpha * L_B + beta * L_consistency

Path A is supervised on the dataset's ground-truth ``s_next``; Path B is
supervised on Path A's own detached prediction, not ``s_next`` directly.
See ``stable_worldmodel/wm/physwm/`` for the design rules and the
invariants that enforce them.

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

import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
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


def to_device(batch, device, non_blocking=False):
    return {
        k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def autocast_context(device, precision):
    """Return the configured mixed-precision context.

    BF16 is intentionally the only mixed mode exposed here: it is native on
    H100, has FP32-like range, and does not require gradient scaling.
    """
    precision = str(precision).lower()
    if precision in {'32', 'fp32', '32-true'}:
        return nullcontext()
    if precision in {'bf16', 'bf16-mixed'}:
        if device.type != 'cuda':
            raise ValueError('BF16 training requires a CUDA device')
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    raise ValueError(
        f'unsupported precision {precision!r}; choose fp32 or bf16'
    )


def atomic_torch_save(obj, path):
    """Write a checkpoint atomically on the run directory filesystem."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


# keys that define the EXPERIMENT. Two runs that differ in any of these are
# different experiments and must not resume from each other's checkpoint.
# Everything else (max_epochs, device, dataloader perf knobs, wandb, resume
# itself) may legitimately change when continuing a run.
_FINGERPRINT_KEYS = (
    'seed',
    'deterministic',
    'norm_samples',
    'bench',
    'loss',
    'optimizer',
)
_FINGERPRINT_TRAINER_KEYS = (
    'grad_clip',
    'warmup_frac',
    'accumulate_grad_batches',
    'precision',
)


def config_fingerprint(cfg) -> str:
    """Stable hash of the experiment-defining parts of the config."""
    payload = {k: OmegaConf.to_container(cfg[k], resolve=True)
               if OmegaConf.is_config(cfg[k]) else cfg[k]
               for k in _FINGERPRINT_KEYS}
    payload['trainer'] = {
        k: cfg.trainer[k] for k in _FINGERPRINT_TRAINER_KEYS
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def git_state():
    """Commit sha and dirtiness, or ``None`` outside a git checkout."""
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip())
        return {'commit': sha, 'dirty': dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def run_metadata(cfg, device) -> dict:
    """Everything needed to reproduce and to log this run in progress.md."""
    meta = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'hostname': platform.node(),
        'python': sys.executable,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'device': str(device),
        'fingerprint': config_fingerprint(cfg),
        'git': git_state(),
        'env': {
            k: os.environ[k]
            for k in ('MUJOCO_GL', 'CUDA_VISIBLE_DEVICES')
            if k in os.environ
        },
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        meta['gpu'] = torch.cuda.get_device_name(device)
    return meta


def run_epoch(model, loader, cfg, device, optimizer=None, scheduler=None):
    """One pass. ``optimizer=None`` -> evaluation."""
    train = optimizer is not None
    accumulate = int(cfg.trainer.get('accumulate_grad_batches', 1))
    if accumulate < 1:
        raise ValueError('trainer.accumulate_grad_batches must be >= 1')
    model.train(train)
    totals, count = {}, 0
    # theta is accumulated over the WHOLE epoch: the identifiability R^2
    # is only meaningful across many episodes, never within one minibatch
    theta_pred, theta_true, theta_episode = [], [], []

    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        batch = to_device(
            batch,
            device,
            non_blocking=bool(cfg.loader.get('pin_memory', False)),
        )
        with (
            torch.set_grad_enabled(train),
            autocast_context(device, cfg.trainer.precision),
        ):
            out = model(batch)
            losses = physwm_loss(
                out,
                alpha=cfg.loss.alpha,
                beta=cfg.loss.beta,
                consistency_detach=cfg.loss.consistency_detach,
                physics_target=cfg.loss.physics_target,
            )

        if train:
            group_start = (batch_idx // accumulate) * accumulate
            group_size = min(accumulate, len(loader) - group_start)
            (losses['loss'] / group_size).backward()
            end_of_group = (
                batch_idx + 1
            ) % accumulate == 0 or batch_idx + 1 == len(loader)
            if end_of_group:
                if cfg.trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.trainer.grad_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler()

        metrics = physwm_metrics(out, solver=model.solver)
        if 'theta_true' in batch:
            theta = out['theta']
            if theta.dim() == 3:  # per-step probe -> average over time
                theta = theta.mean(dim=1)
            theta_pred.append(theta.detach().cpu())
            theta_true.append(batch['theta_true'].detach().cpu())
            theta_episode.append(batch['episode'].detach().cpu())
        merged = {**{k: v.detach() for k, v in losses.items()}, **metrics}
        bs = batch['state'].shape[0]
        for k, v in merged.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bs
        count += bs

    stats = {k: v / max(1, count) for k, v in totals.items()}
    if theta_pred:
        pred = torch.cat(theta_pred)
        true = torch.cat(theta_true)
        episode = torch.cat(theta_episode)
        pred_ep, true_ep = [], []
        for ep in episode.unique(sorted=True):
            keep = episode == ep
            pred_ep.append(pred[keep].mean(0))
            true_ep.append(true[keep][0])
        stats.update(
            {
                k: float(v)
                for k, v in theta_r2(
                    torch.stack(pred_ep),
                    torch.stack(true_ep),
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
    loader_kwargs = {
        'batch_size': cfg.loader.batch_size,
        'num_workers': cfg.loader.num_workers,
        'pin_memory': cfg.loader.pin_memory,
    }
    if cfg.loader.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=cfg.loader.persistent_workers,
            prefetch_factor=cfg.loader.prefetch_factor,
        )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=len(train_set) > cfg.loader.batch_size,
        generator=gen,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_kwargs,
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
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.trainer.accumulate_grad_batches
    )
    total_steps = cfg.trainer.max_epochs * max(1, optimizer_steps_per_epoch)
    warmup = max(1, int(cfg.trainer.warmup_frac * total_steps))
    state = {'step': 0}

    def scheduler():
        state['step'] += 1
        lr = lr_at(state['step'], total_steps, warmup, cfg.optimizer.lr)
        for group in optimizer.param_groups:
            group['lr'] = lr

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

    # provenance: commit, env, exact command. Written every run so a result
    # can always be traced back without relying on shell history.
    meta = run_metadata(cfg, device)
    with open(run_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    git = meta['git']
    log.info(
        f'run {cfg.output_model_name} | fingerprint {meta["fingerprint"]} | '
        f'git {git["commit"][:9] if git else "n/a"}'
        f'{"+dirty" if git and git["dirty"] else ""}'
    )

    # ---------------- resume ----------------
    best = float('inf')
    history = []
    start_epoch = 0
    resume = cfg.trainer.get('resume')
    if resume:
        resume_path = (
            run_dir / 'last.pt' if str(resume) == 'auto' else Path(resume)
        )
        if resume_path.exists():
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            saved_fp = checkpoint.get('fingerprint')
            if saved_fp is not None and saved_fp != meta['fingerprint']:
                raise RuntimeError(
                    f'refusing to resume {resume_path}: it was written by a '
                    f'different experiment (fingerprint {saved_fp} != '
                    f'{meta["fingerprint"]}). The seed, benchmark, loss or '
                    f'optimizer settings changed. Use a different '
                    f'output_model_name, or set trainer.resume=null to start '
                    f'fresh.'
                )
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = int(checkpoint['epoch']) + 1
            best = float(checkpoint['best'])
            history = checkpoint['history']
            state['step'] = int(checkpoint['scheduler_step'])
            log.info(
                f'resumed {resume_path} at epoch {start_epoch} '
                f'(optimizer step {state["step"]})'
            )
        elif str(resume) != 'auto':
            raise FileNotFoundError(
                f'resume checkpoint not found: {resume_path}'
            )
        else:
            log.info(f'no checkpoint at {resume_path}; starting fresh')

    # ---------------- logging ----------------
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb

        wandb_kwargs = {
            'project': cfg.wandb.project,
            'name': cfg.wandb.name,
            'mode': cfg.wandb.mode,
            'config': OmegaConf.to_container(cfg, resolve=True),
        }
        if cfg.wandb.get('id'):
            wandb_kwargs['id'] = cfg.wandb.id
            wandb_kwargs['resume'] = cfg.wandb.resume
        wandb_run = wandb.init(**wandb_kwargs)

    # ---------------- train ----------------
    for epoch in range(start_epoch, cfg.trainer.max_epochs):
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
            atomic_torch_save(
                model.state_dict(),
                run_dir / ('weights.pt' if is_best else f'ep{epoch + 1}.pt'),
            )
        atomic_torch_save(
            {
                'epoch': epoch,
                'fingerprint': meta['fingerprint'],
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_step': state['step'],
                'best': best,
                'history': history,
            },
            run_dir / 'last.pt',
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
    run()
