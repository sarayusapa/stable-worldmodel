"""Workshop-paper experiment sweep for the PhysWM claim.

The primary comparison keeps data, architecture and seed fixed while
changing only the training signal:

* ``predictive``: Path A only -- must predict accurately while raw physics
  remains weakly decodable;
* ``physwm``: the proposed stopped-gradient self-distillation through the
  frozen solver, with the probe reading Path A's action-conditioned latent;
* ``theta_oracle``: explicit parameter labels, used only as a ceiling.

The critical ablations test the two phrases that distinguish the method:
"same action-conditioned latent" (predicted vs pre-action encoder latent)
and "model's own prediction" (Path A target vs dataset next state).  Other
runs test whether the loss induces the representation or merely fits a
post-hoc probe, visual-only identifiability, context, probe capacity and
encoder scale.

Scheduling: these models are small (tiny_cnn peaks at 0.54 GiB, frozen
DINOv2-small at 4.14 GiB) against a 24 GB card, so the GPU is idle at
batch 32 and the way to use it is CONCURRENCY, not a bigger batch.
Experiments run in a bounded pool sized by their memory class.

Results land in ``--results-dir`` (JSON metrics, stdout log, and model
weights), and every finished experiment is appended to ``progress.md`` as
it lands, so an interrupted night still leaves a complete record of what
did run.

Run:
    MUJOCO_GL=egl python scripts/eval/overnight.py --dry-run
    MUJOCO_GL=egl python scripts/eval/overnight.py
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PROGRESS = ROOT / 'progress.md'

# one writer at a time: several experiments finish concurrently and each
# appends to the same file
_LOG_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# experiment matrix
# ----------------------------------------------------------------------


def decod(name, *, encoder='tiny_cnn', epochs, episodes, length=48,
          probe_hidden=0, probe_source='predicted', detach_probe=False,
          window=8, tactile=True, alpha=1.0, seed=0, batch=32,
          conditions='predictive,physwm,theta_oracle', group, why):
    return {
        'name': name, 'group': group, 'why': why, 'kind': 'decodability',
        'encoder': encoder, 'heavy': encoder == 'dinov2',
        'args': [
            '--encoder', encoder, '--epochs', str(epochs),
            '--episodes', str(episodes), '--length', str(length),
            '--probe-hidden', str(probe_hidden), '--alpha', str(alpha),
            '--probe-source', probe_source, '--window', str(window),
            '--seed', str(seed), '--batch-size', str(batch),
            '--conditions', conditions,
        ]
        + (['--detach-probe-input'] if detach_probe else [])
        + (['--no-tactile'] if not tactile else []),
    }


def funct(name, *, encoder='tiny_cnn', epochs, episodes, length=48,
          probe_hidden=0, probe_source='predicted', detach_probe=False,
          physics_target='path_a', window=8, tactile=True,
          alpha=1.0, seed=0, batch=32, theta_sup=0.0,
          horizon=7, group, why):
    return {
        'name': name, 'group': group, 'why': why, 'kind': 'functional_use',
        'encoder': encoder, 'heavy': encoder == 'dinov2',
        'args': [
            '--encoder', encoder, '--epochs', str(epochs),
            '--episodes', str(episodes), '--length', str(length),
            '--probe-hidden', str(probe_hidden), '--alpha', str(alpha),
            '--probe-source', probe_source,
            '--physics-target', physics_target, '--window', str(window),
            '--seed', str(seed), '--batch-size', str(batch),
            '--theta-supervision', str(theta_sup), '--horizon', str(horizon),
        ]
        + (['--detach-probe-input'] if detach_probe else [])
        + (['--no-tactile'] if not tactile else []),
    }


def build_matrix(seeds=(0, 1, 2)):
    """Minimal experiment matrix that maps one-to-one to paper claims."""
    exps = []

    # A. Main result: accurate-but-blind baseline vs proposed induction.
    for s in seeds:
        exps.append(decod(
            f'A_claim_tiny_seed{s}', epochs=60, episodes=512, seed=s,
            group='A. primary claim (predictive vs PhysWM vs theta ceiling)',
            why='Same checkpoint reports Path A accuracy, baseline '
                'blindness, PhysWM recovery, and the label-supervised ceiling.',
        ))

    # B. Critical route ablation: pre-action z was the old, misaligned code.
    for s in seeds:
        exps.append(decod(
            f'B_preaction_seed{s}', epochs=60, episodes=512, seed=s,
            probe_source='encoded', conditions='physwm',
            group='B. routing ablation (pre-action encoder latent)',
            why='Tests whether reading the SAME action-conditioned latent '
                'is necessary; this is the route used by the earlier code.',
        ))

    # C. Mechanism controls: teacher source and representation induction.
    exps.append(decod(
        'C_dataset_target_seed0', epochs=60, episodes=512, seed=0,
        conditions='state_target',
        group='C. target ablation (dataset next-state target)',
        why='Replaces the model prediction with the raw state label while '
            'keeping the solver and route fixed.',
    ))
    exps.append(decod(
        'C_posthoc_seed0', epochs=60, episodes=512, seed=0,
        detach_probe=True, conditions='physwm',
        group='C. induction ablation (detached probe input)',
        why='Tests representation induction against fitting only a post-hoc '
            'probe on a latent the physics loss cannot shape.',
    ))

    # D. Visual-only result: raw theta is non-identifiable, ratios are not.
    for s in seeds:
        exps.append(decod(
            f'D_visual_only_seed{s}', epochs=60, episodes=512, seed=s,
            tactile=False,
            group='D. visual-only identifiability',
            why='Matches the visual-encoder claim and reports identifiable '
                'dynamics coordinates k/m and c/m, not misleading raw R2.',
        ))

    # E. The actual frozen self-supervised visual encoder.
    for s in seeds:
        exps.append(decod(
            f'E_dinov2_seed{s}', encoder='dinov2', epochs=40,
            episodes=128, seed=s,
            group='E. frozen DINOv2-small @224',
            why='Runs the full claim on the self-supervised visual encoder '
                'the paper is about, rather than the smoke-test CNN.',
        ))

    # F. Functional certificate: correct theta must improve held-out rollouts.
    for s in seeds:
        exps.append(funct(
            f'F_functional_seed{s}', epochs=60, episodes=512, seed=s,
            window=16,
            group='F. functional use (substitution and rollout)',
            why='Distinguishes physically useful parameters from values that '
                'are merely correlated or decodable.',
        ))
    exps.append(funct(
        'F_functional_dinov2_seed0', encoder='dinov2', epochs=40,
        episodes=128, seed=0, window=16, batch=8,
        group='F. functional use (DINOv2)',
        why='Functional certificate on the paper encoder.',
    ))

    # G. Compact robustness checks; secondary, not substitutes for A-F.
    exps.append(decod(
        'G_probe_mlp_seed0', epochs=60, episodes=512,
        probe_hidden=128, seed=0, conditions='physwm',
        group='G. probe capacity ablation',
        why='Checks that the result is not an artifact of the linear probe.',
    ))
    exps.append(decod(
        'G_context16_seed0', epochs=60, episodes=512,
        window=16, seed=0, conditions='physwm',
        group='G. context-length ablation',
        why='Tests whether longer action-response history strengthens '
            'persistent parameter identification.',
    ))

    return exps


# ----------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------


def run_one(exp, results_dir, timeout, threads=2):
    out_json = results_dir / f'{exp["name"]}.json'
    log_path = results_dir / f'{exp["name"]}.log'
    save_dir = results_dir / 'models' / exp['name']
    cmd = [PY, '-u', str(ROOT / 'scripts' / 'eval' / f'{exp["kind"]}.py')]
    cmd += exp['args'] + ['--out', str(out_json)]
    if exp['kind'] == 'decodability':
        cmd += ['--save-dir', str(save_dir)]

    # Cap intra-op threads. Torch defaults to roughly one thread per core,
    # so a 12-wide pool on 32 cores spawns ~120 runnable threads and
    # thrashes (load average 117 measured). These jobs are GPU-resident and
    # CPU-light, so a couple of threads each is plenty.
    env = {
        **os.environ,
        'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl'),
        'OMP_NUM_THREADS': str(threads),
        'MKL_NUM_THREADS': str(threads),
        'OPENBLAS_NUM_THREADS': str(threads),
    }
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        ok, out = proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        ok = False
        out = (e.stdout or '') + (e.stderr or '') + f'\nTIMEOUT after {timeout}s'
    dt = time.time() - start
    log_path.write_text(out)
    return {
        'exp': exp, 'ok': ok, 'seconds': dt, 'cmd': ' '.join(cmd),
        'json': out_json if out_json.exists() else None,
        'log': log_path,
    }


def git_state():
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return 'unknown', False


def hardware_state():
    """Describe the hardware actually visible to this runner."""
    if torch.cuda.is_available():
        return f'{torch.cuda.get_device_name(0)}, {Path(PY).name}'
    cpu = platform.processor() or platform.machine() or 'unknown CPU'
    return f'CPU ({cpu}), {Path(PY).name}'


def fmt_table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def render_entry(res, sha, dirty):
    """One progress.md entry for a finished experiment."""
    exp = res['exp']
    stamp = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')
    lines = [
        f'### {stamp} — [{exp["group"]}] `{exp["name"]}`',
        '',
        f'- **Command:** `{res["cmd"]}`',
        f'- **Why this run:** {exp["why"]}',
        (f'- **Commit / working tree:** `{sha[:9]}`'
         f'{", dirty" if dirty else ", clean"}'),
        f'- **Hardware:** {res["hardware"]}',
        f'- **Duration:** {res["seconds"] / 60:.1f} min',
        f'- **Status:** {"pass" if res["ok"] else "**FAIL**"}',
        f'- **Artifacts:** `{res["log"].name}`'
        + (f', `{res["json"].name}`' if res['json'] else '')
        + (f', `models/{exp["name"]}/`'
           if exp['kind'] == 'decodability' and res['ok'] else ''),
        '',
    ]

    if not res['ok']:
        tail = res['log'].read_text().strip().splitlines()[-15:]
        lines += ['```', *tail, '```', '']
        lines += ['- **Notes:** run failed; see the log for the traceback.', '']
        return '\n'.join(lines)

    if res['json'] is None:
        lines += ['- **Notes:** completed but wrote no JSON.', '']
        return '\n'.join(lines)

    payload = json.loads(res['json'].read_text())

    if exp['kind'] == 'decodability':
        r = payload['results']
        cols = list(r)
        params = [p for p in r[cols[0]] if r[cols[0]][p] == r[cols[0]][p]]
        rows = [[p] + [f'{r[c][p]:.4f}' for c in cols] for p in params]
        lines += ['**theta recovery — val R²**', '',
                  fmt_table(rows, ['param'] + cols), '',
                  ('_(radii are constant by construction; their R² is `nan` '
                   'and is omitted)_'), '']
        dr = payload.get('dynamics_results', {})
        if dr:
            dcols = list(dr)
            coords = [p for p in dr[dcols[0]]
                      if dr[dcols[0]][p] == dr[dcols[0]][p]]
            drows = [[p] + [f'{dr[c][p]:.4f}' for c in dcols]
                     for p in coords]
            lines += ['**identifiable dynamics coordinates — val R²**', '',
                      fmt_table(drows, ['coordinate'] + dcols), '']
        pred = payload.get('prediction', {})
        if pred:
            pcols = list(pred)
            metrics = list(pred[pcols[0]])
            prows = [[m] + [f'{pred[c][m]:.4f}' for c in pcols]
                     for m in metrics]
            lines += ['**prediction and distillation — val RMSE**', '',
                      fmt_table(prows, ['metric'] + pcols), '']
    else:
        sub = payload['substitution']
        gap = sub['shuffled'] - sub['true']
        closed = (sub['shuffled'] - sub['probe']) / gap if gap else float('nan')
        lines += [
            '**sensitivity** (prediction shift, theta scaled 1.1x): '
            + ', '.join(f'{k} {v:.5f}'
                        for k, v in payload['sensitivity'].items()),
            '',
            '**substitution** (one-step Path B error vs true `s_next`):',
            '',
            fmt_table([[k, f'{sub[k]:.5f}']
                       for k in ('true', 'nominal', 'shuffled', 'probe')],
                      ['source', 'scaled RMSE']),
            '',
            f'probe closes **{closed:.1%}** of the shuffled-to-true gap.',
            '',
        ]
        mh = payload['multi_horizon']
        H = len(mh['true'])
        rows = [[k] + [f'{v:.4f}' for v in mh[k]]
                for k in ('true', 'probe', 'nominal', 'shuffled')]
        lines += ['**multi-horizon rollout** (scaled RMSE):', '',
                  fmt_table(rows, ['source'] + [str(h + 1) for h in range(H)]),
                  '']
    return '\n'.join(lines)


def append_progress(entry):
    with _LOG_LOCK:
        s = PROGRESS.read_text()
        anchor = '## Log\n\n'
        assert anchor in s, 'progress.md lost its "## Log" heading'
        PROGRESS.write_text(s.replace(anchor, '## Log\n\n' + entry + '\n', 1))


def _flatten_metrics(value, prefix='', out=None):
    """Flatten nested result dictionaries into stable slash-separated keys."""
    out = {} if out is None else out
    if isinstance(value, dict):
        for key, child in value.items():
            name = f'{prefix}/{key}' if prefix else str(key)
            _flatten_metrics(child, name, out)
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            _flatten_metrics(child, f'{prefix}/h{index}', out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            out[prefix] = number
    return out


def write_summary(results_dir, experiments):
    """Aggregate repeated seeds into JSON and a paper-ready Markdown table."""
    grouped = {}
    metric_roots = (
        'results', 'dynamics_results', 'prediction',
        'sensitivity', 'substitution', 'multi_horizon',
    )
    for exp in experiments:
        path = results_dir / f'{exp["name"]}.json'
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metrics = {}
        for root in metric_roots:
            if root in payload:
                _flatten_metrics(payload[root], root, metrics)
        bucket = grouped.setdefault(exp['group'], {})
        for name, value in metrics.items():
            bucket.setdefault(name, []).append(value)

    summary = {}
    rows = []
    for group, metrics in grouped.items():
        summary[group] = {}
        for name, values in sorted(metrics.items()):
            item = {
                'n': len(values),
                'mean': statistics.fmean(values),
                'std': statistics.stdev(values) if len(values) > 1 else None,
                'values': values,
            }
            summary[group][name] = item
            std = '—' if item['std'] is None else f'{item["std"]:.6f}'
            rows.append([
                group, name, str(item['n']), f'{item["mean"]:.6f}', std,
            ])

    (results_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    md = [
        '# Workshop experiment summary',
        '',
        ('Mean and sample standard deviation across completed runs in each '
         'experiment group. Single-run ablations report `—` for std.'),
        '',
        fmt_table(rows, ['group', 'metric', 'n', 'mean', 'std']),
        '',
    ]
    (results_dir / 'summary.md').write_text('\n'.join(md))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--light-workers', type=int, default=4,
                    help='concurrent tiny_cnn runs (0.54 GiB each)')
    ap.add_argument('--heavy-workers', type=int, default=2,
                    help='concurrent DINOv2 runs (4.14 GiB each)')
    ap.add_argument('--timeout', type=int, default=6 * 3600)
    ap.add_argument('--threads', type=int, default=2,
                    help='intra-op threads per experiment process')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help='2 tiny experiments, to check the harness itself')
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(','))
    if args.smoke:
        exps = [
            decod('SMOKE_decod', epochs=2, episodes=16, seed=0,
                  conditions='physwm,supervised',
                  group='SMOKE', why='harness check, not a result'),
            funct('SMOKE_funct', epochs=2, episodes=16, seed=0,
                  group='SMOKE', why='harness check, not a result'),
        ]
    else:
        exps = build_matrix(seeds)
    results_dir = Path(
        args.results_dir
        or ROOT / 'outputs' / 'overnight'
        / datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M')
    )

    heavy = [e for e in exps if e['heavy']]
    light = [e for e in exps if not e['heavy']]
    print(f'{len(exps)} experiments '
          f'({len(light)} light, {len(heavy)} heavy) -> {results_dir}')
    for e in exps:
        print(f'  {e["name"]:26s} {e["kind"]:14s} {" ".join(e["args"])}')
    if args.dry_run:
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_state()
    hardware = hardware_state()
    (results_dir / 'matrix.json').write_text(json.dumps(exps, indent=2))

    done, failed = 0, 0
    t0 = time.time()

    def run_pool(pool, workers, label):
        nonlocal done, failed
        if not pool:
            return
        print(f'\n=== {label}: {len(pool)} experiments, {workers} at a time')
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    run_one, e, results_dir, args.timeout, args.threads
                ): e
                for e in pool
            }
            for fut in as_completed(futs):
                res = fut.result()
                res['hardware'] = hardware
                done += 1
                failed += 0 if res['ok'] else 1
                append_progress(render_entry(res, sha, dirty))
                print(f'[{done}/{len(exps)}] '
                      f'{"PASS" if res["ok"] else "FAIL"} '
                      f'{res["exp"]["name"]} ({res["seconds"] / 60:.1f} min)',
                      flush=True)

    # light first: they are the decisive ones and they finish fastest
    run_pool(light, args.light_workers, 'light (tiny_cnn)')
    run_pool(heavy, args.heavy_workers, 'heavy (DINOv2)')
    summary_rows = write_summary(results_dir, exps)

    print(f'\nall done in {(time.time() - t0) / 3600:.2f} h; '
          f'{done - failed} passed, {failed} failed')
    print(f'results in {results_dir}')
    print(f'aggregate summary: {summary_rows} metrics in summary.md/summary.json')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
