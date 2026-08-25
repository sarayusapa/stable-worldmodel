"""End-to-end smoke tests for the PhysWM pipeline.

Runs, in order:

1. **unit tests** -- the design invariants (frozen solver, gradients
   through the solver into the probe, theta never a free variable, Path B
   uses Path A's stopped-gradient prediction, exact loss decomposition);
2. **solver validation** -- can each frozen solver fit its environment at
   all? (free-theta oracle vs. persistence baseline);
3. **short training runs** -- every benchmark trains end to end and
   checkpoints;
4. **checkpoint round-trip** -- a saved run reloads through the repo's
   ``load_pretrained``.

Usage:
    MUJOCO_GL=egl python scripts/smoke/run_smoke.py
    MUJOCO_GL=egl python scripts/smoke/run_smoke.py --benchmarks pokeworld
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(name, cmd, env=None):
    """Run a step, stream nothing, report pass/fail and duration."""
    print(f'\n>>> {name}\n    $ {" ".join(cmd)}', flush=True)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.time() - start
    ok = proc.returncode == 0
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-12:]
    for line in tail:
        print(f'    | {line}')
    print(f'    {"PASS" if ok else "FAIL"}  ({dt:.1f}s)', flush=True)
    return name, ok, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--benchmarks',
        nargs='*',
        default=['pokeworld', 'cartpole', 'pusht'],
    )
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--length', type=int, default=24)
    args = ap.parse_args()

    # dm_control needs a GL backend to render; egl works headless
    env = {'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl')}
    results = []
    tmp = Path(tempfile.mkdtemp(prefix='physwm_smoke_'))

    results.append(
        run(
            'unit tests (design invariants)',
            [PY, '-m', 'pytest', 'tests/wm/test_physwm.py', '-q'],
            env,
        )
    )

    results.append(
        run(
            'solver validation (oracle vs persistence)',
            [
                PY,
                'scripts/smoke/validate_solvers.py',
                '--benchmarks',
                *args.benchmarks,
                '--episodes',
                '4',
                '--length',
                '32',
                '--steps',
                '300',
            ],
            env,
        )
    )

    for bench in args.benchmarks:
        results.append(
            run(
                f'train: {bench}',
                [
                    PY,
                    'scripts/train/physwm.py',
                    f'bench={bench}',
                    f'trainer.max_epochs={args.epochs}',
                    f'bench.data.num_episodes={args.episodes}',
                    f'bench.data.episode_length={args.length}',
                    'loader.batch_size=16',
                    f'output_model_name=smoke_physwm_{bench}',
                    f'hydra.run.dir={tmp}/{bench}',
                ],
                env,
            )
        )

    results.append(
        run(
            'checkpoint round-trip',
            [
                PY,
                '-c',
                ('from stable_worldmodel.wm.utils import load_pretrained;'
                 f'm = load_pretrained("smoke_physwm_{args.benchmarks[0]}'
                 '/weights.pt");'
                 'assert sum(p.numel() for p in m.solver.parameters()) == 0;'
                 'print("reloaded", type(m).__name__, "solver params 0")'),
            ],
            env,
        )
    )

    print('\n' + '=' * 62)
    print('SMOKE TEST SUMMARY')
    print('=' * 62)
    for name, ok, dt in results:
        print(f'{"PASS" if ok else "FAIL"}  {name:<45} {dt:>6.1f}s')
    failed = [n for n, ok, _ in results if not ok]
    print('=' * 62)
    if failed:
        print(f'{len(failed)} step(s) FAILED: {", ".join(failed)}')
        return 1
    print(f'all {len(results)} steps passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
