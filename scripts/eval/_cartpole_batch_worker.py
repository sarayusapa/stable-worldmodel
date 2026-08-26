#!/usr/bin/env python3
"""Collects a batch of dm_control cartpole episodes as a throwaway subprocess.

Root cause (found via a length/episode-count bisection, then confirmed by
checking dm_control's cartpole task): ``Balance(swing_up=True, ...)``
never terminates on falling, and a purely random action policy driving it
for long enough eventually produces a control sequence whose physics
integration goes unstable. That trips a native SIGABRT deep inside
mj_step -- not a Python exception, not catchable, no traceback -- so the
only way to keep a crash from killing the parent training/eval process is
to run episode collection somewhere disposable. One env is built for the
whole batch (import/compile cost is real); each finished episode is
flushed to its own pickle immediately, so a batch that dies partway
through still leaves every episode collected before the crash on disk for
the caller to pick up, instead of losing the whole batch.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-seed', type=int, required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--length', type=int, required=True)
    ap.add_argument('--frameskip', type=int, default=1)
    ap.add_argument('--image-size', type=int, default=64)
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    import gymnasium as gym

    import stable_worldmodel  # noqa: F401  (registers swm/* ids)
    from stable_worldmodel.wm.physwm.data import _cartpole_state, _resize

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make('swm/CartpoleDMControl-v0')

    for i in range(args.count):
        seed = args.start_seed + i
        out_path = out_dir / f'ep_{seed}.pkl'
        if out_path.exists():
            continue
        env.reset(seed=seed)
        states, actions, frames = [], [], []
        for _ in range(args.length):
            s = _cartpole_state(env)
            a = env.action_space.sample()
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            states.append(s)
            actions.append(a)
            if args.render:
                frame = env.unwrapped.render(
                    width=args.image_size, height=args.image_size
                )
                frame = _resize(np.asarray(frame), args.image_size)
                frames.append(np.asarray(frame, dtype=np.float32) / 255.0)
            for _ in range(args.frameskip):
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    env.reset(seed=seed)
                    break
        ep = {
            'state': np.stack(states).astype(np.float32),
            'action': np.stack(actions),
        }
        if args.render:
            ep['pixels'] = np.stack(frames).transpose(0, 3, 1, 2)
        tmp = out_path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(ep, f)
        tmp.replace(out_path)  # atomic
        print(f'DONE seed={seed}', flush=True)
    env.close()


if __name__ == '__main__':
    sys.exit(main())
