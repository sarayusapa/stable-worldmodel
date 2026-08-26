"""Patch for a numpy-scalar / pybind11-enum comparison bug in this exact
mujoco + gymnasium-robotics combination (mujoco==3.12.0,
gymnasium-robotics==1.4.2, verified 2026-08-26 on this pod).

Four sibling functions in ``gymnasium_robotics.utils.mujoco_utils``
(``set_joint_qpos``, ``set_joint_qvel``, ``get_joint_qpos``,
``get_joint_qvel``) all do
``assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)``
where ``joint_type = model.jnt_type[joint_id]`` is a numpy scalar.
Individually, ``joint_type == mujoco.mjtJoint.mjJNT_SLIDE`` correctly
evaluates ``True`` for a genuine slide joint (confirmed directly against
this mujoco build), but the tuple-membership ``in`` check on the SAME
values returns ``False`` -- breaking Fetch env construction entirely
(``gym.make('swm/FetchPush-v3')`` fails inside gymnasium_robotics' own
``_env_setup``/``_get_obs``, before any of our code runs). Casting to
``int()`` first reliably fixes the comparison in every case tested.

This module monkeypatches ONLY the function objects in the
``mujoco_utils`` module namespace (imported by
``gymnasium_robotics.envs.fetch.fetch_env`` as ``self._utils``, looked up
fresh on every call) -- it does not edit the installed package files,
does not change any version, and only affects this process. Importing
``stable_worldmodel.envs.gymnasium_robotics.fetch`` applies it
automatically.
"""

import numpy as np
import mujoco
from gymnasium_robotics.utils import mujoco_utils as _mu

_PATCHED_FLAG = '_swm_int_cast_patch'


def _resolve(model, name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
    return joint_id, int(model.jnt_type[joint_id])


def _set_joint_qpos(model, data, name, value):
    joint_id, joint_type = _resolve(model, name)
    joint_addr = model.jnt_qposadr[joint_id]
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 7
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        ndim = 4
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        ndim = 1
    start_idx, end_idx = joint_addr, joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (end_idx - start_idx), (
            f'Value has incorrect shape {name}: {value}'
        )
    data.qpos[start_idx:end_idx] = value


def _set_joint_qvel(model, data, name, value):
    joint_id, joint_type = _resolve(model, name)
    joint_addr = model.jnt_dofadr[joint_id]
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 6
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        ndim = 3
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        ndim = 1
    start_idx, end_idx = joint_addr, joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (end_idx - start_idx), (
            f'Value has incorrect shape {name}: {value}'
        )
    data.qvel[start_idx:end_idx] = value


def _get_joint_qpos(model, data, name):
    joint_id, joint_type = _resolve(model, name)
    joint_addr = model.jnt_qposadr[joint_id]
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 7
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        ndim = 4
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        ndim = 1
    start_idx, end_idx = joint_addr, joint_addr + ndim
    return data.qpos[start_idx:end_idx].copy()


def _get_joint_qvel(model, data, name):
    joint_id, joint_type = _resolve(model, name)
    joint_addr = model.jnt_dofadr[joint_id]
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 6
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        # NOTE: faithfully replicating the upstream value (4) here, even
        # though every other qvel ndim table uses 3 for BALL -- not
        # touching logic beyond the int-cast fix this module exists for.
        ndim = 4
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        ndim = 1
    start_idx, end_idx = joint_addr, joint_addr + ndim
    return data.qvel[start_idx:end_idx].copy()


_REPLACEMENTS = {
    'set_joint_qpos': _set_joint_qpos,
    'set_joint_qvel': _set_joint_qvel,
    'get_joint_qpos': _get_joint_qpos,
    'get_joint_qvel': _get_joint_qvel,
}

for _name, _fn in _REPLACEMENTS.items():
    _current = getattr(_mu, _name)
    if not getattr(_current, _PATCHED_FLAG, False):
        _fn.__dict__[_PATCHED_FLAG] = True
        setattr(_mu, _name, _fn)
