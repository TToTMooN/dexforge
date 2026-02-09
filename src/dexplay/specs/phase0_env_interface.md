# Phase-0 Environment Interface (DexPlay)

This document defines the env contract used by training and evaluation.

## Single Env API

`ReorientEnv.reset()` returns:
- `obs: np.ndarray[float32]` shape `(2 * n_joints + 13,)`
- `info: dict` containing `target_quat`

`ReorientEnv.step(action)` returns:
- `obs: np.ndarray[float32]`
- `reward: float`
- `terminated: bool`
- `truncated: bool`
- `info: dict`

When `terminated or truncated` is true, `info` includes:
- `termination_reason` in `{SUCCESS, DROP, OOB, TIMEOUT, NAN}`
- `episode_return`
- `episode_length`
- `final_orientation_error`
- `mean_action_norm`
- `max_action_norm`

## Vector Env API (rsl_rl)

`ReorientVecEnv` implements `rsl_rl.env.VecEnv`:
- `get_observations() -> TensorDict` with groups `policy` and `critic`
- `step(actions) -> (obs, rewards, dones, extras)`

`extras` includes:
- `time_outs: torch.Tensor[num_envs]`
- optional `log` dict with aggregate episode metrics

## Observation Layout

For a robot with `n_joints`:
1. joint positions `(n_joints)`
2. joint velocities `(n_joints)`
3. cube position `(3)`
4. cube quaternion `(4)`
5. cube linear velocity `(3)`
6. cube angular velocity `(3)`

## Action Layout

- `action` shape `(n_joints,)`
- interpreted as joint target deltas
- converted to absolute targets
- clamped and tracked with conservative PD + torque limits
