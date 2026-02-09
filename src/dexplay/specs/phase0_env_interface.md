# Phase-0 Environment Interface (DexPlay)

This document defines the Phase-0 task contract for `allegro_xhand_reorient`.

## Core Env

Primary env class: `ReorientMjlabEnv` (`mjlab` manager-based RL env).

Step API:
- `obs: dict[str, torch.Tensor]`
- `reward: torch.Tensor[num_envs]`
- `terminated: torch.Tensor[num_envs]`
- `truncated: torch.Tensor[num_envs]`
- `extras: dict`

`ReorientEnv` provides a single-env numpy adapter and returns:
- `obs: np.ndarray[float32]`
- `reward: float`
- `terminated: bool`
- `truncated: bool`
- `info: dict` with `termination_reason` on done

## Observation Layout

Per environment (`policy` and `critic` groups use this same flat vector):
1. joint positions `(n_joints)`
2. joint velocities `(n_joints)`
3. cube position `(3)`
4. cube quaternion `(4)`
5. cube linear velocity `(3)`
6. cube angular velocity `(3)`

Total dimension: `2 * n_joints + 13`.

## Action Layout

- Action shape: `(n_joints,)`
- Semantics: joint position target deltas
- Internal control: `JointPositionAction` with conservative actuator limits + clamps

## Termination Reasons

Logged termination taxonomy:
- `SUCCESS`
- `DROP`
- `OOB`
- `TIMEOUT`
- `NAN`

`ReorientVecEnv` maps manager termination terms to these labels and writes per-episode JSONL
records in train/eval runs.
