from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from dexplay.envs.robots.common_robot_cfg import RobotCfg
from dexplay.envs.tasks.allegro_xhand_reorient.config import ReorientTaskCfg
from dexplay.utils.logging import EpisodeRecord, JsonlEpisodeLogger
from dexplay.utils.paths import eval_log_path, train_log_path

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - import guard for runtime env setup
    raise ImportError(
        "mujoco is required for dexplay Phase-0. Install dependencies with `pixi install`."
    ) from exc


TERMINATION_SUCCESS = "SUCCESS"
TERMINATION_DROP = "DROP"
TERMINATION_OOB = "OOB"
TERMINATION_TIMEOUT = "TIMEOUT"
TERMINATION_NAN = "NAN"


@dataclass
class EpisodeStats:
    episode_return: float = 0.0
    episode_length: int = 0
    action_norm_sum: float = 0.0
    max_action_norm: float = 0.0


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(q)
    if denom < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / denom


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(angle) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    half = 0.5 * angle
    s = math.sin(half)
    return _normalize_quat(np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]))


def _quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    q = _normalize_quat(q)
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    axis = q[1:] / s
    return axis.astype(np.float64), angle


def _quat_angle_distance(q: np.ndarray, q_target: np.ndarray) -> float:
    dq = _normalize_quat(_quat_multiply(q_target, _quat_conjugate(q)))
    angle = 2.0 * math.acos(float(np.clip(abs(dq[0]), -1.0, 1.0)))
    return min(angle, 2.0 * math.pi - angle)


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    axis = _random_unit_vector(rng)
    angle = rng.uniform(-math.pi, math.pi)
    return _axis_angle_to_quat(axis, angle)


def _quat_perturb(base_quat: np.ndarray, max_angle: float, rng: np.random.Generator) -> np.ndarray:
    axis = _random_unit_vector(rng)
    angle = rng.uniform(-max_angle, max_angle)
    return _normalize_quat(_quat_multiply(_axis_angle_to_quat(axis, angle), base_quat))


class ReorientEnv:
    """Single-environment Phase-0 in-hand reorientation task."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        task_cfg: ReorientTaskCfg,
        seed: int,
        backend: Literal["auto", "mjlab", "mujoco"] = "auto",
        debug: bool = False,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.task_cfg = task_cfg
        self.seed = seed
        self.debug = debug
        self.rng = np.random.default_rng(seed)

        self.backend = self._resolve_backend(backend)

        self.model = mujoco.MjModel.from_xml_path(str(robot_cfg.mjcf_path))
        self.model.opt.timestep = task_cfg.sim_dt
        self.data = mujoco.MjData(self.model)

        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in robot_cfg.joint_names
            ],
            dtype=np.int32,
        )
        self._joint_qpos_ids = self.model.jnt_qposadr[self._joint_ids].astype(np.int32)
        self._joint_dof_ids = self.model.jnt_dofadr[self._joint_ids].astype(np.int32)

        self._palm_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            robot_cfg.palm_site_name,
        )

        cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            robot_cfg.cube_joint_name,
        )
        self._cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint_id])
        self._cube_dof_adr = int(self.model.jnt_dofadr[cube_joint_id])

        self._joint_lower = robot_cfg.joint_lower_np()
        self._joint_upper = robot_cfg.joint_upper_np()
        self._default_qpos = robot_cfg.default_qpos_np()
        self._kp = robot_cfg.kp_np()
        self._kd = robot_cfg.kd_np()
        self._torque_lower = robot_cfg.torque_lower_np()
        self._torque_upper = robot_cfg.torque_upper_np()

        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.success_streak = 0
        self.stats = EpisodeStats()
        self.global_step = 0
        self._last_obs = np.zeros(self.observation_dim, dtype=np.float32)

        self.reset()

    @property
    def action_dim(self) -> int:
        return self.robot_cfg.action_dim

    @property
    def observation_dim(self) -> int:
        return 2 * self.action_dim + 13

    @property
    def episode_length(self) -> int:
        return self.stats.episode_length

    def _resolve_backend(self, backend: str) -> str:
        if backend == "mujoco":
            return "mujoco"
        if backend not in {"auto", "mjlab"}:
            raise ValueError(f"Unsupported backend '{backend}'. Use auto|mjlab|mujoco")

        try:
            import mjlab  # noqa: F401

            warnings.warn(
                "mjlab detected. Phase-0 currently executes the stable MuJoCo CPU backend while "
                "keeping a manager-style env API for future mjlab migration.",
                stacklevel=2,
            )
            return "mujoco"
        except Exception:
            if backend == "mjlab":
                warnings.warn(
                    "Requested backend=mjlab, but mjlab import failed. Falling back to mujoco.",
                    stacklevel=2,
                )
            return "mujoco"

    def _cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3]

    def _cube_quat(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7]

    def _set_cube_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = pos
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = _normalize_quat(quat)

    def _sample_target_quat(self) -> np.ndarray:
        return _random_quat(self.rng)

    def _workspace_violation(self, cube_pos: np.ndarray) -> bool:
        low = np.asarray(self.task_cfg.workspace_low, dtype=np.float64)
        high = np.asarray(self.task_cfg.workspace_high, dtype=np.float64)
        return bool(np.any(cube_pos < low) or np.any(cube_pos > high))

    def _safe_obs(self, obs: np.ndarray) -> np.ndarray:
        if np.isfinite(obs).all():
            return obs.astype(np.float32)
        return np.zeros_like(obs, dtype=np.float32)

    def _collect_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._joint_qpos_ids]
        qvel = self.data.qvel[self._joint_dof_ids]

        cube_pos = self._cube_pos()
        cube_quat = _normalize_quat(self._cube_quat())
        cube_vel = self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 3]
        cube_ang = self.data.qvel[self._cube_dof_adr + 3 : self._cube_dof_adr + 6]

        obs = np.concatenate(
            [qpos, qvel, cube_pos, cube_quat, cube_vel, cube_ang],
            axis=0,
        ).astype(np.float32)
        self._last_obs = obs
        return obs

    def _apply_pd_control(self, target_qpos: np.ndarray) -> None:
        for _ in range(self.task_cfg.frame_skip):
            q = self.data.qpos[self._joint_qpos_ids]
            qd = self.data.qvel[self._joint_dof_ids]
            tau = self._kp * (target_qpos - q) - self._kd * qd
            tau = np.clip(tau, self._torque_lower, self._torque_upper)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self._joint_dof_ids] = tau
            mujoco.mj_step(self.model, self.data)

    def _apply_cube_assist(self, action: np.ndarray) -> None:
        cube_quat = _normalize_quat(self._cube_quat())
        err_quat = _normalize_quat(_quat_multiply(self.target_quat, _quat_conjugate(cube_quat)))
        err_axis, _ = _quat_to_axis_angle(err_quat)

        action_energy = float(np.linalg.norm(action) / math.sqrt(float(action.size)))
        assist_angle = (
            self.task_cfg.orientation_assist_gain * action_energy * self.task_cfg.control_dt
        )
        assist_angle = min(assist_angle, self.task_cfg.max_assist_angle_per_step)

        delta_quat = _axis_angle_to_quat(err_axis, assist_angle)
        new_quat = _normalize_quat(_quat_multiply(delta_quat, cube_quat))

        palm_pos = self.data.site_xpos[self._palm_site_id]
        anchor_pos = np.array(
            [palm_pos[0], palm_pos[1], palm_pos[2] + self.task_cfg.cube_anchor_height],
            dtype=np.float64,
        )

        cube_pos = self._cube_pos().copy()
        cube_pos += self.task_cfg.cube_anchor_gain * (anchor_pos - cube_pos)

        slip = max(action_energy - self.task_cfg.slip_action_threshold, 0.0)
        if slip > 0.0:
            axis_hint = np.array(
                [
                    float(np.mean(action[0::3])) if action.size >= 1 else 0.0,
                    float(np.mean(action[1::3])) if action.size >= 2 else 0.0,
                    float(np.mean(action[2::3])) if action.size >= 3 else 0.0,
                ],
                dtype=np.float64,
            )
            dir_xy = axis_hint[:2]
            norm_xy = np.linalg.norm(dir_xy)
            if norm_xy > 1e-8:
                dir_xy = dir_xy / norm_xy
            cube_pos[0:2] += self.task_cfg.slip_xy_gain * slip * dir_xy
            cube_pos[2] -= self.task_cfg.slip_z_gain * slip

        self._set_cube_pose(cube_pos, new_quat)
        self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 3] *= 0.2
        self.data.qvel[self._cube_dof_adr + 3 : self._cube_dof_adr + 6] = (
            err_axis * assist_angle / max(self.task_cfg.control_dt, 1e-6)
        )
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> tuple[np.ndarray, dict]:
        mujoco.mj_resetData(self.model, self.data)

        joint_noise = self.rng.normal(0.0, self.task_cfg.joint_noise_std, size=self.action_dim)
        q0 = np.clip(self._default_qpos + joint_noise, self._joint_lower, self._joint_upper)
        self.data.qpos[self._joint_qpos_ids] = q0
        self.data.qvel[self._joint_dof_ids] = self.rng.normal(0.0, 0.02, size=self.action_dim)

        mujoco.mj_forward(self.model, self.data)
        palm_pos = self.data.site_xpos[self._palm_site_id].copy()

        cube_pos = palm_pos + np.array(
            [0.0, 0.0, self.task_cfg.cube_anchor_height], dtype=np.float64
        )
        cube_pos += self.rng.normal(0.0, self.task_cfg.cube_pos_noise, size=3)

        cube_quat = _quat_perturb(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            self.task_cfg.cube_quat_noise_rad,
            self.rng,
        )
        self._set_cube_pose(cube_pos, cube_quat)
        self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

        self.target_quat = self._sample_target_quat()
        self.success_streak = 0
        self.stats = EpisodeStats()

        mujoco.mj_forward(self.model, self.data)
        obs = self._safe_obs(self._collect_obs())
        return obs, {"target_quat": self.target_quat.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Action size mismatch: got {action.size}, expected {self.action_dim}")

        terminated = False
        truncated = False
        reason: str | None = None

        if not np.isfinite(action).all():
            action = np.zeros_like(action)
            terminated = True
            reason = TERMINATION_NAN

        action = np.clip(action, -1.0, 1.0)
        action_norm = float(np.linalg.norm(action))

        if reason is None:
            delta = np.clip(
                action * self.robot_cfg.action_scale,
                -self.robot_cfg.target_delta_clip,
                self.robot_cfg.target_delta_clip,
            )
            q = self.data.qpos[self._joint_qpos_ids].copy()
            target = np.clip(q + delta, self._joint_lower, self._joint_upper)

            self._apply_pd_control(target)
            self._apply_cube_assist(action)

        obs = self._collect_obs()
        cube_pos = self._cube_pos().copy()
        cube_quat = _normalize_quat(self._cube_quat().copy())
        palm_pos = self.data.site_xpos[self._palm_site_id].copy()

        ori_err = _quat_angle_distance(cube_quat, self.target_quat)
        palm_dist = float(np.linalg.norm(cube_pos - palm_pos))

        reward = -self.task_cfg.reward_ori_scale * ori_err
        reward -= self.task_cfg.reward_dist_scale * palm_dist
        reward -= self.task_cfg.reward_action_scale * action_norm

        if np.isfinite(obs).all() and np.isfinite(reward):
            if ori_err < self.task_cfg.orientation_tolerance_rad:
                self.success_streak += 1
            else:
                self.success_streak = 0
        else:
            terminated = True
            reason = TERMINATION_NAN

        self.stats.episode_return += float(reward)
        self.stats.episode_length += 1
        self.stats.action_norm_sum += action_norm
        self.stats.max_action_norm = max(self.stats.max_action_norm, action_norm)
        self.global_step += 1

        if reason is None and self.success_streak >= self.task_cfg.success_hold_steps:
            reward += self.task_cfg.success_bonus
            terminated = True
            reason = TERMINATION_SUCCESS

        if reason is None and (
            cube_pos[2] < self.task_cfg.drop_z_threshold
            or np.linalg.norm(cube_pos - palm_pos) > self.task_cfg.drop_distance_threshold
        ):
            terminated = True
            reason = TERMINATION_DROP

        if reason is None and self._workspace_violation(cube_pos):
            terminated = True
            reason = TERMINATION_OOB

        if reason is None and self.stats.episode_length >= self.task_cfg.max_steps:
            truncated = True
            reason = TERMINATION_TIMEOUT

        if reason is None and (not np.isfinite(obs).all() or not np.isfinite(reward)):
            terminated = True
            reason = TERMINATION_NAN

        if self.debug and (self.global_step % self.task_cfg.debug_print_every == 0):
            print(
                f"[debug] step={self.global_step} err={ori_err:.3f} "
                f"dist={palm_dist:.3f} rew={reward:.3f}"
            )

        info: dict = {}
        if terminated or truncated:
            mean_action = self.stats.action_norm_sum / max(self.stats.episode_length, 1)
            info = {
                "termination_reason": reason,
                "episode_return": float(self.stats.episode_return),
                "episode_length": int(self.stats.episode_length),
                "final_orientation_error": float(ori_err),
                "mean_action_norm": float(mean_action),
                "max_action_norm": float(self.stats.max_action_norm),
            }

        return self._safe_obs(obs), float(reward), terminated, truncated, info

    def close(self) -> None:
        return


class ReorientVecEnv(VecEnv):
    """rsl_rl-compatible vectorized wrapper around `ReorientEnv`."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        task_cfg: ReorientTaskCfg,
        num_envs: int,
        seed: int,
        run_name: str,
        split: Literal["train", "eval"],
        backend: Literal["auto", "mjlab", "mujoco"] = "auto",
        debug: bool = False,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.task_cfg = task_cfg
        self.num_envs = int(num_envs)
        self.num_actions = robot_cfg.action_dim
        self.max_episode_length = task_cfg.max_steps
        self.device = torch.device("cpu")

        self.cfg = {
            "robot": robot_cfg.name,
            "backend": backend,
            "sim_dt": task_cfg.sim_dt,
            "control_hz": task_cfg.control_hz,
        }

        log_path = train_log_path(run_name) if split == "train" else eval_log_path(run_name)
        self.logger = JsonlEpisodeLogger(log_path)
        self.run_name = run_name
        self.split = split
        self.seed = seed

        self.envs = [
            ReorientEnv(
                robot_cfg=robot_cfg,
                task_cfg=task_cfg,
                seed=seed + (i * 1009),
                backend=backend,
                debug=debug and i == 0,
            )
            for i in range(self.num_envs)
        ]

        obs0, _ = self.envs[0].reset()
        self.num_obs = int(obs0.shape[0])
        self.obs_buf = np.zeros((self.num_envs, self.num_obs), dtype=np.float32)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._timeouts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._episode_counter = 0
        self._recent_records: list[EpisodeRecord] = []

        self.reset()

    def _tensor_obs(self) -> TensorDict:
        obs_t = torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "policy": obs_t,
                "critic": obs_t,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self) -> TensorDict:
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self.obs_buf[i] = obs
            self.episode_length_buf[i] = 0
        self.done_buf.zero_()
        self._timeouts.zero_()
        return self._tensor_obs()

    def get_observations(self) -> TensorDict:
        return self._tensor_obs()

    def consume_recent_records(self) -> list[EpisodeRecord]:
        out = self._recent_records
        self._recent_records = []
        return out

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        act = actions.detach().to("cpu").numpy()
        if act.shape != (self.num_envs, self.num_actions):
            expected_shape = (self.num_envs, self.num_actions)
            raise ValueError(
                f"Action tensor shape mismatch: got {act.shape}, expected {expected_shape}"
            )

        self._timeouts.zero_()
        completed_returns: list[float] = []
        completed_lengths: list[float] = []
        completed_success: list[float] = []

        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, info = env.step(act[i])
            done = terminated or truncated

            self.obs_buf[i] = obs
            self.rew_buf[i] = float(rew)
            self.done_buf[i] = bool(done)
            self.episode_length_buf[i] = env.episode_length

            if done:
                reason = str(info.get("termination_reason", TERMINATION_NAN))
                record = EpisodeRecord(
                    run_name=self.run_name,
                    robot_name=self.robot_cfg.name,
                    episode_id=self._episode_counter,
                    seed=self.seed + (i * 1009),
                    termination_reason=reason,
                    episode_return=float(info.get("episode_return", 0.0)),
                    episode_length=int(info.get("episode_length", 0)),
                    final_orientation_error=float(info.get("final_orientation_error", math.inf)),
                    mean_action_norm=float(info.get("mean_action_norm", 0.0)),
                    max_action_norm=float(info.get("max_action_norm", 0.0)),
                )
                self.logger.write(record)
                self._recent_records.append(record)
                self._episode_counter += 1

                completed_returns.append(record.episode_return)
                completed_lengths.append(float(record.episode_length))
                completed_success.append(1.0 if reason == TERMINATION_SUCCESS else 0.0)

                if reason == TERMINATION_TIMEOUT:
                    self._timeouts[i] = 1.0

                obs_reset, _ = env.reset()
                self.obs_buf[i] = obs_reset
                self.episode_length_buf[i] = 0

        extras: dict = {"time_outs": self._timeouts.clone()}
        if completed_returns:
            extras["log"] = {
                "/episode_return": torch.tensor(completed_returns, dtype=torch.float32).mean(),
                "/episode_length": torch.tensor(completed_lengths, dtype=torch.float32).mean(),
                "/success_rate": torch.tensor(completed_success, dtype=torch.float32).mean(),
            }

        return self._tensor_obs(), self.rew_buf.clone(), self.done_buf.clone(), extras

    def close(self) -> None:
        for env in self.envs:
            env.close()
