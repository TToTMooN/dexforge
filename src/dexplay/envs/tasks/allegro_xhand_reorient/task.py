from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as common_terminations
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg

from dexplay.envs.robots.common_robot_cfg import RobotCfg
from dexplay.envs.tasks.allegro_xhand_reorient import mdp
from dexplay.envs.tasks.allegro_xhand_reorient.config import ReorientTaskCfg
from dexplay.utils.logging import EpisodeRecord, JsonlEpisodeLogger
from dexplay.utils.paths import eval_log_path, train_log_path

TERMINATION_SUCCESS = "SUCCESS"
TERMINATION_DROP = "DROP"
TERMINATION_OOB = "OOB"
TERMINATION_TIMEOUT = "TIMEOUT"
TERMINATION_NAN = "NAN"


def _as_tensor_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long)


def reset_reorient_state(
    env,
    env_ids: torch.Tensor | slice,
    robot_asset_cfg: SceneEntityCfg,
    cube_asset_cfg: SceneEntityCfg,
    task_cfg: ReorientTaskCfg,
) -> None:
    env_ids = _as_tensor_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot = env.scene[robot_asset_cfg.name]
    cube = env.scene[cube_asset_cfg.name]

    num = int(env_ids.numel())
    q_default = env.robot_default_qpos[None].repeat(num, 1)
    q_noise = torch.randn(num, env.robot_action_dim, device=env.device) * task_cfg.joint_noise_std
    q = torch.clamp(q_default + q_noise, env.robot_joint_lower, env.robot_joint_upper)
    qd = torch.randn(num, env.robot_action_dim, device=env.device) * task_cfg.joint_vel_noise_std
    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)

    origins = env.scene.env_origins[env_ids]
    cube_pos = origins + torch.tensor([0.0, 0.0, 0.125], device=env.device)
    cube_pos += torch.randn(num, 3, device=env.device) * task_cfg.cube_pos_noise
    cube_quat = mdp.sample_perturbed_quat(num, task_cfg.cube_quat_noise_rad, env.device)
    cube_pose = torch.cat((cube_pos, cube_quat), dim=-1)

    cube.write_root_link_pose_to_sim(cube_pose, env_ids=env_ids)
    cube.write_root_link_velocity_to_sim(torch.zeros(num, 6, device=env.device), env_ids=env_ids)

    env.target_quat[env_ids] = mdp.sample_random_quat(num, env.device)
    env.success_streak[env_ids] = 0
    env.last_orientation_error[env_ids] = math.pi
    env.last_palm_distance[env_ids] = 0.0
    env.last_action_norm[env_ids] = 0.0
    env.last_drop[env_ids] = False
    env.last_oob[env_ids] = False
    env.last_nan[env_ids] = False


@dataclass
class ReorientActionCfg:
    joint_pos: JointPositionActionCfg = term(
        JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],
        scale=0.1,
        use_default_offset=True,
    )


@dataclass
class ReorientObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        joint_pos: ObsTerm = term(
            ObsTerm,
            func=mdp.robot_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel: ObsTerm = term(
            ObsTerm,
            func=mdp.robot_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        cube_pos: ObsTerm = term(
            ObsTerm,
            func=mdp.cube_pos,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )
        cube_quat: ObsTerm = term(
            ObsTerm,
            func=mdp.cube_quat,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )
        cube_linvel: ObsTerm = term(
            ObsTerm,
            func=mdp.cube_linvel,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )
        cube_angvel: ObsTerm = term(
            ObsTerm,
            func=mdp.cube_angvel,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )

    @dataclass
    class CriticCfg(PolicyCfg):
        pass

    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)


@dataclass
class ReorientRewardCfg:
    orientation: RewardTerm = term(RewardTerm, func=mdp.reward_orientation_error, weight=-20.0)
    palm_distance: RewardTerm = term(
        RewardTerm,
        func=mdp.reward_cube_palm_distance,
        weight=-4.0,
    )
    action_penalty: RewardTerm = term(
        RewardTerm,
        func=mdp.reward_action_norm,
        weight=-1.5,
    )
    success_bonus: RewardTerm = term(
        RewardTerm,
        func=mdp.reward_success_bonus,
        weight=45.0,
        params={"hold_steps": 10},
    )


@dataclass
class ReorientTerminationCfg:
    time_out: DoneTerm = term(DoneTerm, func=common_terminations.time_out, time_out=True)
    success_hold: DoneTerm = term(
        DoneTerm,
        func=mdp.term_success_hold,
        params={"hold_steps": 10},
    )
    drop: DoneTerm = term(DoneTerm, func=mdp.term_drop)
    oob: DoneTerm = term(DoneTerm, func=mdp.term_oob)
    nan_state: DoneTerm = term(DoneTerm, func=mdp.term_nan)


@dataclass
class ReorientEventCfg:
    reset_task_state: EventTerm = term(
        EventTerm,
        func=reset_reorient_state,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "cube_asset_cfg": SceneEntityCfg("cube"),
            "task_cfg": ReorientTaskCfg(),
        },
    )


@dataclass
class ReorientManagerEnvCfg(ManagerBasedRlEnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            num_envs=1,
            env_spacing=0.55,
            terrain=TerrainImporterCfg(terrain_type="plane"),
            extent=1.0,
        )
    )
    observations: ReorientObservationCfg = field(default_factory=ReorientObservationCfg)
    actions: ReorientActionCfg = field(default_factory=ReorientActionCfg)
    rewards: ReorientRewardCfg = field(default_factory=ReorientRewardCfg)
    terminations: ReorientTerminationCfg = field(default_factory=ReorientTerminationCfg)
    events: ReorientEventCfg = field(default_factory=ReorientEventCfg)
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            nconmax=30_000,
            njmax=3_000,
            mujoco=MujocoCfg(
                timestep=0.0025,
                integrator="implicitfast",
                gravity=(0.0, 0.0, -1.5),
                iterations=25,
                ls_iterations=30,
            ),
        )
    )
    decimation: int = 10
    episode_length_s: float = 6.0
    is_finite_horizon: bool = False


def _cube_entity_cfg(cube_xml_path: Path) -> EntityCfg:
    import mujoco

    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(cube_xml_path)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.13),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        articulation=None,
    )


def build_reorient_mjlab_cfg(
    robot_cfg: RobotCfg,
    task_cfg: ReorientTaskCfg,
    num_envs: int,
    seed: int,
) -> ReorientManagerEnvCfg:
    cfg = ReorientManagerEnvCfg()
    cfg.seed = seed
    cfg.scene.num_envs = int(num_envs)
    cfg.decimation = task_cfg.decimation
    cfg.episode_length_s = task_cfg.max_episode_seconds

    cfg.sim.mujoco.timestep = task_cfg.sim_dt

    cube_xml = Path(__file__).resolve().parent / "model" / "cube.xml"
    cfg.scene.entities = {
        "robot": robot_cfg.to_mjlab_entity_cfg(),
        "cube": _cube_entity_cfg(cube_xml),
    }

    palm_site_cfg = SceneEntityCfg(
        "robot", site_names=[robot_cfg.palm_site_name], preserve_order=True
    )
    joint_cfg = SceneEntityCfg(
        "robot", joint_names=list(robot_cfg.joint_names), preserve_order=True
    )

    cfg.observations.policy.joint_pos.params["asset_cfg"] = joint_cfg
    cfg.observations.policy.joint_vel.params["asset_cfg"] = joint_cfg
    cfg.observations.critic.joint_pos.params["asset_cfg"] = joint_cfg
    cfg.observations.critic.joint_vel.params["asset_cfg"] = joint_cfg

    cfg.actions.joint_pos.scale = robot_cfg.action_scale

    cfg.rewards.orientation.weight = -task_cfg.reward_ori_scale
    cfg.rewards.palm_distance.weight = -task_cfg.reward_dist_scale
    cfg.rewards.action_penalty.weight = -task_cfg.reward_action_scale
    cfg.rewards.success_bonus.weight = task_cfg.success_bonus
    cfg.rewards.success_bonus.params["hold_steps"] = task_cfg.success_hold_steps

    cfg.terminations.success_hold.params["hold_steps"] = task_cfg.success_hold_steps

    cfg.events.reset_task_state.params["robot_asset_cfg"] = joint_cfg
    cfg.events.reset_task_state.params["cube_asset_cfg"] = SceneEntityCfg("cube")
    cfg.events.reset_task_state.params["task_cfg"] = task_cfg

    # Cache useful config for state updates.
    cfg._robot_joint_cfg = joint_cfg  # type: ignore[attr-defined]
    cfg._palm_site_cfg = palm_site_cfg  # type: ignore[attr-defined]
    cfg._cube_cfg = SceneEntityCfg("cube")  # type: ignore[attr-defined]
    cfg._robot_cfg = robot_cfg  # type: ignore[attr-defined]
    cfg._task_cfg = task_cfg  # type: ignore[attr-defined]

    return cfg


class ReorientMjlabEnv(ManagerBasedRlEnv):
    """Manager-based Phase-0 in-hand reorientation environment on mjlab."""

    def __init__(
        self,
        cfg: ReorientManagerEnvCfg,
        robot_cfg: RobotCfg,
        task_cfg: ReorientTaskCfg,
        device: str,
        debug: bool = False,
    ) -> None:
        self.robot_name = robot_cfg.name
        self.task_cfg = task_cfg
        self.debug = debug

        super().__init__(cfg=cfg, device=device)

        n = self.num_envs
        self.robot_action_dim = robot_cfg.action_dim
        self.robot_joint_lower = torch.tensor(
            robot_cfg.joint_lower, dtype=torch.float32, device=self.device
        )
        self.robot_joint_upper = torch.tensor(
            robot_cfg.joint_upper, dtype=torch.float32, device=self.device
        )
        self.robot_default_qpos = torch.tensor(
            robot_cfg.default_qpos, dtype=torch.float32, device=self.device
        )

        self._robot_state_cfg = SceneEntityCfg(
            "robot", joint_names=list(robot_cfg.joint_names), preserve_order=True
        )
        self._robot_state_cfg.resolve(self.scene)
        self._cube_state_cfg = SceneEntityCfg("cube")
        self._cube_state_cfg.resolve(self.scene)
        self._palm_site_cfg = SceneEntityCfg(
            "robot", site_names=[robot_cfg.palm_site_name], preserve_order=True
        )
        self._palm_site_cfg.resolve(self.scene)

        self.target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        self.success_streak = torch.zeros(n, dtype=torch.long, device=self.device)

        self.last_orientation_error = torch.full(
            (n,), math.pi, dtype=torch.float32, device=self.device
        )
        self.last_palm_distance = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.last_action_norm = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.last_drop = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_oob = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_nan = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Preserved across per-step resets for logging after done envs are auto-reset.
        self.last_done_orientation_error = self.last_orientation_error.clone()
        self.last_done_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_drop = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_oob = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_nan = torch.zeros(n, dtype=torch.bool, device=self.device)

    def _apply_cube_assist(self) -> None:
        robot = self.scene[self._robot_state_cfg.name]
        cube = self.scene[self._cube_state_cfg.name]

        cube_pos = cube.data.root_link_pos_w
        cube_quat = mdp.normalize_quat(cube.data.root_link_quat_w)

        palm_site_pose = robot.data.site_pose_w[:, self._palm_site_cfg.site_ids, :3]
        palm_pos = palm_site_pose[:, 0, :] if palm_site_pose.ndim == 3 else palm_site_pose

        target = mdp.normalize_quat(self.target_quat)
        err_quat = mdp.normalize_quat(mdp.quat_mul(target, mdp.quat_conjugate(cube_quat)))
        err_axis = mdp.quat_to_axis(err_quat)

        action_energy = self.last_action_norm
        assist_angle = self.task_cfg.orientation_assist_gain * action_energy * self.step_dt
        assist_angle = torch.clamp(assist_angle, max=self.task_cfg.max_assist_angle_per_step)

        delta_quat = mdp.axis_angle_to_quat(err_axis, assist_angle[:, None])
        new_quat = mdp.normalize_quat(mdp.quat_mul(delta_quat, cube_quat))

        anchor = palm_pos + torch.tensor(
            [0.0, 0.0, self.task_cfg.cube_anchor_height], device=self.device
        )
        new_pos = cube_pos + self.task_cfg.cube_anchor_gain * (anchor - cube_pos)

        action = self.action_manager.action
        slip = torch.clamp(action_energy - self.task_cfg.slip_action_threshold, min=0.0)

        if action.shape[1] > 0:
            x_comp = action[:, 0::3].mean(dim=1)
            y_comp = (
                action[:, 1::3].mean(dim=1) if action.shape[1] > 1 else torch.zeros_like(x_comp)
            )
            dir_xy = torch.stack((x_comp, y_comp), dim=-1)
            dir_xy = dir_xy / torch.clamp(torch.linalg.norm(dir_xy, dim=-1, keepdim=True), min=1e-6)
            new_pos[:, :2] += self.task_cfg.slip_xy_gain * slip[:, None] * dir_xy

        new_pos[:, 2] -= self.task_cfg.slip_z_gain * slip

        cube.write_root_link_pose_to_sim(torch.cat((new_pos, new_quat), dim=-1))

        cube_vel = torch.zeros(self.num_envs, 6, device=self.device)
        cube_vel[:, 3:] = err_axis * (assist_angle / max(self.step_dt, 1e-6))[:, None]
        cube.write_root_link_velocity_to_sim(cube_vel)

    def _update_task_state(self) -> None:
        robot = self.scene[self._robot_state_cfg.name]
        cube = self.scene[self._cube_state_cfg.name]

        cube_pos, _cube_quat, ori_err, palm_dist = mdp.compute_reorient_state(
            self,
            self._robot_state_cfg,
            self._cube_state_cfg,
            self._palm_site_cfg,
        )

        self.last_orientation_error = ori_err
        self.last_palm_distance = palm_dist

        self.success_streak = torch.where(
            ori_err < self.task_cfg.orientation_tolerance_rad,
            self.success_streak + 1,
            torch.zeros_like(self.success_streak),
        )

        self.last_drop = (cube_pos[:, 2] < self.task_cfg.drop_z_threshold) | (
            palm_dist > self.task_cfg.drop_distance_threshold
        )

        low = torch.tensor(self.task_cfg.workspace_low, device=self.device)
        high = torch.tensor(self.task_cfg.workspace_high, device=self.device)
        self.last_oob = torch.any(cube_pos < low, dim=-1) | torch.any(cube_pos > high, dim=-1)

        finite_flags = (
            torch.isfinite(robot.data.joint_pos).all(dim=1)
            & torch.isfinite(robot.data.joint_vel).all(dim=1)
            & torch.isfinite(cube.data.root_link_pose_w).all(dim=1)
            & torch.isfinite(cube.data.root_link_vel_w).all(dim=1)
            & torch.isfinite(self.target_quat).all(dim=1)
            & torch.isfinite(ori_err)
            & torch.isfinite(palm_dist)
        )
        self.last_nan = self.last_nan | (~finite_flags)

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        invalid_action = ~torch.isfinite(action).all(dim=1)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.clamp(action, -1.0, 1.0)

        if action.shape[1] != self.action_manager.total_action_dim:
            expected = self.action_manager.total_action_dim
            raise ValueError(f"Action shape mismatch: got {action.shape[1]}, expected {expected}")

        self.last_action_norm = torch.linalg.norm(action, dim=1) / math.sqrt(
            max(action.shape[1], 1)
        )
        self.last_nan = invalid_action.clone()

        self.action_manager.process_action(action)

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self._apply_cube_assist()
        self.scene.write_data_to_sim()
        self.sim.forward()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_task_state()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        self.last_done_orientation_error = self.last_orientation_error.clone()
        self.last_done_success = self.success_streak >= self.task_cfg.success_hold_steps
        self.last_done_drop = self.last_drop.clone()
        self.last_done_oob = self.last_oob.clone()
        self.last_done_nan = self.last_nan.clone()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute()

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )


class ReorientVecEnv(RslRlVecEnvWrapper):
    """rsl_rl-compatible wrapper with per-episode JSONL diagnostics."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        task_cfg: ReorientTaskCfg,
        num_envs: int,
        seed: int,
        run_name: str,
        split: Literal["train", "eval"],
        device: str,
        debug: bool = False,
    ) -> None:
        cfg = build_reorient_mjlab_cfg(
            robot_cfg=robot_cfg,
            task_cfg=task_cfg,
            num_envs=num_envs,
            seed=seed,
        )
        env = ReorientMjlabEnv(
            cfg=cfg,
            robot_cfg=robot_cfg,
            task_cfg=task_cfg,
            device=device,
            debug=debug,
        )
        super().__init__(env=env, clip_actions=1.0)

        self.run_name = run_name
        self.robot_name = robot_cfg.name
        self.seed = int(seed)
        self.split = split

        path = train_log_path(run_name) if split == "train" else eval_log_path(run_name)
        self.logger = JsonlEpisodeLogger(path)

        self._episode_counter = 0
        self._recent_records: list[EpisodeRecord] = []

        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_norm_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._action_norm_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._seed_offset = torch.arange(self.num_envs, device=self.device, dtype=torch.long) * 1009

    def consume_recent_records(self) -> list[EpisodeRecord]:
        out = self._recent_records
        self._recent_records = []
        return out

    def _reasons_for(self, done_ids: torch.Tensor, timeouts: torch.Tensor) -> list[str]:
        reasons: list[str] = []
        for env_id in done_ids.tolist():
            if bool(timeouts[env_id].item()):
                reasons.append(TERMINATION_TIMEOUT)
            elif bool(self.unwrapped.last_done_nan[env_id].item()):
                reasons.append(TERMINATION_NAN)
            elif bool(self.unwrapped.last_done_success[env_id].item()):
                reasons.append(TERMINATION_SUCCESS)
            elif bool(self.unwrapped.last_done_drop[env_id].item()):
                reasons.append(TERMINATION_DROP)
            elif bool(self.unwrapped.last_done_oob[env_id].item()):
                reasons.append(TERMINATION_OOB)
            else:
                reasons.append(TERMINATION_NAN)
        return reasons

    def step(self, actions: torch.Tensor):
        action_norm = torch.linalg.norm(actions.to(self.device), dim=1)

        obs, rewards, dones, extras = super().step(actions)

        done_mask = dones.to(dtype=torch.bool)
        self._episode_return += rewards
        self._episode_length += 1
        self._action_norm_sum += action_norm
        self._action_norm_max = torch.maximum(self._action_norm_max, action_norm)

        if torch.any(done_mask):
            done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
            timeouts = extras.get("time_outs", torch.zeros_like(done_mask))
            reasons = self._reasons_for(done_ids, timeouts)

            for idx, env_id in enumerate(done_ids.tolist()):
                episode_length = int(self._episode_length[env_id].item())
                mean_action = self._action_norm_sum[env_id] / max(episode_length, 1)
                reason = reasons[idx]

                rec = EpisodeRecord(
                    run_name=self.run_name,
                    robot_name=self.robot_name,
                    episode_id=self._episode_counter,
                    seed=int((self.seed + self._seed_offset[env_id]).item()),
                    termination_reason=reason,
                    episode_return=float(self._episode_return[env_id].item()),
                    episode_length=episode_length,
                    final_orientation_error=float(
                        self.unwrapped.last_done_orientation_error[env_id].item()
                    ),
                    mean_action_norm=float(mean_action.item()),
                    max_action_norm=float(self._action_norm_max[env_id].item()),
                )
                self.logger.write(rec)
                self._recent_records.append(rec)
                self._episode_counter += 1

            self._episode_return[done_ids] = 0.0
            self._episode_length[done_ids] = 0
            self._action_norm_sum[done_ids] = 0.0
            self._action_norm_max[done_ids] = 0.0

        return obs, rewards, dones, extras


class ReorientEnv:
    """Single-env numpy API adapter for debugging and contract checks."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        task_cfg: ReorientTaskCfg,
        seed: int,
        device: str,
        debug: bool = False,
    ) -> None:
        cfg = build_reorient_mjlab_cfg(
            robot_cfg=robot_cfg, task_cfg=task_cfg, num_envs=1, seed=seed
        )
        self._env = ReorientMjlabEnv(
            cfg=cfg,
            robot_cfg=robot_cfg,
            task_cfg=task_cfg,
            device=device,
            debug=debug,
        )
        self._ep_return = 0.0
        self._ep_length = 0
        self._act_norm_sum = 0.0
        self._act_norm_max = 0.0

    @property
    def action_dim(self) -> int:
        return int(self._env.action_manager.total_action_dim)

    @property
    def observation_dim(self) -> int:
        obs, _ = self._env.reset()
        policy = obs["policy"]
        return int(policy.shape[-1])

    def reset(self):
        obs, _ = self._env.reset()
        self._ep_return = 0.0
        self._ep_length = 0
        self._act_norm_sum = 0.0
        self._act_norm_max = 0.0

        policy = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
        info = {"target_quat": self._env.target_quat[0].detach().cpu().numpy().copy()}
        return policy, info

    def step(self, action: np.ndarray):
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self._env.device).reshape(
            1, -1
        )
        obs, rew, terminated, truncated, extras = self._env.step(action_t)

        reward = float(rew[0].item())
        self._ep_return += reward
        self._ep_length += 1

        act_norm = float(torch.linalg.norm(action_t[0]).item())
        self._act_norm_sum += act_norm
        self._act_norm_max = max(self._act_norm_max, act_norm)

        done = bool(terminated[0].item() or truncated[0].item())
        info: dict = {}

        if done:
            if bool(truncated[0].item()):
                reason = TERMINATION_TIMEOUT
            elif bool(self._env.last_done_nan[0].item()):
                reason = TERMINATION_NAN
            elif bool(self._env.last_done_success[0].item()):
                reason = TERMINATION_SUCCESS
            elif bool(self._env.last_done_drop[0].item()):
                reason = TERMINATION_DROP
            elif bool(self._env.last_done_oob[0].item()):
                reason = TERMINATION_OOB
            else:
                reason = TERMINATION_NAN

            info = {
                "termination_reason": reason,
                "episode_return": float(self._ep_return),
                "episode_length": int(self._ep_length),
                "final_orientation_error": float(self._env.last_done_orientation_error[0].item()),
                "mean_action_norm": float(self._act_norm_sum / max(self._ep_length, 1)),
                "max_action_norm": float(self._act_norm_max),
            }

        policy = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
        return policy, reward, bool(terminated[0].item()), bool(truncated[0].item()), info

    def close(self) -> None:
        self._env.close()


__all__ = [
    "TERMINATION_SUCCESS",
    "TERMINATION_DROP",
    "TERMINATION_OOB",
    "TERMINATION_TIMEOUT",
    "TERMINATION_NAN",
    "ReorientEnv",
    "ReorientVecEnv",
    "ReorientMjlabEnv",
    "build_reorient_mjlab_cfg",
]
