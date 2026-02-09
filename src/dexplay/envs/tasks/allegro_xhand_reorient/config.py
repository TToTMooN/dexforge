from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReorientTaskCfg:
    """Task-level simulation and reward parameters for Phase-0 reorientation."""

    sim_dt: float = 0.0025
    control_hz: int = 25
    max_episode_seconds: float = 6.0

    joint_noise_std: float = 0.03
    joint_vel_noise_std: float = 0.02
    cube_pos_noise: float = 0.004
    cube_quat_noise_rad: float = 0.18

    orientation_tolerance_rad: float = 0.24
    success_hold_steps: int = 10

    drop_z_threshold: float = 0.02
    drop_distance_threshold: float = 0.180

    workspace_low: tuple[float, float, float] = (-0.22, -0.22, 0.00)
    workspace_high: tuple[float, float, float] = (0.22, 0.22, 0.24)

    reward_ori_scale: float = 20.0
    reward_dist_scale: float = 4.0
    reward_action_scale: float = 1.5
    success_bonus: float = 45.0

    cube_anchor_gain: float = 0.12
    cube_anchor_height: float = 0.035
    slip_action_threshold: float = 0.8
    slip_z_gain: float = 0.0045
    slip_xy_gain: float = 0.0015

    orientation_assist_gain: float = 0.8
    max_assist_angle_per_step: float = 0.10

    cube_mass: float = 0.035
    cube_half_extent: float = 0.015

    @property
    def decimation(self) -> int:
        return max(1, int(round(1.0 / (self.sim_dt * self.control_hz))))

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation

    @property
    def max_steps(self) -> int:
        return int(round(self.max_episode_seconds * self.control_hz))
