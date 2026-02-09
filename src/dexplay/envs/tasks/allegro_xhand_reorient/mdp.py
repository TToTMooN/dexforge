from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-6)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_angle_distance(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = normalize_quat(current)
    target = normalize_quat(target)
    dot = torch.sum(current * target, dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1e-6)
    half = 0.5 * angle
    s = torch.sin(half)
    quat = torch.cat((torch.cos(half), axis * s), dim=-1)
    return normalize_quat(quat)


def quat_to_axis(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat(quat)
    xyz = quat[..., 1:]
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    safe = torch.where(norm > 1e-6, xyz / norm, torch.tensor([1.0, 0.0, 0.0], device=quat.device))
    return safe


def sample_random_quat(num: int, device: str) -> torch.Tensor:
    axis = torch.randn(num, 3, device=device)
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1e-6)
    angle = (2.0 * math.pi) * torch.rand(num, 1, device=device) - math.pi
    return axis_angle_to_quat(axis, angle)


def sample_perturbed_quat(num: int, max_angle: float, device: str) -> torch.Tensor:
    base = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num, 1)
    axis = torch.randn(num, 3, device=device)
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1e-6)
    angle = (2.0 * torch.rand(num, 1, device=device) - 1.0) * max_angle
    delta = axis_angle_to_quat(axis, angle)
    return normalize_quat(quat_mul(delta, base))


def _site_pos_from_robot(robot, palm_site_cfg: SceneEntityCfg) -> torch.Tensor:
    site_pose = robot.data.site_pose_w[:, palm_site_cfg.site_ids, :3]
    if site_pose.ndim == 3:
        return site_pose[:, 0, :]
    return site_pose


def compute_reorient_state(
    env,
    robot_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    palm_site_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]

    cube_pos = cube.data.root_link_pos_w
    cube_quat = normalize_quat(cube.data.root_link_quat_w)
    palm_pos = _site_pos_from_robot(robot, palm_site_cfg)

    ori_err = quat_angle_distance(cube_quat, env.target_quat)
    palm_dist = torch.linalg.norm(cube_pos - palm_pos, dim=-1)
    return cube_pos, cube_quat, ori_err, palm_dist


# Observations.


def robot_joint_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids]


def robot_joint_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids]


def cube_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cube = env.scene[asset_cfg.name]
    return cube.data.root_link_pos_w


def cube_quat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cube = env.scene[asset_cfg.name]
    return normalize_quat(cube.data.root_link_quat_w)


def cube_linvel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cube = env.scene[asset_cfg.name]
    return cube.data.root_link_lin_vel_w


def cube_angvel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cube = env.scene[asset_cfg.name]
    return cube.data.root_link_ang_vel_w


# Rewards.


def reward_orientation_error(env) -> torch.Tensor:
    return env.last_orientation_error


def reward_cube_palm_distance(env) -> torch.Tensor:
    return env.last_palm_distance


def reward_action_norm(env) -> torch.Tensor:
    return env.last_action_norm


def reward_success_bonus(env, hold_steps: int) -> torch.Tensor:
    return (env.success_streak >= hold_steps).to(dtype=torch.float32)


# Terminations.


def term_success_hold(env, hold_steps: int) -> torch.Tensor:
    return env.success_streak >= hold_steps


def term_drop(env) -> torch.Tensor:
    return env.last_drop


def term_oob(env) -> torch.Tensor:
    return env.last_oob


def term_nan(env) -> torch.Tensor:
    return env.last_nan
