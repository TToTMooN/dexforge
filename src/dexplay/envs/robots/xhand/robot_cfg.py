from __future__ import annotations

from pathlib import Path

from dexplay.envs.robots.common_robot_cfg import RobotCfg

XHAND_JOINTS = (
    "x_thumb_0",
    "x_thumb_1",
    "x_thumb_2",
    "x_index_0",
    "x_index_1",
    "x_index_2",
    "x_middle_0",
    "x_middle_1",
    "x_middle_2",
    "x_ring_0",
    "x_ring_1",
    "x_ring_2",
)


XHAND_CFG = RobotCfg(
    name="xhand",
    mjcf_path=Path(__file__).resolve().parent / "model" / "xhand_phase0.xml",
    joint_names=XHAND_JOINTS,
    joint_lower=(
        -0.35,
        -0.30,
        -0.20,
        -0.35,
        -0.25,
        -0.25,
        -0.35,
        -0.25,
        -0.25,
        -0.35,
        -0.25,
        -0.25,
    ),
    joint_upper=(0.90, 1.40, 1.35, 0.55, 1.35, 1.35, 0.55, 1.35, 1.35, 0.55, 1.35, 1.35),
    default_qpos=(0.30, 0.70, 0.65, 0.08, 0.62, 0.58, 0.10, 0.65, 0.60, 0.12, 0.68, 0.60),
    kp=(2.7,) * 12,
    kd=(0.07,) * 12,
    action_scale=0.09,
    torque_lower=(-1.0,) * 12,
    torque_upper=(1.0,) * 12,
    target_delta_clip=0.22,
)

XHAND_CFG.validate()
