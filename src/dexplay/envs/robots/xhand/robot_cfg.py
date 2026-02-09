from __future__ import annotations

from pathlib import Path

from dexplay.envs.robots.common_robot_cfg import ActuatorGroupCfg, RobotCfg

# References used for naming and gain conventions:
# - dexmachina: dexmachina/envs/hand_cfgs/xhand.py (finger kp=20, kv=1.5)
# - ManipTrans: maniptrans_envs/lib/envs/dexhands/xhand.py (joint naming layout)
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
    source_reference=(
        "dexmachina/envs/hand_cfgs/xhand.py, maniptrans_envs/lib/envs/dexhands/xhand.py"
    ),
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
    action_scale=0.09,
    target_delta_clip=0.22,
    actuator_groups=(
        ActuatorGroupCfg(
            name="xhand_fingers",
            joint_exprs=("x_.*",),
            effort_limit=6.0,
            stiffness=16.0,
            damping=1.2,
            frictionloss=0.02,
            armature=0.003,
        ),
    ),
    palm_site_name="palm_center",
)

XHAND_CFG.validate()
