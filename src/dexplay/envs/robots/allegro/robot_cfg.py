from __future__ import annotations

from pathlib import Path

from dexplay.envs.robots.common_robot_cfg import ActuatorGroupCfg, RobotCfg

# References used for naming and gain conventions:
# - dexmachina: dexmachina/envs/hand_cfgs/allegro.py (finger kp=30, kv=2)
# - ManipTrans: maniptrans_envs/lib/envs/dexhands/allegro.py (joint naming layout)
ALLEGRO_JOINTS = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "index_0",
    "index_1",
    "index_2",
    "index_3",
    "middle_0",
    "middle_1",
    "middle_2",
    "middle_3",
    "ring_0",
    "ring_1",
    "ring_2",
    "ring_3",
)


ALLEGRO_CFG = RobotCfg(
    name="allegro",
    mjcf_path=Path(__file__).resolve().parent / "model" / "allegro_phase0.xml",
    source_reference=(
        "dexmachina/envs/hand_cfgs/allegro.py, maniptrans_envs/lib/envs/dexhands/allegro.py"
    ),
    joint_names=ALLEGRO_JOINTS,
    joint_lower=(
        -0.50,
        -0.30,
        -0.20,
        -0.20,
        -0.40,
        -0.20,
        -0.20,
        -0.20,
        -0.40,
        -0.20,
        -0.20,
        -0.20,
        -0.40,
        -0.20,
        -0.20,
        -0.20,
    ),
    joint_upper=(
        0.80,
        1.20,
        1.40,
        1.30,
        0.40,
        1.40,
        1.40,
        1.30,
        0.40,
        1.40,
        1.40,
        1.30,
        0.40,
        1.40,
        1.40,
        1.30,
    ),
    default_qpos=(
        0.24,
        0.60,
        0.70,
        0.55,
        0.00,
        0.50,
        0.60,
        0.55,
        0.05,
        0.55,
        0.65,
        0.55,
        0.08,
        0.58,
        0.68,
        0.58,
    ),
    action_scale=0.08,
    target_delta_clip=0.20,
    actuator_groups=(
        ActuatorGroupCfg(
            name="allegro_fingers",
            joint_exprs=("index_.*", "middle_.*", "ring_.*", "thumb_.*"),
            effort_limit=7.5,
            stiffness=22.0,
            damping=1.5,
            frictionloss=0.02,
            armature=0.004,
        ),
    ),
    palm_site_name="palm_center",
)

ALLEGRO_CFG.validate()
