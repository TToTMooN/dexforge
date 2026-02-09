from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActuatorGroupCfg:
    """Actuator parameters for a joint subset."""

    name: str
    joint_exprs: tuple[str, ...]
    effort_limit: float
    stiffness: float
    damping: float
    frictionloss: float = 0.0
    armature: float = 0.0

    def to_mjlab_cfg(self):
        from mjlab.utils.spec_config import ActuatorCfg

        return ActuatorCfg(
            joint_names_expr=list(self.joint_exprs),
            effort_limit=float(self.effort_limit),
            stiffness=float(self.stiffness),
            damping=float(self.damping),
            frictionloss=float(self.frictionloss),
            armature=float(self.armature),
        )


@dataclass(frozen=True)
class RobotCfg:
    """Robot-specific model and control configuration."""

    name: str
    mjcf_path: Path
    source_reference: str
    joint_names: tuple[str, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    default_qpos: tuple[float, ...]
    action_scale: float
    target_delta_clip: float
    actuator_groups: tuple[ActuatorGroupCfg, ...]
    palm_site_name: str = "palm_center"

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    def validate(self) -> None:
        n = self.action_dim
        expected = [
            ("joint_lower", len(self.joint_lower)),
            ("joint_upper", len(self.joint_upper)),
            ("default_qpos", len(self.default_qpos)),
        ]
        for name, size in expected:
            if size != n:
                raise ValueError(
                    f"RobotCfg '{self.name}' mismatch: {name} has len={size}, expected {n}"
                )

        if self.action_scale <= 0.0:
            raise ValueError(f"RobotCfg '{self.name}' must have positive action_scale")
        if self.target_delta_clip <= 0.0:
            raise ValueError(f"RobotCfg '{self.name}' must have positive target_delta_clip")
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"Robot MJCF not found: {self.mjcf_path}")

    def joint_lower_np(self) -> np.ndarray:
        return np.asarray(self.joint_lower, dtype=np.float32)

    def joint_upper_np(self) -> np.ndarray:
        return np.asarray(self.joint_upper, dtype=np.float32)

    def default_qpos_np(self) -> np.ndarray:
        return np.asarray(self.default_qpos, dtype=np.float32)

    def to_mjlab_entity_cfg(self):
        import mujoco
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

        self.validate()

        pairs = zip(self.joint_names, self.default_qpos, strict=True)
        joint_pos = {name: float(q) for name, q in pairs}
        articulation = EntityArticulationInfoCfg(
            actuators=tuple(group.to_mjlab_cfg() for group in self.actuator_groups),
            soft_joint_pos_limit_factor=0.95,
        )

        return EntityCfg(
            spec_fn=lambda: mujoco.MjSpec.from_file(str(self.mjcf_path)),
            init_state=EntityCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=joint_pos,
                joint_vel={".*": 0.0},
            ),
            articulation=articulation,
        )
