from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RobotCfg:
    """Robot-specific control and model configuration."""

    name: str
    mjcf_path: Path
    joint_names: tuple[str, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    default_qpos: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    action_scale: float
    torque_lower: tuple[float, ...]
    torque_upper: tuple[float, ...]
    target_delta_clip: float
    palm_site_name: str = "palm_center"
    cube_joint_name: str = "cube_free"

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    def joint_lower_np(self) -> np.ndarray:
        return np.asarray(self.joint_lower, dtype=np.float64)

    def joint_upper_np(self) -> np.ndarray:
        return np.asarray(self.joint_upper, dtype=np.float64)

    def default_qpos_np(self) -> np.ndarray:
        return np.asarray(self.default_qpos, dtype=np.float64)

    def kp_np(self) -> np.ndarray:
        return np.asarray(self.kp, dtype=np.float64)

    def kd_np(self) -> np.ndarray:
        return np.asarray(self.kd, dtype=np.float64)

    def torque_lower_np(self) -> np.ndarray:
        return np.asarray(self.torque_lower, dtype=np.float64)

    def torque_upper_np(self) -> np.ndarray:
        return np.asarray(self.torque_upper, dtype=np.float64)

    def validate(self) -> None:
        n = self.action_dim
        expected = [
            ("joint_lower", len(self.joint_lower)),
            ("joint_upper", len(self.joint_upper)),
            ("default_qpos", len(self.default_qpos)),
            ("kp", len(self.kp)),
            ("kd", len(self.kd)),
            ("torque_lower", len(self.torque_lower)),
            ("torque_upper", len(self.torque_upper)),
        ]
        for name, size in expected:
            if size != n:
                raise ValueError(
                    f"RobotCfg '{self.name}' mismatch: {name} has len={size}, expected {n}"
                )
