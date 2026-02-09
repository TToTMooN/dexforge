"""Robot configuration registry."""

from dexplay.envs.robots.allegro.robot_cfg import ALLEGRO_CFG
from dexplay.envs.robots.common_robot_cfg import RobotCfg
from dexplay.envs.robots.xhand.robot_cfg import XHAND_CFG


def get_robot_cfg(name: str) -> RobotCfg:
    key = name.lower()
    if key == "allegro":
        return ALLEGRO_CFG
    if key == "xhand":
        return XHAND_CFG
    raise ValueError(f"Unsupported robot '{name}'. Expected one of: allegro, xhand")


__all__ = ["RobotCfg", "ALLEGRO_CFG", "XHAND_CFG", "get_robot_cfg"]
