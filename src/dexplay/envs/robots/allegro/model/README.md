# Allegro Placeholder Model (Phase-0)

This repository currently ships `allegro_phase0.xml`, a lightweight Allegro-like MJCF
used for Phase-0 infrastructure testing with `mjlab`.

Reference sources used for naming and control conventions:
- `dexmachina/envs/hand_cfgs/allegro.py`
- `maniptrans_envs/lib/envs/dexhands/allegro.py`

Note: full upstream Allegro mesh assets in those repos were incomplete in this environment,
so this file remains a replaceable placeholder. Keep joint names aligned with
`src/dexplay/envs/robots/allegro/robot_cfg.py` when swapping in a full asset.
