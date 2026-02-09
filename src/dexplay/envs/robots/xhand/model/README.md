# xHand Placeholder Model (Phase-0)

`xhand_phase0.xml` is a placeholder xHand-like MJCF used to validate Phase-0 RL
training/eval plumbing on `mjlab`.

Reference sources used for naming and control conventions:
- `dexmachina/envs/hand_cfgs/xhand.py`
- `maniptrans_envs/lib/envs/dexhands/xhand.py`

Full upstream xHand mesh assets were not complete in this environment, so this model is
intentionally replaceable. Keep joint names aligned with
`src/dexplay/envs/robots/xhand/robot_cfg.py` when replacing it.
