# DexPlay Phase-0 RL Playground

Phase-0 dexterous manipulation playground with:
- `rsl_rl` PPO training loop
- one toy task: in-hand cube reorientation
- two robot configs selectable by flag: `allegro` and `xhand`
- Pixi-managed environment and commands only

## Quick Start (Pixi)

1. Install dependencies:

```bash
pixi install
```

2. Train Allegro:

```bash
pixi run train_allegro
```

3. Train xHand:

```bash
pixi run train_xhand
```

4. Deterministic evaluation:

```bash
pixi run eval_allegro
pixi run eval_xhand
```

## Project Commands

- `pixi run train_allegro`
- `pixi run train_xhand`
- `pixi run eval_allegro`
- `pixi run eval_xhand`
- `pixi run fmt`
- `pixi run lint`

## Outputs

All run artifacts are written under `runs/<run_name>/`:
- `train.jsonl`: per-episode training diagnostics
- `eval.jsonl`: per-episode evaluation diagnostics
- `eval_report.md`: aggregated eval summary and failure histogram
- `checkpoints/latest.pt` and timed checkpoints
- `train_config.json`

## Task Scope

Implemented task: `allegro_xhand_reorient` (Phase-0)
- no vision
- no domain randomization
- no curriculum
- no multi-task

Observation vector is fixed within each run and contains:
- robot joint qpos/qvel
- cube pose and velocities

Action is joint target delta with internal PD control and clamps.

Termination reasons:
- `SUCCESS`
- `DROP`
- `OOB`
- `TIMEOUT`
- `NAN`

## DoD Checklist (Phase-0)

- [x] Reset/step runs with finite observations and NaN guards for both robots.
- [x] One PPO pipeline (`rsl_rl`) works for both robots via `--robot` config flag.
- [x] Deterministic eval with fixed seed outputs success rate, episode length, failure histogram.
- [x] Episode-level JSONL diagnostics written for train/eval.
- [x] Pixi is the only required workflow (`pixi install`, `pixi run ...`).

## Notes on `mjlab`

`mjlab` is evolving quickly and may require extra indexes / platform-specific Warp wheels.
This Phase-0 repo keeps env logic isolated under `src/dexplay/envs/` and defaults to a
self-contained MuJoCo CPU backend while preserving a manager-style API and backend flag:

- `--backend auto` (default)
- `--backend mjlab` (falls back with warning if unavailable)
- `--backend mujoco`

This makes swapping in native `mjlab` simulation straightforward in later phases.
