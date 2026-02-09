# DexPlay Phase-0 RL Playground

Phase-0 dexterous manipulation playground built on:
- `mjlab` (MuJoCo Warp + manager-based API)
- `rsl_rl` PPO
- `Pixi` for all environment management and commands

Task scope is intentionally narrow: one toy task (`allegro_xhand_reorient`) with one cube,
no vision, no domain randomization, no curriculum, no multi-task.

## Pixi Workflow

Install once:

```bash
pixi install
```

Train:

```bash
pixi run train_allegro
pixi run train_xhand
```

Evaluate deterministic checkpoints:

```bash
pixi run eval_allegro
pixi run eval_xhand
```

Quality:

```bash
pixi run fmt
pixi run lint
```

## Notes on CLI

Train/eval entrypoints use `tyro`-based CLIs:
- `python -m dexplay.rl.train_ppo --help`
- `python -m dexplay.rl.eval --help`

## Outputs

All run artifacts are written under `runs/<run_name>/`:
- `train.jsonl`
- `eval.jsonl`
- `eval_report.md`
- `checkpoints/latest.pt` and timestamped checkpoints
- `train_config.json`

## Robot Support

Robot selection is a config flag (`--robot allegro|xhand`), not a code rewrite.

Reference repositories used for naming/gain conventions:
- [dexmachina](https://github.com/MandiZhao/dexmachina)
- [ManipTrans](https://github.com/ManipTrans/ManipTrans)

Current repository assets for both hands are replaceable placeholder MJCFs, with matching
joint naming so swapping to full assets later is straightforward.

## Phase-0 DoD Checklist

- [x] Env reset/step runs with finite observations and NaN guards for both robots.
- [x] `mjlab` manager-based env is the primary backend.
- [x] PPO train/eval pipeline uses `rsl_rl` and logs episode JSONL diagnostics.
- [x] Deterministic eval reports success rate, episode length, failure histogram.
- [x] Robot switching is `--robot` only.
- [x] Pixi is the only required install/run path.
