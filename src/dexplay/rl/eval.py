from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from dexplay.envs.robots import get_robot_cfg
from dexplay.envs.tasks.allegro_xhand_reorient.config import ReorientTaskCfg
from dexplay.envs.tasks.allegro_xhand_reorient.task import ReorientVecEnv
from dexplay.rl.diagnostics import failure_histogram, summary_stats, write_eval_report
from dexplay.rl.ppo_config import build_ppo_train_cfg
from dexplay.utils.logging import write_jsonl
from dexplay.utils.paths import eval_log_path, eval_report_path
from dexplay.utils.seeding import seed_everything


@dataclass
class EvalArgs:
    run_name: str
    robot: Literal["allegro", "xhand"]
    checkpoint_path: Path
    episodes: int = 50
    seed: int = 0
    device: Literal["cuda", "cpu"] = "cuda"
    debug: bool = False


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate_checkpoint(
    run_name: str,
    robot: str,
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    device: str,
    debug: bool = False,
) -> dict[str, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    seed_everything(seed)

    eval_path = eval_log_path(run_name)
    if eval_path.exists():
        eval_path.unlink()

    robot_cfg = get_robot_cfg(robot)
    task_cfg = ReorientTaskCfg()

    env = ReorientVecEnv(
        robot_cfg=robot_cfg,
        task_cfg=task_cfg,
        num_envs=1,
        seed=seed,
        run_name=run_name,
        split="eval",
        device=device,
        debug=debug,
    )

    cfg = build_ppo_train_cfg(save_interval_iters=10_000_000)
    runner = OnPolicyRunner(env=env, train_cfg=cfg, log_dir=None, device=device)
    runner.load(str(checkpoint_path), load_optimizer=False, map_location=device)

    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()

    collected: list[dict] = []
    while len(collected) < episodes:
        with torch.inference_mode():
            actions = policy(obs.to(device))
        obs, _, _, _ = env.step(actions.to(env.device))

        for rec in env.consume_recent_records():
            collected.append(rec.__dict__.copy())
            if len(collected) >= episodes:
                break

    collected = collected[:episodes]
    write_jsonl(eval_path, collected)

    report_path = eval_report_path(run_name)
    write_eval_report(
        report_path=report_path,
        run_name=run_name,
        robot_name=robot,
        checkpoint_path=str(checkpoint_path),
        seed=seed,
        episodes=episodes,
        records=collected,
    )

    summary = summary_stats(collected)
    hist = failure_histogram(collected)

    print(f"Eval summary for run={run_name} robot={robot}")
    print(f"  episodes: {int(summary['num_episodes'])}")
    print(f"  success_rate: {summary['success_rate']:.3f}")
    print(f"  mean_episode_length: {summary['mean_episode_length']:.2f}")
    print(f"  mean_episode_return: {summary['mean_episode_return']:.3f}")
    print(f"  mean_final_orientation_error: {summary['mean_final_orientation_error']:.3f}")
    print("  failure_histogram:")
    for key in sorted(hist):
        print(f"    {key}: {hist[key]}")

    env.close()
    return summary


def main(args: EvalArgs) -> None:
    resolved_device = _resolve_device(args.device)
    if args.device == "cuda" and resolved_device == "cpu":
        print("CUDA requested but unavailable. Falling back to CPU for eval.")

    evaluate_checkpoint(
        run_name=args.run_name,
        robot=args.robot,
        checkpoint_path=args.checkpoint_path,
        episodes=max(1, args.episodes),
        seed=args.seed,
        device=resolved_device,
        debug=args.debug,
    )


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
