from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Literal

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from dexplay.envs.robots import get_robot_cfg
from dexplay.envs.tasks.allegro_xhand_reorient.config import ReorientTaskCfg
from dexplay.envs.tasks.allegro_xhand_reorient.task import ReorientVecEnv
from dexplay.rl.eval import evaluate_checkpoint
from dexplay.rl.ppo_config import build_ppo_train_cfg
from dexplay.utils.paths import checkpoint_dir, eval_log_path, run_dir, train_log_path
from dexplay.utils.seeding import seed_everything


@dataclass
class TrainArgs:
    run_name: str = "demo_allegro"
    robot: Literal["allegro", "xhand"] = "allegro"
    seed: int = 0
    total_timesteps: int = 200_000
    num_envs: int | None = None
    device: Literal["cuda", "cpu"] = "cuda"
    save_every: int = 50_000
    eval_episodes: int = 50
    debug: bool = False


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(args: TrainArgs) -> None:
    resolved_device = _resolve_device(args.device)
    if args.device == "cuda" and resolved_device == "cpu":
        print("CUDA requested but unavailable. Falling back to CPU for training.")

    default_envs = 1024 if resolved_device == "cuda" else 256
    num_envs = int(args.num_envs if args.num_envs is not None else default_envs)
    if args.debug:
        num_envs = 1

    seed_everything(args.seed)

    robot_cfg = get_robot_cfg(args.robot)
    task_cfg = ReorientTaskCfg()

    run_path = run_dir(args.run_name)
    ckpt_path = checkpoint_dir(args.run_name)

    train_path = train_log_path(args.run_name)
    eval_path = eval_log_path(args.run_name)
    if train_path.exists():
        train_path.unlink()
    if eval_path.exists():
        eval_path.unlink()

    env = ReorientVecEnv(
        robot_cfg=robot_cfg,
        task_cfg=task_cfg,
        num_envs=num_envs,
        seed=args.seed,
        run_name=args.run_name,
        split="train",
        device=resolved_device,
        debug=args.debug,
    )

    tmp_cfg = build_ppo_train_cfg(save_interval_iters=1)
    steps_per_iter = int(tmp_cfg["num_steps_per_env"])
    transitions_per_iter = num_envs * steps_per_iter
    total_iterations = max(1, math.ceil(args.total_timesteps / transitions_per_iter))

    save_interval_iters = max(1, args.save_every // max(1, transitions_per_iter))
    train_cfg = build_ppo_train_cfg(save_interval_iters=save_interval_iters)
    train_cfg["seed"] = args.seed

    cfg_dump = {
        "args": asdict(args),
        "resolved_device": resolved_device,
        "num_envs": num_envs,
        "train_cfg": train_cfg,
        "task_cfg": asdict(task_cfg),
        "robot_source": robot_cfg.source_reference,
    }
    (run_path / "train_config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")

    runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=None, device=resolved_device)
    if hasattr(runner, "disable_logs"):
        runner.disable_logs = True
    runner.logger_type = getattr(runner, "logger_type", "tensorboard")
    if hasattr(runner, "logger"):
        if hasattr(runner.logger, "disable_logs"):
            runner.logger.disable_logs = True
        if getattr(runner.logger, "log_dir", None) is None:
            runner.logger.log_dir = str(ckpt_path)
    runner.current_learning_iteration = max(
        1,
        int(getattr(runner, "current_learning_iteration", 0)),
    )

    done_timesteps = 0
    next_save = args.save_every
    latest_checkpoint = ckpt_path / "latest.pt"

    for iteration in range(1, total_iterations + 1):
        runner.learn(num_learning_iterations=1)
        done_timesteps += transitions_per_iter

        if done_timesteps >= next_save or iteration == total_iterations:
            ckpt_file = ckpt_path / f"model_{args.robot}_steps_{done_timesteps}.pt"
            runner.save(
                str(ckpt_file),
                infos={
                    "timesteps": done_timesteps,
                    "iteration": iteration,
                    "robot": args.robot,
                    "run_name": args.run_name,
                },
            )
            shutil.copy2(ckpt_file, latest_checkpoint)
            next_save += args.save_every
            print(f"Saved checkpoint: {ckpt_file}")

        if iteration % 10 == 0 or iteration == total_iterations:
            print(
                f"Iter {iteration}/{total_iterations} "
                f"timesteps={done_timesteps}/{args.total_timesteps}"
            )

    eval_episodes = max(1, int(args.eval_episodes))
    print(f"Training finished. Running deterministic evaluation ({eval_episodes} episodes).")
    evaluate_checkpoint(
        run_name=args.run_name,
        robot=args.robot,
        checkpoint_path=latest_checkpoint,
        episodes=eval_episodes,
        seed=args.seed,
        device=resolved_device,
        debug=args.debug,
    )

    env.close()


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
