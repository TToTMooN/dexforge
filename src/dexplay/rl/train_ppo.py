from __future__ import annotations

import argparse
import json
import math
import os
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from rsl_rl.runners import OnPolicyRunner

from dexplay.envs.robots import get_robot_cfg
from dexplay.envs.tasks.allegro_xhand_reorient.config import ReorientTaskCfg
from dexplay.envs.tasks.allegro_xhand_reorient.task import ReorientVecEnv
from dexplay.rl.eval import evaluate_checkpoint
from dexplay.rl.ppo_config import build_ppo_train_cfg
from dexplay.utils.paths import checkpoint_dir, eval_log_path, run_dir, train_log_path
from dexplay.utils.seeding import seed_everything


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on dexplay Phase-0")
    parser.add_argument("--run_name", type=str, default="demo_allegro")
    parser.add_argument("--robot", type=str, choices=["allegro", "xhand"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=200000)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--backend", type=str, choices=["auto", "mjlab", "mujoco"], default="auto")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_device = _resolve_device(args.device)
    if args.device == "cuda" and resolved_device == "cpu":
        print("CUDA requested but unavailable. Falling back to CPU for training.")

    default_envs = 1024 if resolved_device == "cuda" else 256
    num_envs = args.num_envs if args.num_envs is not None else default_envs
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
        backend=args.backend,
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
        "args": {
            "run_name": args.run_name,
            "robot": args.robot,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "num_envs": num_envs,
            "device": resolved_device,
            "save_every": args.save_every,
            "backend": args.backend,
        },
        "train_cfg": train_cfg,
    }
    (run_path / "train_config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")

    runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=None, device=resolved_device)
    # Cross-version guard for rsl_rl logger behavior differences.
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
            ckpt_file = ckpt_path / f"model_steps_{done_timesteps}.pt"
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

    print("Training finished. Running deterministic evaluation (50 episodes).")
    evaluate_checkpoint(
        run_name=args.run_name,
        robot=args.robot,
        checkpoint_path=latest_checkpoint,
        episodes=50,
        seed=args.seed,
        device=resolved_device,
        backend=args.backend,
        debug=args.debug,
    )

    env.close()


if __name__ == "__main__":
    main()
