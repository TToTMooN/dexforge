from __future__ import annotations


def build_ppo_train_cfg(save_interval_iters: int) -> dict:
    """Minimal rsl_rl PPO configuration for Phase-0."""
    return {
        "seed": 0,
        "num_steps_per_env": 24,
        "save_interval": max(1, int(save_interval_iters)),
        "logger": "tensorboard",
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["critic"],
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.8,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.02,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": True,
        },
    }
