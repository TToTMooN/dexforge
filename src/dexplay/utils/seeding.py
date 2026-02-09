from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some kernels are not deterministic on all platforms.
            pass
