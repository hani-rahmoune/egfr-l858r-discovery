"""Set random seeds for Python, NumPy, and (optionally) PyTorch in one call."""

from __future__ import annotations

import random

import numpy as np

from src.utils.config import get_seed


def set_all_seeds(seed: int | None = None) -> int:
    if seed is None:
        seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
