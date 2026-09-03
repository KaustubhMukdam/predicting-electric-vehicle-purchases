"""Small utility helpers. Pure functions only — no I/O, no globals."""

from __future__ import annotations

import os
import random

import numpy as np

from src.config import RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and (if present) python hash seed.

    LightGBM is seeded per-call via its `seed` parameter, not here.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
