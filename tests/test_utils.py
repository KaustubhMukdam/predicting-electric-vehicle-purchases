"""Tests for src/utils.py."""

from __future__ import annotations

import os
import random

import numpy as np


def test_seed_everything_seeds_python_random():
    from src.utils import seed_everything

    seed_everything(123)
    a = random.random()
    seed_everything(123)
    b = random.random()
    assert a == b


def test_seed_everything_seeds_numpy():
    from src.utils import seed_everything

    seed_everything(123)
    a = np.random.rand(5).tolist()
    seed_everything(123)
    b = np.random.rand(5).tolist()
    assert a == b


def test_seed_everything_sets_pythonhashseed():
    from src.config import RANDOM_SEED
    from src.utils import seed_everything

    seed_everything()
    assert os.environ["PYTHONHASHSEED"] == str(RANDOM_SEED)


def test_seed_everything_accepts_custom_seed():
    from src.utils import seed_everything

    seed_everything(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_seed_everything_different_seeds_produce_different_sequences():
    from src.utils import seed_everything

    seed_everything(1)
    a = np.random.rand(10).tolist()
    seed_everything(2)
    b = np.random.rand(10).tolist()
    assert a != b
