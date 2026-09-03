"""Tests for src/cv.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _toy_y(n: int = 1000, pos_rate: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < pos_rate).astype(int)


def test_make_folds_returns_ndarray():
    from src.cv import make_folds

    y = _toy_y()
    folds = make_folds(y, n_splits=5, seed=42)
    assert isinstance(folds, np.ndarray)


def test_make_folds_length_matches_y():
    from src.cv import make_folds

    y = _toy_y(n=500)
    folds = make_folds(y, n_splits=5, seed=42)
    assert len(folds) == 500


def test_make_folds_values_are_0_to_k_minus_1():
    from src.cv import make_folds

    y = _toy_y(n=2000)
    folds = make_folds(y, n_splits=5, seed=42)
    assert set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4}


def test_make_folds_are_a_partition():
    """Each row is in exactly one fold."""
    from src.cv import make_folds

    y = _toy_y(n=1000)
    folds = make_folds(y, n_splits=5, seed=42)
    assert len(folds) == 1000
    # No -1 sentinels.
    assert (folds >= 0).all()


def test_make_folds_stratified_preserves_class_balance_per_fold():
    """Each fold's positive rate should be within a few % of the overall rate."""
    from src.cv import make_folds

    y = _toy_y(n=10_000, pos_rate=0.175, seed=0)
    folds = make_folds(y, n_splits=5, seed=42)

    overall_pos = y.mean()
    for k in range(5):
        fold_pos = y[folds == k].mean()
        assert abs(fold_pos - overall_pos) < 0.01, (
            f"Fold {k} has pos rate {fold_pos:.4f}, expected ~{overall_pos:.4f}"
        )


def test_make_folds_is_deterministic():
    from src.cv import make_folds

    y = _toy_y(n=2000, seed=0)
    a = make_folds(y, n_splits=5, seed=42)
    b = make_folds(y, n_splits=5, seed=42)
    np.testing.assert_array_equal(a, b)


def test_make_folds_different_seeds_produce_different_partitions():
    from src.cv import make_folds

    y = _toy_y(n=2000, seed=0)
    a = make_folds(y, n_splits=5, seed=42)
    b = make_folds(y, n_splits=5, seed=43)
    assert not np.array_equal(a, b)


def test_make_folds_each_fold_has_reasonable_size():
    """No fold should be wildly larger or smaller than 1/n_splits."""
    from src.cv import make_folds

    y = _toy_y(n=1001)  # not divisible by 5
    folds = make_folds(y, n_splits=5, seed=42)
    sizes = np.bincount(folds)
    expected = len(y) / 5
    for k, size in enumerate(sizes):
        assert abs(size - expected) <= 1, (
            f"Fold {k} has size {size}, expected ~{expected}"
        )


def test_make_folds_validates_inputs():
    from src.cv import make_folds

    with pytest.raises(ValueError):
        make_folds(np.array([0, 1, 0]), n_splits=5, seed=42)  # too small

    with pytest.raises(ValueError):
        make_folds(np.array([]), n_splits=5, seed=42)  # empty
