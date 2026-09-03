"""Cross-validation split generation.

Returns an integer array of fold assignments (0..n_splits-1), stratified
on the target. Pure function — no fit state, no leakage between folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import N_FOLDS, RANDOM_SEED


def make_folds(
    y: np.ndarray,
    n_splits: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Stratified K-Fold split.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target vector. Integer or string labels; anything with `.ravel()`
        and that `StratifiedKFold` can hash.
    n_splits : int
        Number of folds. Default from config.
    seed : int
        Random state. Default from config.

    Returns
    -------
    np.ndarray of shape (n_samples,), dtype int64
        Fold assignment per row. Values in 0..n_splits-1.

    Raises
    ------
    ValueError if y is empty or too small for `n_splits`.
    """
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("y is empty")
    if y.size < n_splits:
        raise ValueError(
            f"y has {y.size} samples but n_splits={n_splits}; "
            "need at least n_splits samples"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(y.shape[0], -1, dtype=np.int64)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(y.shape[0]), y)):
        folds[val_idx] = k
    return folds
