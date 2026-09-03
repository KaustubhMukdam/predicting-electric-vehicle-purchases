"""Submission file generation.

The only public function is `make_submission`, which takes the model's
test-set predictions and writes a Kaggle-format submission CSV whose
column layout exactly matches the provided template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from src.config import TARGET_COL


PathLike = Union[str, Path]


def make_submission(
    test_ids: Iterable,
    test_pred: np.ndarray,
    template_path: PathLike,
    out_path: PathLike,
) -> Path:
    """Build a Kaggle submission CSV.

    Parameters
    ----------
    test_ids : iterable of length n
        Row identifiers (anything pandas can convert to a column). Must be
        the same length as `test_pred` and free of NaNs.
    test_pred : np.ndarray of shape (n,)
        Predicted probabilities. Must be in [0, 1] and free of NaN/inf.
    template_path : path-like
        Path to the sample submission CSV. The output will have the
        same column order, with the target column replaced by `test_pred`.
    out_path : path-like
        Where to write the output CSV. Parent directories are created
        if they don't exist.

    Returns
    -------
    Path
        The absolute, resolved `out_path`.

    Raises
    ------
    ValueError
        If `test_ids` and `test_pred` have different lengths, if any
        prediction is outside [0, 1], if any prediction is NaN/inf, or
        if any id is NaN.
    FileNotFoundError
        If `template_path` does not exist.
    KeyError
        If the template does not contain the target column.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    test_pred = np.asarray(test_pred, dtype=np.float64)
    if test_pred.ndim != 1:
        raise ValueError(
            f"test_pred must be 1-D, got shape {test_pred.shape}"
        )

    test_ids = pd.Series(test_ids).reset_index(drop=True)
    if len(test_ids) != len(test_pred):
        raise ValueError(
            f"Length mismatch: test_ids has {len(test_ids)} rows, "
            f"test_pred has {len(test_pred)}"
        )
    if test_ids.isna().any():
        raise ValueError("test_ids contains NaN values")
    if np.isnan(test_pred).any() or np.isinf(test_pred).any():
        raise ValueError("test_pred contains NaN or inf values")
    if not ((test_pred >= 0) & (test_pred <= 1)).all():
        raise ValueError("test_pred contains values outside [0, 1]")

    template = pd.read_csv(template_path)
    if TARGET_COL not in template.columns:
        raise KeyError(
            f"Template is missing target column {TARGET_COL!r}. "
            f"Columns present: {list(template.columns)}"
        )

    # Build the output frame with the same column order as the template.
    out_df = template.copy()
    out_df[TARGET_COL] = test_pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path.resolve()
