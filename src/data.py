"""Data loading and schema validation.

Pure, side-effect-bounded: the only I/O is reading the two CSVs the
caller hands us. No path detection, no Kaggle-specific branching here
— that lives in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    ID_COL,
    NUMERIC_COLS,
    TARGET_COL,
    TARGET_MAP,
)

# Columns we expect in train (everything except the target).
_EXPECTED_TRAIN_BASE = {ID_COL, TARGET_COL, *NUMERIC_COLS, *CATEGORICAL_COLS}
_EXPECTED_TEST_BASE = {ID_COL, *NUMERIC_COLS, *CATEGORICAL_COLS}


def _validate_schema(df: pd.DataFrame, expected: set[str], name: str) -> None:
    """Raise ValueError if `df` is missing required columns or has
    unexpected ones. Catches silent schema drift early."""
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{name} CSV schema mismatch. "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def load_data(
    train_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and test CSVs, validate schema, map the target to {0, 1},
    coerce `Environmental_Concern_Level` from float to int.

    Returns
    -------
    (train_df, test_df)
        train_df has a numeric `Will_Buy_EV` column with values in {0, 1}.
        test_df has no target column.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    _validate_schema(train, _EXPECTED_TRAIN_BASE, "train")
    _validate_schema(test, _EXPECTED_TEST_BASE, "test")

    # Map the target. Anything not in TARGET_MAP will surface as NaN
    # and trigger an explicit error below — fail loud, not silent.
    train[TARGET_COL] = train[TARGET_COL].map(TARGET_MAP)
    if train[TARGET_COL].isnull().any():
        bad = train.loc[train[TARGET_COL].isnull(), TARGET_COL].head().tolist()
        raise ValueError(
            f"Target column contains values not in {TARGET_MAP}: {bad}"
        )
    train[TARGET_COL] = train[TARGET_COL].astype("int64")

    # Coerce Environmental_Concern_Level from float64 -> int64.
    # Every value is an integer 1-5; storing as int is what LightGBM expects
    # for native categorical handling and saves a few MB.
    for df in (train, test):
        if df["Environmental_Concern_Level"].dtype.kind == "f":
            df["Environmental_Concern_Level"] = (
                df["Environmental_Concern_Level"].astype("int64")
            )

    return train, test
