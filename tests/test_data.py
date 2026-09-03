"""Tests for src/data.py.

These tests use the real data/ CSVs (per the user's choice). The CSVs
are gitignored and assumed to be present locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.config import (
    CATEGORICAL_COLS,
    ID_COL,
    NUMERIC_COLS,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)


# Skip the whole module if the data files aren't where we expect.
pytestmark = pytest.mark.skipif(
    not TRAIN_CSV.exists() or not TEST_CSV.exists(),
    reason="data/train.csv and data/test.csv not present",
)


def test_load_data_returns_two_dataframes():
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_load_data_preserves_row_counts():
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    # Hardcoded against the known dataset shape; if the dataset is
    # regenerated, this test will fail loudly and force a review.
    assert len(train) == 668_665
    assert len(test) == 286_571


def test_load_data_train_has_target_column():
    from src.data import load_data

    train, _ = load_data(TRAIN_CSV, TEST_CSV)
    assert TARGET_COL in train.columns


def test_load_data_test_does_not_have_target_column():
    from src.data import load_data

    _, test = load_data(TRAIN_CSV, TEST_CSV)
    assert TARGET_COL not in test.columns


def test_load_data_target_is_binary_int():
    from src.data import load_data

    train, _ = load_data(TRAIN_CSV, TEST_CSV)
    assert train[TARGET_COL].dtype.kind in ("i", "u")  # signed/unsigned int
    assert set(train[TARGET_COL].unique()).issubset({0, 1})


def test_load_data_id_columns_are_unique_within_each_split():
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    assert train[ID_COL].is_unique
    assert test[ID_COL].is_unique
    # No overlap between train ids and test ids.
    assert set(train[ID_COL]).isdisjoint(set(test[ID_COL]))


def test_load_data_does_not_introduce_missing_values():
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    assert train.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0


def test_load_data_categorical_columns_have_consistent_levels():
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    for col in CATEGORICAL_COLS:
        train_levels = set(train[col].unique())
        test_levels = set(test[col].unique())
        # Same set of levels (order doesn't matter).
        assert train_levels == test_levels, (
            f"Category mismatch in {col}: "
            f"train_only={train_levels - test_levels}, "
            f"test_only={test_levels - train_levels}"
        )


def test_load_data_numeric_columns_are_numeric_dtype():
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    for col in NUMERIC_COLS:
        assert pd.api.types.is_numeric_dtype(train[col]), (
            f"train[{col}] dtype is {train[col].dtype}, expected numeric"
        )
        assert pd.api.types.is_numeric_dtype(test[col]), (
            f"test[{col}] dtype is {test[col].dtype}, expected numeric"
        )


def test_load_data_class_distribution_matches_known_imbalance():
    from src.data import load_data

    train, _ = load_data(TRAIN_CSV, TEST_CSV)
    pos_rate = train[TARGET_COL].mean()
    # Known from EDA: 17.46% positive. Allow a small tolerance.
    assert 0.17 < pos_rate < 0.18


def test_load_data_environmental_concern_is_int_after_load():
    """The raw CSV stores this column as float64. The loader should
    coerce it to int64 since every value is an integer 1-5."""
    from src.data import load_data

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    assert train["Environmental_Concern_Level"].dtype.kind in ("i", "u")
    assert test["Environmental_Concern_Level"].dtype.kind in ("i", "u")


def test_load_data_raises_on_missing_file():
    from src.data import load_data

    with pytest.raises((FileNotFoundError, OSError)):
        load_data(Path("/nonexistent/train.csv"), TEST_CSV)  # type: ignore[name-defined]


def test_load_data_raises_on_unexpected_columns():
    """If the train CSV is missing a required column, the loader must
    raise rather than silently produce a broken DataFrame."""
    from src.data import load_data

    # Write a tiny CSV missing the target column.
    bad_path = SAMPLE_SUBMISSION_CSV.parent / "_bad_train.csv"
    bad_path.write_text("id,Age\n1,30\n")
    try:
        with pytest.raises((KeyError, ValueError)):
            load_data(bad_path, TEST_CSV)
    finally:
        bad_path.unlink(missing_ok=True)


def test_load_data_is_idempotent():
    """Calling load_data twice on the same files produces equal DataFrames."""
    from src.data import load_data

    a, b = load_data(TRAIN_CSV, TEST_CSV)
    a2, b2 = load_data(TRAIN_CSV, TEST_CSV)
    pd.testing.assert_frame_equal(a, a2)
    pd.testing.assert_frame_equal(b, b2)
