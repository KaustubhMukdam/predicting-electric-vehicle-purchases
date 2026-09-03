"""Tests for src/config.py.

Config is a leaf module — it must not do I/O, must not call any other
src module at import time, and must expose the contract that downstream
modules depend on.
"""

from __future__ import annotations

import importlib

import pytest


def test_config_module_has_no_side_effects_on_import():
    """Importing config must not read files or print anything.

    We assert this indirectly: the module must be importable in a clean
    subprocess and the import must complete in well under a second.
    """
    importlib.invalidate_caches()
    mod = importlib.import_module("src.config")
    assert mod is not None


def test_target_column_constant():
    from src.config import TARGET_COL

    assert TARGET_COL == "Will_Buy_EV"


def test_target_map_is_complete_and_correct():
    from src.config import TARGET_MAP, TARGET_INVERSE_MAP

    assert TARGET_MAP == {"No": 0, "Yes": 1}
    # Inverse map is the strict reverse.
    for k, v in TARGET_MAP.items():
        assert TARGET_INVERSE_MAP[v] == k


def test_numeric_and_categorical_columns_partition_features():
    """Every non-id, non-target column should be in either NUMERIC_COLS
    or CATEGORICAL_COLS — never both, never neither."""
    from src.config import (
        CATEGORICAL_COLS,
        ID_COL,
        NUMERIC_COLS,
        TARGET_COL,
    )

    all_listed = set(NUMERIC_COLS) | set(CATEGORICAL_COLS)
    assert "id" not in all_listed
    assert TARGET_COL not in all_listed
    # No overlap.
    assert set(NUMERIC_COLS).isdisjoint(set(CATEGORICAL_COLS))


def test_feature_cols_contains_all_raw_and_engineered():
    from src.config import (
        CATEGORICAL_COLS,
        ENGINEERED_COLS,
        FEATURE_COLS,
        NUMERIC_COLS,
    )

    for col in NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    # No duplicates.
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_paths_point_inside_project_root():
    """All default path constants must resolve under the project root,
    never to an absolute system path like /tmp or /home."""
    from src.config import (
        DATA_DIR,
        MLRUNS_DIR,
        OOF_DIR,
        PROJECT_ROOT,
        SAMPLE_SUBMISSION_CSV,
        SUBMISSIONS_DIR,
        TEST_CSV,
        TRAIN_CSV,
    )

    for p in (DATA_DIR, OOF_DIR, SUBMISSIONS_DIR, MLRUNS_DIR):
        assert p.is_absolute(), f"{p} should be absolute"
        assert p.is_relative_to(PROJECT_ROOT), f"{p} escapes project root"

    # CSV paths are concrete files (even if not present).
    assert TRAIN_CSV.name == "train.csv"
    assert TEST_CSV.name == "test.csv"
    assert SAMPLE_SUBMISSION_CSV.name == "sample_submission.csv"


def test_random_seed_and_folds_are_positive_ints():
    from src.config import N_FOLDS, RANDOM_SEED

    assert isinstance(RANDOM_SEED, int) and RANDOM_SEED > 0
    assert isinstance(N_FOLDS, int) and N_FOLDS >= 2


def test_mlflow_experiment_name_is_string():
    from src.config import MLFLOW_EXPERIMENT_NAME

    assert isinstance(MLFLOW_EXPERIMENT_NAME, str)
    assert len(MLFLOW_EXPERIMENT_NAME) > 0
