"""
train.py
============================================================
Kaggle Playground Series S6E9
Predicting Electric Vehicle Purchases

Clean training entry point for the current src/ project.

The important design choice here is that FEATURE_COLS from config.py is
NOT assumed to describe the raw CSV. features_v2.py creates additional
columns, so this script builds the final feature list AFTER feature
engineering.

Local:
    python3 train.py

Expected local structure:
    project_root/
    ├── train.py
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── cv.py
    │   ├── data.py
    │   ├── features_v2.py
    │   ├── predict.py
    │   └── train_lgbm.py
    ├── data/
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    └── outputs/

MLflow is disabled for the normal training run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
# 1. PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# 2. PROJECT IMPORTS
# ---------------------------------------------------------------------

from src.config import (  # noqa: E402
    CATEGORICAL_COLS,
    ID_COL,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)
from src.cv import make_folds  # noqa: E402
from src.data import load_data  # noqa: E402
from src.features_v2 import build_features  # noqa: E402
from src.predict import make_submission  # noqa: E402
from src.train_lgbm import DEFAULT_LGBM_PARAMS, train_lgbm  # noqa: E402


# ---------------------------------------------------------------------
# 3. SETTINGS
# ---------------------------------------------------------------------

NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 200

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_PATH = OUTPUT_DIR / "submission_lgbm_v2.csv"
OOF_PATH = OUTPUT_DIR / "oof_lgbm_v2.npy"
TEST_PRED_PATH = OUTPUT_DIR / "test_pred_lgbm_v2.npy"


# ---------------------------------------------------------------------
# 4. KAGGLE DETECTION
# ---------------------------------------------------------------------

def is_kaggle() -> bool:
    """Return True only inside a normal Kaggle runtime."""
    return Path("/kaggle/input").is_dir()


def resolve_paths() -> tuple[Path, Path, Path]:
    """
    Resolve data paths.

    Local:
        Uses paths from src.config.py.

    Kaggle:
        Searches /kaggle/input for a directory containing all three
        competition files.
    """
    if not is_kaggle():
        return (
            Path(TRAIN_CSV),
            Path(TEST_CSV),
            Path(SAMPLE_SUBMISSION_CSV),
        )

    preferred_dirs = [
        Path("/kaggle/input/playground-series-s6e9"),
        Path("/kaggle/input/playground-series-s6e9-2026"),
    ]

    for data_dir in preferred_dirs:
        paths = (
            data_dir / "train.csv",
            data_dir / "test.csv",
            data_dir / "sample_submission.csv",
        )
        if all(p.exists() for p in paths):
            return paths

    # Generic fallback for attached Kaggle datasets.
    for train_path in Path("/kaggle/input").glob("**/train.csv"):
        data_dir = train_path.parent
        test_path = data_dir / "test.csv"
        sample_path = data_dir / "sample_submission.csv"

        if test_path.exists() and sample_path.exists():
            return train_path, test_path, sample_path

    raise FileNotFoundError(
        "Could not find train.csv, test.csv and sample_submission.csv "
        "under /kaggle/input."
    )


# ---------------------------------------------------------------------
# 5. RAW DATA VALIDATION
# ---------------------------------------------------------------------

def validate_raw_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate only columns that should actually exist in the raw CSVs.

    Do NOT validate engineered columns here. Those are created later by
    features_v2.build_features().
    """
    if ID_COL not in train.columns or TARGET_COL not in train.columns:
        raise ValueError(
            f"Train must contain {ID_COL!r} and {TARGET_COL!r}. "
            f"Found: {list(train.columns)}"
        )

    if ID_COL not in test.columns:
        raise ValueError(
            f"Test must contain {ID_COL!r}. Found: {list(test.columns)}"
        )

    if train[ID_COL].isna().any():
        raise ValueError("Train ID contains missing values.")

    if test[ID_COL].isna().any():
        raise ValueError("Test ID contains missing values.")

    if train[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs detected in train.")

    if test[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs detected in test.")

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Train/test data cannot be empty.")


# ---------------------------------------------------------------------
# 6. FINAL FEATURE DISCOVERY
# ---------------------------------------------------------------------

def get_feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Determine the final feature columns AFTER feature engineering.

    This deliberately does not depend on FEATURE_COLS in config.py,
    because the current features_v2.py contains a different engineered
    feature set from the original V1 configuration.

    Feature rule:
        everything except ID and target.

    Categorical rule:
        original configured categoricals + any newly-created string /
        categorical columns.
    """
    if TARGET_COL not in train.columns:
        raise ValueError(
            f"Target {TARGET_COL!r} missing after feature engineering."
        )

    if TARGET_COL in test.columns:
        raise ValueError(
            "Test dataframe unexpectedly contains the target column."
        )

    feature_columns = [
        col
        for col in train.columns
        if col not in {ID_COL, TARGET_COL}
    ]

    missing_in_test = [
        col for col in feature_columns
        if col not in test.columns
    ]

    if missing_in_test:
        raise ValueError(
            "Feature engineering produced columns that are absent from "
            f"test: {missing_in_test}"
        )

    # Start with configured categoricals.
    categorical_columns = [
        col
        for col in CATEGORICAL_COLS
        if col in feature_columns
    ]

    # Add newly-created string/category interaction features.
    for col in feature_columns:
        if (
            pd.api.types.is_object_dtype(train[col])
            or pd.api.types.is_categorical_dtype(train[col])
        ):
            if col not in categorical_columns:
                categorical_columns.append(col)

    return feature_columns, categorical_columns


# ---------------------------------------------------------------------
# 7. TARGET VALIDATION
# ---------------------------------------------------------------------

def validate_target(y: pd.Series) -> np.ndarray:
    """
    Validate the already-encoded target returned by load_data().
    """
    values = y.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError("Target contains NaN or infinite values.")

    unique = np.unique(values)

    if not set(unique).issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0/1 after load_data(). "
            f"Found: {unique}"
        )

    return values.astype(np.int8, copy=False)


# ---------------------------------------------------------------------
# 8. MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 72)
    print("EV PURCHASE PREDICTION — LIGHTGBM V2")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    train_path, test_path, sample_path = resolve_paths()

    print("\nEnvironment")
    print("-" * 72)
    print(f"Running on Kaggle : {is_kaggle()}")
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Train             : {train_path}")
    print(f"Test              : {test_path}")
    print(f"Sample submission : {sample_path}")
    print(f"Output directory  : {OUTPUT_DIR}")

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("1. LOAD DATA")
    print("=" * 72)

    train_raw, test_raw = load_data(
        train_path,
        test_path,
    )

    print(f"Train shape: {train_raw.shape}")
    print(f"Test shape : {test_raw.shape}")

    validate_raw_data(train_raw, test_raw)
    print("Raw-data validation: PASSED")

    # ---------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("2. TARGET")
    print("=" * 72)

    y = validate_target(train_raw[TARGET_COL])

    print(f"Positive samples : {int(y.sum()):,}")
    print(f"Negative samples : {int((1 - y).sum()):,}")
    print(f"Positive rate    : {y.mean():.6f}")

    # ---------------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("3. FEATURE ENGINEERING — V2")
    print("=" * 72)

    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    feature_cols, categorical_cols = get_feature_columns(
        train_feat,
        test_feat,
    )

    print(f"Final feature count : {len(feature_cols)}")
    print(f"Categorical count   : {len(categorical_cols)}")

    print("\nFeatures:")
    for i, col in enumerate(feature_cols, start=1):
        suffix = " [categorical]" if col in categorical_cols else ""
        print(f"  {i:02d}. {col}{suffix}")

    print("\nFeature validation: PASSED")

    # ---------------------------------------------------------------
    # CV folds
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("4. STRATIFIED CV")
    print("=" * 72)

    folds = make_folds(
        y,
        n_splits=N_FOLDS,
        seed=RANDOM_SEED,
    )

    unique_folds = np.unique(folds)

    if len(unique_folds) != N_FOLDS:
        raise ValueError(
            f"Expected {N_FOLDS} folds, got {len(unique_folds)}."
        )

    for fold in unique_folds:
        mask = folds == fold
        print(
            f"Fold {fold}: "
            f"{mask.sum():,} rows | "
            f"positive rate = {y[mask].mean():.6f}"
        )

    # ---------------------------------------------------------------
    # LightGBM parameters
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("5. LIGHTGBM")
    print("=" * 72)

    params = DEFAULT_LGBM_PARAMS.copy()

    params.update(
        {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.90,
            "bagging_fraction": 0.90,
            "bagging_freq": 5,
            "seed": RANDOM_SEED,
            "verbosity": -1,
        }
    )

    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print(f"\nMax boosting rounds : {NUM_BOOST_ROUND}")
    print(f"Early stopping      : {EARLY_STOPPING_ROUNDS}")
    print("MLflow tracking     : DISABLED")

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("6. TRAINING")
    print("=" * 72)

    oof, test_pred, metrics = train_lgbm(
        train=train_feat,
        test=test_feat,
        folds=folds,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        params=params,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        categorical_cols=categorical_cols,
        tracking_enabled=False,
        run_name=None,
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("7. OOF EVALUATION")
    print("=" * 72)

    pooled_auc = roc_auc_score(y, oof)

    print(f"Mean fold AUC : {metrics['cv_auc_mean']:.6f}")
    print(f"Std fold AUC  : {metrics['cv_auc_std']:.6f}")
    print(f"Pooled OOF AUC: {pooled_auc:.6f}")

    print("\nPer-fold AUC:")
    for fold, auc in enumerate(metrics["fold_aucs"]):
        print(f"  Fold {fold}: {auc:.6f}")

    # ---------------------------------------------------------------
    # Prediction validation
    # ---------------------------------------------------------------

    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions contain NaN/Inf.")

    if not np.isfinite(test_pred).all():
        raise ValueError("Test predictions contain NaN/Inf.")

    if not ((0 <= oof).all() and (oof <= 1).all()):
        raise ValueError("OOF predictions are outside [0, 1].")

    if not ((0 <= test_pred).all() and (test_pred <= 1).all()):
        raise ValueError("Test predictions are outside [0, 1].")

    # ---------------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("8. SAVE PREDICTIONS")
    print("=" * 72)

    np.save(OOF_PATH, oof)
    np.save(TEST_PRED_PATH, test_pred)

    print(f"OOF predictions : {OOF_PATH}")
    print(f"Test predictions: {TEST_PRED_PATH}")

    # ---------------------------------------------------------------
    # Submission
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("9. CREATE SUBMISSION")
    print("=" * 72)

    submission_path = make_submission(
        test_ids=test_feat[ID_COL],
        test_pred=test_pred,
        template_path=sample_path,
        out_path=SUBMISSION_PATH,
    )

    submission = pd.read_csv(submission_path)

    if len(submission) != len(test_raw):
        raise ValueError(
            f"Submission has {len(submission)} rows; "
            f"expected {len(test_raw)}."
        )

    if not submission[ID_COL].equals(
        test_raw[ID_COL].reset_index(drop=True)
    ):
        raise ValueError(
            "Submission IDs do not match test IDs in order."
        )

    if TARGET_COL not in submission.columns:
        raise ValueError(
            f"Submission missing target column {TARGET_COL!r}."
        )

    print(f"Submission: {submission_path}")
    print(f"Rows      : {len(submission):,}")
    print(f"Pred min  : {test_pred.min():.6f}")
    print(f"Pred max  : {test_pred.max():.6f}")
    print(f"Pred mean : {test_pred.mean():.6f}")

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(f"\nPooled OOF AUC : {pooled_auc:.6f}")
    print(f"Submission      : {submission_path}")
    print()


if __name__ == "__main__":
    main()
