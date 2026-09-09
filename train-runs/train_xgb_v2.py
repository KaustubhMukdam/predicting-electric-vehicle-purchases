"""
train_xgb_v2.py
============================================================
Kaggle Playground Series S6E9
Predicting Electric Vehicle Purchases

Experiment 4B
-------------
XGBoost with:
    * Existing V2 feature engineering
    * Native categorical feature handling
    * GPU on Kaggle / CPU locally
    * Same 5-fold stratified CV protocol
    * ROC-AUC + early stopping

IMPORTANT:
This is a controlled experiment. Do not add new features or tune
multiple parameters simultaneously after this run. Compare the
OOF AUC against XGBoost V1 and LightGBM V2 first.

Outputs:
    outputs/oof_xgb_v2.npy
    outputs/test_pred_xgb_v2.npy
    outputs/submission_xgb_v2.csv
============================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

SEED = 42
N_FOLDS = 5

N_ESTIMATORS = 5000
EARLY_STOPPING = 200

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",

    "learning_rate": 0.05,

    # 4B changes versus XGB V1:
    # - deeper trees
    # - smaller min_child_weight
    "max_depth": 10,
    "min_child_weight": 5,

    "subsample": 0.90,
    "colsample_bytree": 0.90,

    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    "max_bin": 256,

    "tree_method": "hist",

    "random_state": SEED,
    "n_jobs": -1,
}


# ============================================================
# ENVIRONMENT / PATHS
# ============================================================

IS_KAGGLE = Path("/kaggle").exists()


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent

    for p in [here, *here.parents]:
        if (p / "src").is_dir():
            return p

    raise FileNotFoundError(
        "Could not find project root containing src/."
    )


ROOT = Path("/kaggle/working") if IS_KAGGLE else find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# On Kaggle, src may be attached as a separate dataset.
def configure_kaggle_src() -> None:
    if not IS_KAGGLE:
        return

    candidates = [
        Path("/kaggle/input/ev-purchase-src/src"),
        Path("/kaggle/input/ev-purchase-src-final/src"),
        Path(
            "/kaggle/input/datasets/"
            "kaustubhmukdam/ev-purchase-src-final/src"
        ),
    ]

    src_dir = next(
        (p for p in candidates if p.is_dir()),
        None,
    )

    if src_dir is None:
        matches = list(Path("/kaggle/input").glob("*/src"))
        src_dir = matches[0] if matches else None

    if src_dir is None:
        raise FileNotFoundError(
            "Could not find src/ under /kaggle/input. "
            "Attach the project source-code dataset."
        )

    src_parent = src_dir.parent

    if str(src_parent) not in sys.path:
        sys.path.insert(0, str(src_parent))


configure_kaggle_src()


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data import load_data
from src.features_v2 import build_features
from src.cv import make_folds
from src.predict import make_submission


# ============================================================
# DATA PATHS
# ============================================================

def resolve_data_paths():
    if not IS_KAGGLE:
        data_dir = ROOT / "data"

        return (
            data_dir / "train.csv",
            data_dir / "test.csv",
            data_dir / "sample_submission.csv",
        )

    # Do not assume a particular Kaggle dataset slug.
    # Find a directory containing all three competition files.
    input_root = Path("/kaggle/input")

    for train_path in input_root.rglob("train.csv"):
        parent = train_path.parent

        test_path = parent / "test.csv"
        sample_path = parent / "sample_submission.csv"

        if test_path.exists() and sample_path.exists():
            return train_path, test_path, sample_path

    raise FileNotFoundError(
        "Could not locate train.csv, test.csv and "
        "sample_submission.csv under /kaggle/input."
    )


TRAIN_PATH, TEST_PATH, SAMPLE_PATH = resolve_data_paths()


# ============================================================
# TARGET
# ============================================================

def encode_target(y: pd.Series) -> np.ndarray:
    """
    Defensive target conversion.

    src.data.load_data() already converts the competition target
    to 0/1 in the current project, but this keeps the script robust
    if that implementation changes.
    """

    if pd.api.types.is_numeric_dtype(y):
        values = y.to_numpy()

        if not set(np.unique(values)).issubset({0, 1}):
            raise ValueError(
                f"Target contains values other than 0/1: "
                f"{np.unique(values)}"
            )

        return values.astype(np.int8)

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    }

    encoded = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if encoded.isna().any():
        raise ValueError(
            "Could not encode target values: "
            f"{y[encoded.isna()].unique()}"
        )

    return encoded.to_numpy(dtype=np.int8)


# ============================================================
# NATIVE CATEGORICAL PREPARATION
# ============================================================

def prepare_native_categorical_features(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    target_col: str,
):
    """
    Prepare V2 data for XGBoost native categorical handling.

    We deliberately DO NOT one-hot encode categories.

    Every categorical feature is converted to pandas 'category',
    with train/test sharing the same category vocabulary.

    This is important because XGBoost 3.x supports native categorical
    splits when enable_categorical=True.
    """

    drop_cols = {"id", target_col}

    train_x = train_feat.drop(
        columns=[
            c for c in drop_cols
            if c in train_feat.columns
        ]
    ).copy()

    test_x = test_feat.drop(
        columns=[
            c for c in drop_cols
            if c in test_feat.columns
        ]
    ).copy()

    if list(train_x.columns) != list(test_x.columns):
        raise ValueError(
            "Train/test V2 feature columns do not match."
        )

    categorical_cols = []
    numeric_cols = []

    for col in train_x.columns:
        train_is_cat = (
            pd.api.types.is_object_dtype(train_x[col])
            or isinstance(
                train_x[col].dtype,
                pd.CategoricalDtype,
            )
        )

        test_is_cat = (
            pd.api.types.is_object_dtype(test_x[col])
            or isinstance(
                test_x[col].dtype,
                pd.CategoricalDtype,
            )
        )

        if train_is_cat or test_is_cat:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    # Numeric columns: explicit float32 to reduce memory.
    for col in numeric_cols:
        train_x[col] = pd.to_numeric(
            train_x[col],
            errors="raise",
        ).astype(np.float32)

        test_x[col] = pd.to_numeric(
            test_x[col],
            errors="raise",
        ).astype(np.float32)

    # Categorical columns:
    # use a shared vocabulary derived from train + test values.
    # No target information is involved.
    for col in categorical_cols:
        combined = pd.concat(
            [
                train_x[col].astype(str),
                test_x[col].astype(str),
            ],
            ignore_index=True,
        )

        categories = pd.Index(
            combined.drop_duplicates()
        )

        train_x[col] = pd.Categorical(
            train_x[col].astype(str),
            categories=categories,
        )

        test_x[col] = pd.Categorical(
            test_x[col].astype(str),
            categories=categories,
        )

    return (
        train_x,
        test_x,
        numeric_cols,
        categorical_cols,
    )


# ============================================================
# XGBOOST VERSION / GPU CHECK
# ============================================================

def configure_device():
    """
    Kaggle -> CUDA
    Local -> CPU

    XGBoost >= 2 uses:
        tree_method='hist'
        device='cuda'
    """

    params = XGB_PARAMS.copy()

    if IS_KAGGLE:
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"

    return params


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    print()
    print("=" * 78)
    print("EV PURCHASE — XGBOOST EXPERIMENT 4B")
    print("=" * 78)

    print()
    print("Environment :", "Kaggle" if IS_KAGGLE else "Local")
    print("XGBoost     :", xgb.__version__)
    print("Train path  :", TRAIN_PATH)
    print("Test path   :", TEST_PATH)

    params = configure_device()

    print("Device      :", params["device"].upper())

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("1. LOAD DATA")
    print("=" * 78)

    # IMPORTANT:
    # Current src.data.load_data() accepts only train_path/test_path.
    train_raw, test_raw = load_data(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
    )

    print("Train shape:", train_raw.shape)
    print("Test shape :", test_raw.shape)

    target_col = "Will_Buy_EV"

    if target_col not in train_raw.columns:
        raise KeyError(
            f"Target column {target_col!r} not found."
        )

    y = encode_target(train_raw[target_col])

    print()
    print("Positive samples:", f"{int(y.sum()):,}")
    print(
        "Negative samples:",
        f"{int((1 - y).sum()):,}",
    )
    print(
        "Positive rate   :",
        f"{y.mean():.6f}",
    )

    # --------------------------------------------------------
    # 2. V2 FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("2. BUILD V2 FEATURES")
    print("=" * 78)

    train_feat = build_features(train_raw.copy())
    test_feat = build_features(test_raw.copy())

    feature_cols = [
        c
        for c in train_feat.columns
        if c not in {"id", target_col}
    ]

    missing_test = [
        c
        for c in feature_cols
        if c not in test_feat.columns
    ]

    if missing_test:
        raise ValueError(
            "Test is missing V2 features: "
            f"{missing_test}"
        )

    print("V2 feature count:", len(feature_cols))
    print("Train V2 shape :", train_feat.shape)
    print("Test V2 shape  :", test_feat.shape)

    # --------------------------------------------------------
    # 3. NATIVE CATEGORICAL FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("3. NATIVE CATEGORICAL PREPARATION")
    print("=" * 78)

    X, X_test, numeric_cols, categorical_cols = (
        prepare_native_categorical_features(
            train_feat,
            test_feat,
            target_col,
        )
    )

    print("Numeric columns    :", len(numeric_cols))
    print("Categorical columns:", len(categorical_cols))

    print()
    print("Categorical columns:")
    for col in categorical_cols:
        print("  -", col)

    print()
    print(
        "Matrix representation: pandas DataFrame "
        "(NO one-hot encoding)"
    )

    # --------------------------------------------------------
    # 4. FOLDS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("4. STRATIFIED 5-FOLD CV")
    print("=" * 78)

    folds = make_folds(
        y,
        n_splits=N_FOLDS,
        seed=SEED,
    )

    for fold in range(N_FOLDS):
        mask = folds == fold

        print(
            f"Fold {fold}: "
            f"{mask.sum():,} rows | "
            f"positive rate = {y[mask].mean():.6f}"
        )

    # --------------------------------------------------------
    # 5. PARAMETERS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("5. XGBOOST 4B PARAMETERS")
    print("=" * 78)

    for key, value in params.items():
        print(f"{key:20s}: {value}")

    print(f"{'n_estimators':20s}: {N_ESTIMATORS}")
    print(f"{'early_stopping':20s}: {EARLY_STOPPING}")
    print(f"{'enable_categorical':20s}: True")

    # --------------------------------------------------------
    # 6. TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("6. XGBOOST 5-FOLD TRAINING")
    print("=" * 78)

    oof = np.zeros(
        len(train_feat),
        dtype=np.float64,
    )

    test_pred = np.zeros(
        len(test_feat),
        dtype=np.float64,
    )

    fold_aucs = []
    best_iterations = []

    for fold in range(N_FOLDS):

        fold_start = time.perf_counter()

        print()
        print("-" * 78)
        print(f"FOLD {fold}")
        print("-" * 78)

        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)

        X_tr = X.iloc[train_idx]
        X_va = X.iloc[valid_idx]

        y_tr = y[train_idx]
        y_va = y[valid_idx]

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            enable_categorical=True,
            verbosity=1,
            **params,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[
                (X_va, y_va),
            ],
            verbose=100,
        )

        valid_pred = model.predict_proba(
            X_va
        )[:, 1]

        fold_test_pred = model.predict_proba(
            X_test
        )[:, 1]

        oof[valid_idx] = valid_pred

        test_pred += (
            fold_test_pred / N_FOLDS
        )

        fold_auc = roc_auc_score(
            y_va,
            valid_pred,
        )

        fold_aucs.append(fold_auc)

        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        best_iterations.append(
            best_iteration
        )

        elapsed = (
            time.perf_counter()
            - fold_start
        )

        print()
        print(
            f"Fold {fold} AUC       : "
            f"{fold_auc:.6f}"
        )
        print(
            f"Best iteration         : "
            f"{best_iteration}"
        )
        print(
            f"Fold time              : "
            f"{elapsed / 60:.2f} min"
        )

    # --------------------------------------------------------
    # 7. EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("7. RESULTS")
    print("=" * 78)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    pooled_auc = roc_auc_score(
        y,
        oof,
    )

    pr_auc = average_precision_score(
        y,
        oof,
    )

    print()
    print(
        f"Mean fold AUC : {mean_auc:.6f}"
    )
    print(
        f"Std fold AUC  : {std_auc:.6f}"
    )
    print(
        f"Pooled OOF AUC: {pooled_auc:.6f}"
    )
    print(
        f"OOF PR-AUC    : {pr_auc:.6f}"
    )

    print()
    print("Fold AUCs:")

    for fold, auc in enumerate(fold_aucs):
        print(
            f"  Fold {fold}: {auc:.6f}"
        )

    print()
    print(
        "Best iterations:",
        best_iterations,
    )

    print()
    print(
        f"Test prediction mean: "
        f"{test_pred.mean():.6f}"
    )

    print(
        f"Test prediction min : "
        f"{test_pred.min():.10f}"
    )

    print(
        f"Test prediction max : "
        f"{test_pred.max():.10f}"
    )

    # --------------------------------------------------------
    # 8. SAVE OUTPUTS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("8. SAVE OUTPUTS")
    print("=" * 78)

    output_dir = (
        Path("/kaggle/working")
        if IS_KAGGLE
        else ROOT / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_path = (
        output_dir / "oof_xgb_v2.npy"
    )

    test_pred_path = (
        output_dir / "test_pred_xgb_v2.npy"
    )

    submission_path = (
        output_dir / "submission_xgb_v2.csv"
    )

    np.save(
        oof_path,
        oof,
    )

    np.save(
        test_pred_path,
        test_pred,
    )

    print("OOF saved : ", oof_path)
    print(
        "Test pred saved: ",
        test_pred_path,
    )

    # --------------------------------------------------------
    # 9. SUBMISSION
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("9. BUILD SUBMISSION")
    print("=" * 78)

    make_submission(
        test_ids=test_feat["id"],
        test_pred=test_pred,
        template_path=SAMPLE_PATH,
        out_path=submission_path,
    )

    submission = pd.read_csv(
        submission_path
    )

    if len(submission) != len(test_raw):
        raise AssertionError(
            "Submission row count mismatch."
        )

    if not submission["id"].equals(
        test_raw["id"].reset_index(
            drop=True
        )
    ):
        raise AssertionError(
            "Submission IDs do not match test IDs."
        )

    predictions = submission[
        target_col
    ].to_numpy()

    if not np.isfinite(predictions).all():
        raise AssertionError(
            "Submission contains NaN/Inf."
        )

    if not (
        (predictions >= 0).all()
        and (predictions <= 1).all()
    ):
        raise AssertionError(
            "Submission predictions are outside [0, 1]."
        )

    print(
        "Submission saved:",
        submission_path,
    )

    # --------------------------------------------------------
    # 10. COMPLETE
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 78)
    print("10. COMPLETE")
    print("=" * 78)

    print(
        f"Total runtime: "
        f"{total_elapsed / 60:.2f} min"
    )

    print(
        f"Pooled OOF AUC: "
        f"{pooled_auc:.6f}"
    )

    print()
    print("Experiment 4B outputs:")
    print(" - oof_xgb_v2.npy")
    print(" - test_pred_xgb_v2.npy")
    print(" - submission_xgb_v2.csv")


if __name__ == "__main__":
    main()
