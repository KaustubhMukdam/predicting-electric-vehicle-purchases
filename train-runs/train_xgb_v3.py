from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

# ============================================================
# Experiment 4C — controlled XGBoost ablation
#
# XGB V1:
#   one-hot categoricals, depth=8, min_child_weight=10
#   lr=0.05, lambda=1
#
# XGB V3:
#   SAME representation and tree geometry
#   lr=0.03, lambda=2
#
# Run on Kaggle GPU.
# ============================================================

SEED = 42
N_FOLDS = 5
N_ESTIMATORS = 8000
EARLY_STOPPING = 250

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "max_bin": 256,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
}

IS_KAGGLE = Path("/kaggle").exists()


def find_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "src").is_dir():
            return p
    raise FileNotFoundError("Could not find project root containing src/")


ROOT = Path("/kaggle/working") if IS_KAGGLE else find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_kaggle_src():
    if not IS_KAGGLE:
        return

    candidates = [
        Path("/kaggle/input/ev-purchase-src/src"),
        Path("/kaggle/input/ev-purchase-src-final/src"),
        Path("/kaggle/input/datasets/kaustubhmukdam/ev-purchase-src-final/src"),
    ]

    src_dir = next((p for p in candidates if p.is_dir()), None)

    if src_dir is None:
        matches = list(Path("/kaggle/input").glob("*/src"))
        src_dir = matches[0] if matches else None

    if src_dir is None:
        raise FileNotFoundError(
            "Could not find src/ under /kaggle/input."
        )

    if str(src_dir.parent) not in sys.path:
        sys.path.insert(0, str(src_dir.parent))


configure_kaggle_src()

from src.data import load_data
from src.features_v2 import build_features
from src.cv import make_folds
from src.predict import make_submission


def resolve_paths():
    if not IS_KAGGLE:
        d = ROOT / "data"
        return d/"train.csv", d/"test.csv", d/"sample_submission.csv"

    for train_path in Path("/kaggle/input").rglob("train.csv"):
        d = train_path.parent
        test_path = d / "test.csv"
        sample_path = d / "sample_submission.csv"
        if test_path.exists() and sample_path.exists():
            return train_path, test_path, sample_path

    raise FileNotFoundError(
        "Could not find train.csv/test.csv/sample_submission.csv"
    )


def encode_target(y):
    if pd.api.types.is_numeric_dtype(y):
        a = y.to_numpy()
        if not set(np.unique(a)).issubset({0, 1}):
            raise ValueError("Target must contain only 0/1")
        return a.astype(np.int8)

    mapping = {
        "yes": 1, "no": 0, "1": 1, "0": 0,
        "true": 1, "false": 0,
    }
    a = y.astype(str).str.strip().str.lower().map(mapping)
    if a.isna().any():
        raise ValueError(f"Unexpected target values: {y[a.isna()].unique()}")
    return a.to_numpy(np.int8)


def prepare_one_hot(train_feat, test_feat, target_col):
    drop = {"id", target_col}

    tr = train_feat.drop(
        columns=[c for c in drop if c in train_feat]
    ).copy()

    te = test_feat.drop(
        columns=[c for c in drop if c in test_feat]
    ).copy()

    if list(tr.columns) != list(te.columns):
        raise ValueError("Train/test V2 feature columns do not match.")

    cats = [
        c for c in tr.columns
        if (
            pd.api.types.is_object_dtype(tr[c])
            or isinstance(tr[c].dtype, pd.CategoricalDtype)
        )
    ]
    nums = [c for c in tr.columns if c not in cats]

    tr_num = sparse.csr_matrix(
        tr[nums].astype(np.float32).to_numpy()
    )
    te_num = sparse.csr_matrix(
        te[nums].astype(np.float32).to_numpy()
    )

    if cats:
        combined = pd.concat(
            [tr[cats].astype(str), te[cats].astype(str)],
            ignore_index=True,
        )

        combined = pd.get_dummies(
            combined,
            columns=cats,
            dtype=np.float32,
            sparse=True,
        )

        n = len(tr)

        tr_cat = combined.iloc[:n].sparse.to_coo().tocsr()
        te_cat = combined.iloc[n:].sparse.to_coo().tocsr()

        cat_names = list(combined.columns)
    else:
        tr_cat = sparse.csr_matrix((len(tr), 0), dtype=np.float32)
        te_cat = sparse.csr_matrix((len(te), 0), dtype=np.float32)
        cat_names = []

    X = sparse.hstack(
        [tr_num, tr_cat],
        format="csr",
        dtype=np.float32,
    )

    X_test = sparse.hstack(
        [te_num, te_cat],
        format="csr",
        dtype=np.float32,
    )

    return X, X_test, nums + cat_names, cats


def main():
    total_start = time.perf_counter()

    train_path, test_path, sample_path = resolve_paths()

    print("=" * 78)
    print("EV PURCHASE — XGBOOST EXPERIMENT 4C")
    print("=" * 78)
    print("Environment:", "Kaggle" if IS_KAGGLE else "Local")
    print("XGBoost:", xgb.__version__)
    print("Device:", "CUDA" if IS_KAGGLE else "CPU")
    print("Train:", train_path)
    print("Test :", test_path)

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. LOAD DATA")
    print("=" * 78)

    train_raw, test_raw = load_data(
        train_path=train_path,
        test_path=test_path,
    )

    target = "Will_Buy_EV"
    y = encode_target(train_raw[target])

    print("Train shape:", train_raw.shape)
    print("Test shape :", test_raw.shape)
    print(f"Positive rate: {y.mean():.6f}")

    # --------------------------------------------------------
    # 2. V2 FEATURES
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. BUILD V2 FEATURES")
    print("=" * 78)

    train_feat = build_features(train_raw.copy())
    test_feat = build_features(test_raw.copy())

    feature_cols = [
        c for c in train_feat.columns
        if c not in {"id", target}
    ]

    missing = [
        c for c in feature_cols
        if c not in test_feat.columns
    ]

    if missing:
        raise ValueError(f"Test missing V2 features: {missing}")

    print("V2 feature count:", len(feature_cols))

    # --------------------------------------------------------
    # 3. SAME ONE-HOT REPRESENTATION AS V1
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. ONE-HOT ENCODING — SAME AS XGB V1")
    print("=" * 78)

    X, X_test, feature_names, categorical_cols = prepare_one_hot(
        train_feat,
        test_feat,
        target,
    )

    print("Categorical source columns:", len(categorical_cols))
    print("Final encoded columns:", X.shape[1])
    print("Train matrix:", X.shape)
    print("Test matrix :", X_test.shape)

    # --------------------------------------------------------
    # 4. SAME FOLDS
    # --------------------------------------------------------
    print("\n" + "=" * 78)
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
            f"Fold {fold}: {mask.sum():,} rows | "
            f"positive rate={y[mask].mean():.6f}"
        )

    # --------------------------------------------------------
    # 5. PARAMETERS
    # --------------------------------------------------------
    params = PARAMS.copy()
    params["device"] = "cuda" if IS_KAGGLE else "cpu"

    print("\n" + "=" * 78)
    print("5. XGBOOST 4C PARAMETERS")
    print("=" * 78)

    for k, v in params.items():
        print(f"{k:20s}: {v}")

    print(f"{'n_estimators':20s}: {N_ESTIMATORS}")
    print(f"{'early_stopping':20s}: {EARLY_STOPPING}")

    # --------------------------------------------------------
    # 6. CV
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. XGBOOST 5-FOLD TRAINING")
    print("=" * 78)

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test_raw), dtype=np.float64)

    fold_aucs = []
    best_iters = []

    for fold in range(N_FOLDS):
        print("\n" + "-" * 78)
        print(f"FOLD {fold}")
        print("-" * 78)

        t0 = time.perf_counter()

        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            verbosity=1,
            **params,
        )

        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            verbose=100,
        )

        vp = model.predict_proba(X[va_idx])[:, 1]
        tp = model.predict_proba(X_test)[:, 1]

        oof[va_idx] = vp
        test_pred += tp / N_FOLDS

        auc = roc_auc_score(y[va_idx], vp)
        fold_aucs.append(auc)

        best = getattr(model, "best_iteration", None)
        best_iters.append(best)

        print(f"Fold {fold} AUC: {auc:.6f}")
        print(f"Best iteration: {best}")
        print(f"Fold time: {(time.perf_counter()-t0)/60:.2f} min")

    # --------------------------------------------------------
    # 7. RESULTS
    # --------------------------------------------------------
    pooled_auc = roc_auc_score(y, oof)
    pr_auc = average_precision_score(y, oof)

    print("\n" + "=" * 78)
    print("7. RESULTS")
    print("=" * 78)

    print(f"Mean fold AUC : {np.mean(fold_aucs):.6f}")
    print(f"Std fold AUC  : {np.std(fold_aucs):.6f}")
    print(f"Pooled OOF AUC: {pooled_auc:.6f}")
    print(f"OOF PR-AUC    : {pr_auc:.6f}")
    print("Fold AUCs     :", [round(v, 6) for v in fold_aucs])
    print("Best iters    :", best_iters)

    print(f"Test pred mean: {test_pred.mean():.6f}")
    print(f"Test pred min : {test_pred.min():.10f}")
    print(f"Test pred max : {test_pred.max():.10f}")

    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------
    out = (
        Path("/kaggle/working")
        if IS_KAGGLE
        else ROOT / "outputs"
    )
    out.mkdir(parents=True, exist_ok=True)

    oof_path = out / "oof_xgb_v3.npy"
    test_path_out = out / "test_pred_xgb_v3.npy"
    sub_path = out / "submission_xgb_v3.csv"

    np.save(oof_path, oof)
    np.save(test_path_out, test_pred)

    make_submission(
        test_ids=test_feat["id"],
        test_pred=test_pred,
        template_path=sample_path,
        out_path=sub_path,
    )

    sub = pd.read_csv(sub_path)

    assert len(sub) == len(test_raw)
    assert sub["id"].equals(
        test_raw["id"].reset_index(drop=True)
    )
    assert np.isfinite(sub[target]).all()
    assert ((sub[target] >= 0) & (sub[target] <= 1)).all()

    print("\nSaved:")
    print(oof_path)
    print(test_path_out)
    print(sub_path)

    print(
        f"\nTotal runtime: "
        f"{(time.perf_counter()-total_start)/60:.2f} min"
    )


if __name__ == "__main__":
    main()
