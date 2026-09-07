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
# Experiment 4 — XGBoost baseline
# Same V2 features + same 5 stratified folds as LightGBM V2.
# ============================================================

SEED = 42
N_FOLDS = 5
N_ESTIMATORS = 5000
EARLY_STOPPING = 200

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "max_bin": 256,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
}


def find_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "src").is_dir():
            return p
    raise FileNotFoundError("Could not find project root containing src/")


IS_KAGGLE = Path("/kaggle").exists()
ROOT = Path("/kaggle/working") if IS_KAGGLE else find_root()
sys.path.insert(0, str(ROOT))

# On Kaggle the source dataset is mounted separately.
if IS_KAGGLE:
    candidates = [
        Path("/kaggle/input/ev-purchase-src/src"),
        Path("/kaggle/input/ev-purchase-src-final/src"),
        Path("/kaggle/input/datasets/kaustubhmukdam/ev-purchase-src-final/src"),
    ]
    src = next((p for p in candidates if p.is_dir()), None)
    if src is None:
        matches = list(Path("/kaggle/input").glob("*/src"))
        src = matches[0] if matches else None
    if src is None:
        raise FileNotFoundError("Could not find src/ under /kaggle/input")
    sys.path.insert(0, str(src.parent))

from src.data import load_data
from src.features_v2 import build_features
from src.cv import make_folds
from src.predict import make_submission


def resolve_paths():
    if not IS_KAGGLE:
        d = ROOT / "data"
        return d/"train.csv", d/"test.csv", d/"sample_submission.csv"

    # Search attached Kaggle datasets instead of assuming a dataset slug.
    for p in Path("/kaggle/input").rglob("train.csv"):
        d = p.parent
        if (d/"test.csv").exists() and (d/"sample_submission.csv").exists():
            return p, d/"test.csv", d/"sample_submission.csv"
    raise FileNotFoundError("Could not find train.csv/test.csv/sample_submission.csv")


def encode_target(s):
    if pd.api.types.is_numeric_dtype(s):
        a = s.to_numpy()
        if not set(np.unique(a)).issubset({0, 1}):
            raise ValueError("Target must contain only 0/1")
        return a.astype(np.int8)

    m = {
        "yes": 1, "no": 0, "1": 1, "0": 0,
        "true": 1, "false": 0,
    }
    a = s.astype(str).str.strip().str.lower().map(m)
    if a.isna().any():
        raise ValueError(f"Unexpected target values: {s[a.isna()].unique()}")
    return a.to_numpy(np.int8)


def make_xgb_matrix(train_feat, test_feat, target_col):
    drop = {"id", target_col}
    tr = train_feat.drop(columns=[c for c in drop if c in train_feat]).copy()
    te = test_feat.drop(columns=[c for c in drop if c in test_feat]).copy()

    if list(tr.columns) != list(te.columns):
        raise ValueError("Train/test V2 feature columns do not match.")

    cats = [
        c for c in tr.columns
        if pd.api.types.is_object_dtype(tr[c])
        or isinstance(tr[c].dtype, pd.CategoricalDtype)
    ]
    nums = [c for c in tr.columns if c not in cats]

    tr_num = sparse.csr_matrix(tr[nums].astype(np.float32).to_numpy())
    te_num = sparse.csr_matrix(te[nums].astype(np.float32).to_numpy())

    if cats:
        combined = pd.concat(
            [tr[cats].astype(str), te[cats].astype(str)],
            ignore_index=True,
        )
        combined = pd.get_dummies(
            combined, columns=cats, dtype=np.float32, sparse=True
        )
        n = len(tr)
        tr_cat = combined.iloc[:n].sparse.to_coo().tocsr()
        te_cat = combined.iloc[n:].sparse.to_coo().tocsr()
        cat_names = list(combined.columns)
    else:
        tr_cat = sparse.csr_matrix((len(tr), 0), dtype=np.float32)
        te_cat = sparse.csr_matrix((len(te), 0), dtype=np.float32)
        cat_names = []

    Xtr = sparse.hstack([tr_num, tr_cat], format="csr", dtype=np.float32)
    Xte = sparse.hstack([te_num, te_cat], format="csr", dtype=np.float32)

    return Xtr, Xte, nums + cat_names, cats


def main():
    start = time.perf_counter()
    train_path, test_path, sample_path = resolve_paths()

    print("=" * 72)
    print("EV PURCHASE — XGBOOST EXPERIMENT 4")
    print("=" * 72)
    print("Environment:", "Kaggle" if IS_KAGGLE else "Local")
    print("XGBoost:", xgb.__version__)
    print("Train:", train_path)

    # --------------------------------------------------------
    # Load + V2 features
    # --------------------------------------------------------
    train_raw, test_raw, _ = load_data(
        train_path=train_path,
        test_path=test_path,
        sample_path=sample_path,
    )

    target = "Will_Buy_EV"
    y = encode_target(train_raw[target])
    train_raw = train_raw.copy()
    train_raw[target] = y

    print(f"Train shape: {train_raw.shape}")
    print(f"Test shape : {test_raw.shape}")
    print(f"Positive rate: {y.mean():.6f}")

    print("\nBuilding V2 features...")
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    feature_cols = [
        c for c in train_feat.columns if c not in {"id", target}
    ]
    missing = [c for c in feature_cols if c not in test_feat.columns]
    if missing:
        raise ValueError(f"Test missing V2 features: {missing}")

    X, X_test, names, cats = make_xgb_matrix(
        train_feat, test_feat, target
    )

    print("V2 dataframe features:", len(feature_cols))
    print("XGBoost matrix shape :", X.shape)
    print("One-hot categorical source columns:", len(cats))

    # --------------------------------------------------------
    # Same folds as LightGBM V2
    # --------------------------------------------------------
    folds = make_folds(y, n_splits=N_FOLDS, seed=SEED)

    # GPU on Kaggle, CPU locally.
    # XGBoost >= 2.x uses device="cuda" with tree_method="hist".
    device = "cuda" if IS_KAGGLE else "cpu"
    params = PARAMS.copy()
    params["device"] = device

    print("\nDevice:", device.upper())
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("n_estimators:", N_ESTIMATORS)
    print("early stopping:", EARLY_STOPPING)

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test_raw), dtype=np.float64)
    fold_aucs = []
    best_iters = []

    # --------------------------------------------------------
    # 5-fold CV
    # --------------------------------------------------------
    for fold in range(N_FOLDS):
        print("\n" + "-" * 72)
        print(f"FOLD {fold}")
        print("-" * 72)

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
        best_iters.append(getattr(model, "best_iteration", None))

        print(f"Fold AUC: {auc:.6f}")
        print(f"Best iteration: {getattr(model, 'best_iteration', None)}")
        print(f"Fold time: {(time.perf_counter()-t0)/60:.2f} min")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    pooled_auc = roc_auc_score(y, oof)
    pr_auc = average_precision_score(y, oof)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"Mean fold AUC : {np.mean(fold_aucs):.6f}")
    print(f"Std fold AUC  : {np.std(fold_aucs):.6f}")
    print(f"Pooled OOF AUC: {pooled_auc:.6f}")
    print(f"OOF PR-AUC    : {pr_auc:.6f}")
    print("Fold AUCs     :", [round(v, 6) for v in fold_aucs])
    print("Best iters    :", best_iters)
    print(f"Test pred mean: {test_pred.mean():.6f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out = Path("/kaggle/working") if IS_KAGGLE else ROOT/"outputs"
    out.mkdir(parents=True, exist_ok=True)

    np.save(out/"oof_xgb_v1.npy", oof)
    np.save(out/"test_pred_xgb_v1.npy", test_pred)

    make_submission(
        test_ids=test_feat["id"],
        test_pred=test_pred,
        template_path=sample_path,
        out_path=out/"submission_xgb_v1.csv",
    )

    sub = pd.read_csv(out/"submission_xgb_v1.csv")
    assert len(sub) == len(test_raw)
    assert sub["id"].equals(test_raw["id"].reset_index(drop=True))
    assert np.isfinite(sub[target]).all()
    assert ((sub[target] >= 0) & (sub[target] <= 1)).all()

    print("\nSaved:")
    print(out/"oof_xgb_v1.npy")
    print(out/"test_pred_xgb_v1.npy")
    print(out/"submission_xgb_v1.csv")
    print(f"Total runtime: {(time.perf_counter()-start)/60:.2f} min")


if __name__ == "__main__":
    main()
