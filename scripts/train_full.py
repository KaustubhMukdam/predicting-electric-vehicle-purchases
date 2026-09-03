"""Full-data training run.

This is the production script (the equivalent of running `train.ipynb`
on Kaggle, but locally). It:

  1. Loads data/train.csv and data/test.csv
  2. Builds features
  3. Generates 5-fold stratified CV splits
  4. Trains LightGBM with the default params
  5. Logs everything to MLflow at mlruns/
  6. Writes submissions/submission_lgbm_v1.csv

Run from the project root:
    python -m scripts.train_full

Or with venv:
    venv/bin/python -m scripts.train_full
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CATEGORICAL_COLS,
    FEATURE_COLS,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)
from src.cv import make_folds  # noqa: E402
from src.data import load_data  # noqa: E402
from src.features import build_features  # noqa: E402
from src.predict import make_submission  # noqa: E402
from src.train_lgbm import DEFAULT_LGBM_PARAMS, train_lgbm  # noqa: E402
from src import tracking  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Phase 11: full-data training run")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    train, test = load_data(TRAIN_CSV, TEST_CSV)
    print(f"      train shape: {train.shape}")
    print(f"      test  shape: {test.shape}")
    print(f"      positive rate: {train[TARGET_COL].mean():.4f}")

    print("\n[2/5] Building features...")
    train_feat = build_features(train)
    test_feat = build_features(test)
    print(f"      feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    print("\n[3/5] Building CV folds...")
    folds = make_folds(
        train_feat[TARGET_COL].to_numpy(), n_splits=N_FOLDS, seed=RANDOM_SEED
    )
    sizes = np.bincount(folds).tolist()
    print(f"      fold sizes: {sizes}")
    pos_rates = [
        float(train_feat[TARGET_COL].to_numpy()[folds == k].mean())
        for k in range(N_FOLDS)
    ]
    print(f"      per-fold positive rates: {[round(p, 4) for p in pos_rates]}")

    print("\n[4/5] Training LightGBM with 5-fold CV...")
    print(f"      params: {DEFAULT_LGBM_PARAMS}")
    print(f"      num_boost_round=1500, early_stopping_rounds=100")
    oof, test_pred, metrics = train_lgbm(
        train=train_feat,
        test=test_feat,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=DEFAULT_LGBM_PARAMS,
        num_boost_round=1500,
        early_stopping_rounds=100,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=True,
        run_name="phase11_full_data_v1",
    )
    print(f"      mean CV AUC: {metrics['cv_auc_mean']:.5f}")
    print(f"      std  CV AUC: {metrics['cv_auc_std']:.5f}")
    print(f"      per-fold AUC: {[round(a, 5) for a in metrics['fold_aucs']]}")

    # Also compute pooled OOF AUC for reporting.
    from sklearn.metrics import roc_auc_score
    pooled_auc = roc_auc_score(train_feat[TARGET_COL].to_numpy(), oof)
    print(f"      pooled OOF AUC: {pooled_auc:.5f}")

    print("\n[5/5] Building submission...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / "submission_lgbm_v1.csv"
    make_submission(
        test_ids=test_feat["id"],
        test_pred=test_pred,
        template_path=SAMPLE_SUBMISSION_CSV,
        out_path=submission_path,
    )
    sub = pd.read_csv(submission_path)
    print(f"      wrote: {submission_path}")
    print(f"      shape: {sub.shape}")
    print(
        f"      Will_Buy_EV range: [{sub['Will_Buy_EV'].min():.4f}, "
        f"{sub['Will_Buy_EV'].max():.4f}]"
    )
    print(f"      Will_Buy_EV mean:  {sub['Will_Buy_EV'].mean():.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Submission file: {submission_path}")

    # Persist the metrics for the doc-update step.
    import json
    metrics_path = SUBMISSIONS_DIR / "run_metrics_v1.json"
    metrics_path.write_text(
        json.dumps(
            {
                "cv_auc_mean": metrics["cv_auc_mean"],
                "cv_auc_std": metrics["cv_auc_std"],
                "fold_aucs": metrics["fold_aucs"],
                "pooled_oof_auc": pooled_auc,
                "elapsed_seconds": elapsed,
                "num_boost_round": 1500,
                "early_stopping_rounds": 100,
                "params": DEFAULT_LGBM_PARAMS,
            },
            indent=2,
        )
    )
    print(f"Metrics written: {metrics_path}")


if __name__ == "__main__":
    main()
