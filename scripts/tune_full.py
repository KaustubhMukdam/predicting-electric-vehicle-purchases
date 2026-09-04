"""Optuna sweep on the FULL training data.

Mirrors Phase 11's `train_full.py` but uses the Optuna-tuned
hyperparameters instead of the v1 defaults. The best params are
persisted to `submissions/best_params_lgbm_v2.json` and the resulting
submission to `submissions/submission_lgbm_v2.csv`.

Runtime: ~30-60 min for 30 trials on local CPU (each trial = 5-fold
CV with up to 1500 boost rounds + early stopping).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.train_lgbm import train_lgbm  # noqa: E402
from src.tune_lgbm import run_optuna_sweep  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Phase 12: Optuna sweep on full data")
    print("=" * 60)

    print("\n[1/5] Loading data + features...")
    train, test = load_data(TRAIN_CSV, TEST_CSV)
    train_feat = build_features(train)
    test_feat = build_features(test)
    print(f"      train shape: {train_feat.shape}")
    print(f"      test  shape: {test_feat.shape}")

    print("\n[2/5] Building CV folds...")
    folds = make_folds(
        train_feat[TARGET_COL].to_numpy(), n_splits=N_FOLDS, seed=RANDOM_SEED
    )

    print("\n[3/5] Running Optuna sweep (30 trials, full data)...")
    print("      (each trial = 5-fold CV; expect ~30-60 min total)")
    best_params, best_score, study = run_optuna_sweep(
        train=train_feat,
        test=test_feat,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        n_trials=30,
        num_boost_round=1500,
        early_stopping_rounds=100,
        tracking_enabled=True,
        sweep_run_name="phase12_full_optuna",
        seed=RANDOM_SEED,
    )
    print(f"      best CV AUC: {best_score:.5f}")
    print(f"      best params: {best_params}")

    print("\n[4/5] Retraining with best params (final OOF + test preds)...")
    oof, test_pred, metrics = train_lgbm(
        train=train_feat,
        test=test_feat,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=best_params,
        num_boost_round=1500,
        early_stopping_rounds=100,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=True,
        run_name="phase12_full_optuna_retrain",
    )
    from sklearn.metrics import roc_auc_score
    pooled = roc_auc_score(train_feat[TARGET_COL].to_numpy(), oof)
    print(f"      final mean CV AUC: {metrics['cv_auc_mean']:.5f} "
          f"(std {metrics['cv_auc_std']:.5f})")
    print(f"      final pooled OOF AUC: {pooled:.5f}")
    print(f"      per-fold AUC: {[round(a, 5) for a in metrics['fold_aucs']]}")

    print("\n[5/5] Writing submission...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / "submission_lgbm_v2.csv"
    make_submission(
        test_ids=test_feat["id"],
        test_pred=test_pred,
        template_path=SAMPLE_SUBMISSION_CSV,
        out_path=submission_path,
    )
    print(f"      wrote: {submission_path}")

    # Persist best params and metrics.
    params_path = SUBMISSIONS_DIR / "best_params_lgbm_v2.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    metrics_path = SUBMISSIONS_DIR / "run_metrics_v2.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_score_study": best_score,
                "final_cv_auc_mean": metrics["cv_auc_mean"],
                "final_cv_auc_std": metrics["cv_auc_std"],
                "final_pooled_oof_auc": pooled,
                "fold_aucs": metrics["fold_aucs"],
                "elapsed_seconds": time.time() - t0,
                "n_trials": 30,
            },
            indent=2,
        )
    )
    print(f"      params: {params_path}")
    print(f"      metrics: {metrics_path}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
