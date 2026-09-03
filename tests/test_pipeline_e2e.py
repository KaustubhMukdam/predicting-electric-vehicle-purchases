"""End-to-end pipeline test.

Wires every module together:

    load_data
      -> build_features
      -> make_folds
      -> train_lgbm
      -> make_submission

Runs on a 1% stratified slice of the real training data so the test
finishes in a few minutes. The 1% slice is large enough that an LGBM
hits OOF AUC > 0.85 (well above the constant baseline of 0.5).

This is the *integration* test: if this passes, the pipeline is wired
correctly end-to-end. Per-module unit tests already exist for each
piece.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)


# Default LightGBM hyperparameters for the E2E smoke test.
# Conservative to keep the run under a few minutes on a 1% slice.
E2E_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
}


pytestmark = pytest.mark.skipif(
    not TRAIN_CSV.exists() or not TEST_CSV.exists() or not SAMPLE_SUBMISSION_CSV.exists(),
    reason="data CSVs not present",
)


@pytest.fixture(scope="module")
def e2e_artifacts(tmp_path_factory):
    """Run the full pipeline on a 1% train slice but the full test set,
    mirroring what a real Kaggle submission looks like.

    Module-scoped so the (slow) training happens once for all tests in
    this file. Subsequent tests just re-read the produced files.
    """
    from src.cv import make_folds
    from src.data import load_data
    from src.features import build_features
    from src.predict import make_submission
    from src.train_lgbm import train_lgbm

    out_dir = tmp_path_factory.mktemp("e2e")
    submission_path = out_dir / "submission.csv"

    # 1. Load
    train, test = load_data(TRAIN_CSV, TEST_CSV)

    # 2. Take a 1% stratified slice of train to keep the test fast.
    rng = np.random.default_rng(RANDOM_SEED)
    pos_idx = train.index[train[TARGET_COL] == 1].to_numpy()
    neg_idx = train.index[train[TARGET_COL] == 0].to_numpy()
    n_pos = max(50, int(0.01 * len(pos_idx)))
    n_neg = max(200, int(0.01 * len(neg_idx)))
    pos_pick = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_pick = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(keep)
    train_small = train.iloc[keep].reset_index(drop=True)

    # 3. Use the FULL test set — production-shape submission.
    test_full = test.copy()

    # 4. Feature engineering
    train_feat = build_features(train_small)
    test_feat = build_features(test_full)

    # 5. CV folds
    folds = make_folds(train_feat[TARGET_COL].to_numpy(), n_splits=N_FOLDS, seed=RANDOM_SEED)

    # 6. Train (tracking off — this test should not pollute the user's MLflow dir)
    oof, test_pred, metrics = train_lgbm(
        train=train_feat,
        test=test_feat,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=E2E_PARAMS,
        num_boost_round=200,
        early_stopping_rounds=50,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )

    # 7. Submission — must use the full real template + full test ids
    #    since we predicted on the full test set.
    make_submission(
        test_ids=test_feat["id"],
        test_pred=test_pred,
        template_path=SAMPLE_SUBMISSION_CSV,
        out_path=submission_path,
    )

    return {
        "out_dir": out_dir,
        "submission_path": submission_path,
        "oof": oof,
        "test_pred": test_pred,
        "metrics": metrics,
        "y_train": train_feat[TARGET_COL].to_numpy(),
        "test_ids": test_feat["id"],
    }


# ---------------------------------------------------------------------------
# E2E assertions
# ---------------------------------------------------------------------------
def test_e2e_pipeline_produces_a_submission_file(e2e_artifacts):
    assert e2e_artifacts["submission_path"].exists()


def test_e2e_pipeline_submission_has_correct_shape(e2e_artifacts):
    sub = pd.read_csv(e2e_artifacts["submission_path"])
    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    # The submission must have the full template's row count, even though
    # we only used 1% of train — the test set is the full 286k.
    assert sub.shape == template.shape
    assert list(sub.columns) == list(template.columns)


def test_e2e_pipeline_submission_ids_match_test_ids(e2e_artifacts):
    """The submission's id column must be exactly the test ids passed in."""
    sub = pd.read_csv(e2e_artifacts["submission_path"])
    np.testing.assert_array_equal(
        sub["id"].to_numpy(), e2e_artifacts["test_ids"].to_numpy()
    )


def test_e2e_pipeline_oof_auc_above_baseline(e2e_artifacts):
    """The end-to-end model on 1% data must beat the 0.5 constant baseline
    by a wide margin. We assert > 0.85 (a soft floor that allows for
    randomness in a small sample but rejects a broken pipeline)."""
    auc = roc_auc_score(e2e_artifacts["y_train"], e2e_artifacts["oof"])
    assert auc > 0.85, f"E2E OOF AUC {auc:.4f} is below the 0.85 floor"


def test_e2e_pipeline_metrics_have_expected_keys(e2e_artifacts):
    metrics = e2e_artifacts["metrics"]
    for key in ("cv_auc_mean", "cv_auc_std", "fold_aucs"):
        assert key in metrics
    assert len(metrics["fold_aucs"]) == N_FOLDS
    for a in metrics["fold_aucs"]:
        assert 0.5 <= a <= 1.0


def test_e2e_pipeline_test_predictions_in_unit_interval(e2e_artifacts):
    test_pred = e2e_artifacts["test_pred"]
    assert ((test_pred >= 0) & (test_pred <= 1)).all()
    assert not np.isnan(test_pred).any()
    assert not np.isinf(test_pred).any()


def test_e2e_pipeline_oof_predictions_in_unit_interval(e2e_artifacts):
    oof = e2e_artifacts["oof"]
    assert ((oof >= 0) & (oof <= 1)).all()
    assert not np.isnan(oof).any()


def test_e2e_pipeline_submission_predictions_match_test_pred(e2e_artifacts):
    """The submission's Will_Buy_EV column must equal the trainer's
    test_pred (up to CSV rounding) for every row."""
    sub = pd.read_csv(e2e_artifacts["submission_path"])
    np.testing.assert_array_almost_equal(
        sub["Will_Buy_EV"].to_numpy(), e2e_artifacts["test_pred"]
    )
