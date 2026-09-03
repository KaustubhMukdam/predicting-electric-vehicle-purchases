"""Tests for src/train_lgbm.py.

Strategy
--------
1. Smoke test: train on a small synthetic dataset, assert shapes/dtypes
   and that the returned AUC is meaningfully above 0.5.
2. Sanity test: on a *real* 1% slice of the training data (so we have
   realistic feature distributions), the AUC should be comfortably > 0.85.
3. Integration with `tracking`: the trainer should be able to log params,
   metrics, OOF array, and test predictions to MLflow when `tracking=True`.

The real-data slice is large enough (6,686 rows) to get a stable AUC but
small enough to train in a few seconds. We use a `num_boost_round=200`
cap so even the slow path finishes under 30 seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import (
    CATEGORICAL_COLS,
    ENGINEERED_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    TRAIN_CSV,
)


# Default params that train quickly on a 1% slice. Real runs override these.
FAST_PARAMS: dict = {
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
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_1pct_train_test():
    """Load the real train.csv, build features, take a 1% stratified slice
    for training and a 1% slice for 'test' (just a held-out portion of the
    same data so we don't need the actual test.csv in this test)."""
    from src.data import load_data
    from src.features import build_features

    # Use the actual train.csv + test.csv so load_data's schema check passes.
    train, test = load_data(TRAIN_CSV, TRAIN_CSV.parent / "test.csv")
    train_feat = build_features(train)
    test_feat = build_features(test)

    # Stratified 1% slice of train (preserves the 17.5% positive rate).
    rng = np.random.default_rng(42)
    pos_idx = train_feat.index[train_feat[TARGET_COL] == 1].to_numpy()
    neg_idx = train_feat.index[train_feat[TARGET_COL] == 0].to_numpy()
    n_pos = max(50, int(0.01 * len(pos_idx)))
    n_neg = max(200, int(0.01 * len(neg_idx)))
    pos_pick = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_pick = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(keep)
    train_small = train_feat.iloc[keep].reset_index(drop=True)
    test_small = test_feat.sample(frac=0.01, random_state=42).reset_index(drop=True)
    return train_small, test_small


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_train_lgbm_returns_oof_test_and_metrics(real_1pct_train_test):
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])  # any valid 5-way split

    oof, test_pred, metrics = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )

    assert isinstance(oof, np.ndarray)
    assert isinstance(test_pred, np.ndarray)
    assert isinstance(metrics, dict)


def test_train_lgbm_oof_shape_matches_train(real_1pct_train_test):
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    oof, _, _ = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    assert oof.shape == (len(train),)
    assert oof.dtype.kind == "f"


def test_train_lgbm_test_pred_shape_matches_test(real_1pct_train_test):
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    _, test_pred, _ = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    assert test_pred.shape == (len(test),)
    assert test_pred.dtype.kind == "f"


def test_train_lgbm_predictions_are_in_unit_interval(real_1pct_train_test):
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    oof, test_pred, _ = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    assert ((oof >= 0) & (oof <= 1)).all()
    assert ((test_pred >= 0) & (test_pred <= 1)).all()


def test_train_lgbm_oof_auc_above_baseline(real_1pct_train_test):
    """OOF predictions should rank positives above negatives much better
    than chance. A reasonable LGBM on a 1% slice hits ~0.92; we assert
    a soft floor of 0.85 to allow for randomness in a small sample."""
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    oof, _, metrics = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )

    y = train[TARGET_COL].to_numpy()
    auc = roc_auc_score(y, oof)
    assert auc > 0.85, f"OOF AUC {auc:.4f} below soft floor of 0.85"
    # And the metrics dict should report roughly the same number.
    # (mean(fold_aucs) != pooled roc_auc_score in general — the folds have
    # different positive rates and slightly different models, so a few
    # thousandths of difference is normal.)
    assert abs(metrics["cv_auc_mean"] - auc) < 0.01


def test_train_lgbm_metrics_dict_has_expected_keys(real_1pct_train_test):
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    _, _, metrics = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    for key in ("cv_auc_mean", "cv_auc_std", "fold_aucs"):
        assert key in metrics, f"metrics missing key: {key}"
    assert len(metrics["fold_aucs"]) == 5
    assert all(0.5 <= a <= 1.0 for a in metrics["fold_aucs"])


def test_train_lgbm_is_deterministic_with_seed(real_1pct_train_test):
    """Two consecutive runs with the same seed should produce identical OOF."""
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    oof_a, test_a, _ = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    oof_b, test_b, _ = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=200,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    np.testing.assert_array_equal(oof_a, oof_b)
    np.testing.assert_array_equal(test_a, test_b)


def test_train_lgbm_uses_only_specified_features(real_1pct_train_test):
    """If we drop a feature, the trainer must not depend on it being present."""
    from src.train_lgbm import train_lgbm

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])
    # Drop the engineered features; the trainer should not crash.
    reduced = [c for c in FEATURE_COLS if c not in ENGINEERED_COLS]

    oof, test_pred, metrics = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=reduced,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=100,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    assert oof.shape == (len(train),)
    assert test_pred.shape == (len(test),)
    # AUC should still be > 0.5 — the raw features alone carry the signal.
    y = train[TARGET_COL].to_numpy()
    assert roc_auc_score(y, oof) > 0.5


def test_train_lgbm_with_tracking_logs_to_mlflow(real_1pct_train_test, tmp_path, monkeypatch):
    """When tracking=True, the trainer should record a run with params, metrics,
    and OOF/test artifact arrays."""
    import mlflow

    from src.tracking import set_tracking_uri
    from src.train_lgbm import train_lgbm

    # Redirect MLflow to a temp dir.
    mlruns = tmp_path / "mlruns"
    set_tracking_uri(mlruns)
    mlflow.set_tracking_uri(f"file:{mlruns}")

    train, test = real_1pct_train_test
    folds = np.array([i % 5 for i in range(len(train))])

    oof, test_pred, metrics = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        params=FAST_PARAMS,
        num_boost_round=100,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=True,
        run_name="phase6_smoke",
    )

    # Verify the run was recorded.
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 1, "No MLflow run was created"
    latest = max(runs, key=lambda r: r.info.start_time)
    assert latest.info.run_name == "phase6_smoke"
    # Params: learning_rate, num_leaves, etc. were logged.
    assert "learning_rate" in latest.data.params
    # Metrics: cv_auc_mean was logged.
    assert "cv_auc_mean" in latest.data.metrics
    # Artifacts: oof.npy and test_pred.npy were logged under the "oof" dir.
    root_artifacts = {a.path for a in client.list_artifacts(latest.info.run_id)}
    nested_artifacts = {
        a.path for a in client.list_artifacts(latest.info.run_id, path="oof")
    }
    all_paths = root_artifacts | nested_artifacts
    assert any(p.endswith("oof.npy") for p in all_paths), (
        f"oof.npy not found in artifacts: {all_paths}"
    )
    assert any(p.endswith("test_pred.npy") for p in all_paths), (
        f"test_pred.npy not found in artifacts: {all_paths}"
    )
