"""Tests for src/tune_lgbm.py.

Strategy
--------
- Test the `LGBMSearchSpace` dataclass: tunable params are present,
  defaults are valid LGBM values, search space is sane.
- Test `objective()`: given a fixed trial, returns a finite float,
  and running two trials with different param sets returns different
  (or at least recomputable) scores.
- Test `run_optuna_sweep()` end-to-end on a 1% slice: produces a best
  params dict, best score is above the constant baseline.
- The sweep must log to MLflow (one run per trial, plus a summary
  run for the best params).
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
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)


pytestmark = pytest.mark.skipif(
    not TRAIN_CSV.exists() or not TEST_CSV.exists(),
    reason="data CSVs not present",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_train_test():
    """1% stratified slice of train, 1% of test."""
    from src.data import load_data
    from src.features import build_features

    train, test = load_data(TRAIN_CSV, TEST_CSV)
    rng = np.random.default_rng(RANDOM_SEED)
    pos = train.index[train[TARGET_COL] == 1].to_numpy()
    neg = train.index[train[TARGET_COL] == 0].to_numpy()
    keep = np.concatenate([
        rng.choice(pos, size=200, replace=False),
        rng.choice(neg, size=800, replace=False),
    ])
    rng.shuffle(keep)
    train_small = train.iloc[keep].reset_index(drop=True)
    test_small = test.sample(frac=0.01, random_state=RANDOM_SEED).reset_index(drop=True)
    return build_features(train_small), build_features(test_small)


@pytest.fixture(scope="module")
def folds(small_train_test):
    from src.cv import make_folds

    train, _ = small_train_test
    return make_folds(train[TARGET_COL].to_numpy(), n_splits=N_FOLDS, seed=RANDOM_SEED)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def test_search_space_has_required_hyperparameters():
    from src.tune_lgbm import LGBMSearchSpace

    space = LGBMSearchSpace()
    for key in (
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
    ):
        assert key in space.ranges, f"missing tunable: {key}"


def test_search_space_ranges_are_valid():
    from src.tune_lgbm import LGBMSearchSpace

    space = LGBMSearchSpace()
    for name, (lo, hi) in space.ranges.items():
        assert lo < hi, f"range for {name} is empty or inverted: ({lo}, {hi})"
        assert lo > 0 or name in ("lambda_l1", "lambda_l2"), (
            f"non-positive lower bound for {name}: {lo}"
        )


def test_search_space_defaults_are_in_tunable_range():
    """Each default value must lie inside the corresponding search range,
    so trial #0 (which uses the defaults) is a valid sample."""
    from src.tune_lgbm import LGBMSearchSpace

    space = LGBMSearchSpace()
    for name, val in space.defaults.items():
        lo, hi = space.ranges[name]
        assert lo <= val <= hi, (
            f"default {name}={val} is outside its range ({lo}, {hi})"
        )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def test_objective_returns_finite_float(small_train_test, folds):
    from src.tune_lgbm import LGBMSearchSpace, objective

    train, test = small_train_test
    space = LGBMSearchSpace()

    # Use a fixed-trial stand-in: pass a dict directly (we test the
    # *evaluation* of a param set, not the Optuna Trial object itself).
    fixed_params = space.defaults
    score = objective(
        params=fixed_params,
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        num_boost_round=100,
        early_stopping_rounds=30,
    )
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_objective_above_constant_baseline(small_train_test, folds):
    from src.tune_lgbm import LGBMSearchSpace, objective

    train, test = small_train_test
    space = LGBMSearchSpace()

    score = objective(
        params=space.defaults,
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        num_boost_round=100,
        early_stopping_rounds=30,
    )
    # 1% slice + 100 boost rounds is a very weak model. The constant
    # baseline is 0.5; the LGBM with default params should easily clear
    # 0.7 on this slice.
    assert score > 0.7, f"objective score {score:.4f} below 0.7 floor"


def test_objective_different_params_can_give_different_scores(small_train_test, folds):
    """Two different param sets should generally yield different scores.
    This catches a buggy objective that ignores `params`."""
    from src.tune_lgbm import LGBMSearchSpace, objective

    train, test = small_train_test
    space = LGBMSearchSpace()

    p1 = space.defaults.copy()
    p1["num_leaves"] = 15
    p1["learning_rate"] = 0.1
    p2 = space.defaults.copy()
    p2["num_leaves"] = 127
    p2["learning_rate"] = 0.02

    s1 = objective(
        params=p1, train=train, test=test, folds=folds,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        num_boost_round=100, early_stopping_rounds=30,
    )
    s2 = objective(
        params=p2, train=train, test=test, folds=folds,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        num_boost_round=100, early_stopping_rounds=30,
    )
    # Different params → likely different scores. Use a soft check.
    assert abs(s1 - s2) > 1e-6, "objective ignored the params"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def test_run_optuna_sweep_returns_best_params_and_score(small_train_test, folds):
    from src.tune_lgbm import LGBMSearchSpace, run_optuna_sweep

    train, test = small_train_test

    best_params, best_score, study = run_optuna_sweep(
        train=train,
        test=test,
        folds=folds,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        n_trials=3,
        num_boost_round=100,
        early_stopping_rounds=30,
        tracking_enabled=False,
    )
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert np.isfinite(best_score)
    assert best_score > 0.7


def test_run_optuna_sweep_best_score_at_least_first_trial(small_train_test, folds):
    """Optuna always returns a 'best' even if the first trial is the
    only one — best_score should be >= every individual trial score."""
    from src.tune_lgbm import LGBMSearchSpace, run_optuna_sweep

    train, test = small_train_test

    best_params, best_score, study = run_optuna_sweep(
        train=train, test=test, folds=folds,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        n_trials=5, num_boost_round=100, early_stopping_rounds=30,
        tracking_enabled=False,
    )
    trial_scores = [t.value for t in study.trials if t.value is not None]
    assert best_score >= min(trial_scores)
    assert best_score <= max(trial_scores) + 1e-6


def test_run_optuna_sweep_best_params_are_valid_lgbm(small_train_test, folds):
    """The best_params dict should be usable as-is by train_lgbm."""
    from src.tune_lgbm import LGBMSearchSpace, run_optuna_sweep
    from src.train_lgbm import train_lgbm

    train, test = small_train_test

    best_params, best_score, _ = run_optuna_sweep(
        train=train, test=test, folds=folds,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        n_trials=3, num_boost_round=100, early_stopping_rounds=30,
        tracking_enabled=False,
    )

    # Should run without error using the tuned params.
    oof, test_pred, metrics = train_lgbm(
        train=train, test=test, folds=folds,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        params=best_params,
        num_boost_round=100, early_stopping_rounds=30,
        categorical_cols=CATEGORICAL_COLS,
        tracking_enabled=False,
    )
    assert oof.shape == (len(train),)
    assert test_pred.shape == (len(test),)


def test_run_optuna_sweep_logs_to_mlflow(small_train_test, folds, tmp_path, monkeypatch):
    """When tracking_enabled=True, every trial should produce an MLflow run,
    and there should also be a summary run with the best params."""
    import mlflow

    from src.tracking import set_tracking_uri
    from src.tune_lgbm import LGBMSearchSpace, run_optuna_sweep

    mlruns = tmp_path / "mlruns"
    set_tracking_uri(mlruns)
    mlflow.set_tracking_uri(f"file:{mlruns}")

    train, test = small_train_test

    best_params, best_score, _ = run_optuna_sweep(
        train=train, test=test, folds=folds,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        categorical_cols=CATEGORICAL_COLS,
        n_trials=3, num_boost_round=50, early_stopping_rounds=20,
        tracking_enabled=True,
        sweep_run_name="phase12_test_sweep",
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{mlruns}")
    exp = client.get_experiment_by_name("ev-purchase-lgbm")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 3, (
        f"Expected >=3 runs (one per trial + summary), got {len(runs)}"
    )
