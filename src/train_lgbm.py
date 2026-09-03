"""LightGBM trainer with stratified k-fold cross-validation.

The trainer takes a feature-engineered train and test DataFrame, runs
N-fold CV, and returns:

    oof         (n_train,) out-of-fold probability predictions
    test_pred   (n_test,)  averaged-across-folds probability predictions
    metrics     dict with cv_auc_mean, cv_auc_std, fold_aucs

If `tracking=True`, it opens an MLflow run, logs the params, fold
metrics, mean/std, and the OOF and test prediction arrays as artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import tracking


# Default LightGBM hyperparameters for binary classification with native
# categorical handling. Tuned for tabular data; conservative to avoid
# overfitting on the 1% smoke-test slice.
DEFAULT_LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    "seed": 42,
}


@contextmanager
def _maybe_tracking(tracking_enabled: bool, run_name: str | None):
    """Open an MLflow run iff tracking is enabled, else no-op."""
    if tracking_enabled:
        with tracking.start_run(run_name=run_name) as run:
            yield run
    else:
        yield None


def _to_categorical(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    category_unions: dict[str, pd.Index] | None = None,
) -> pd.DataFrame:
    """Cast `categorical_cols` to pandas `category` dtype.

    If `category_unions` is provided, every column is reindexed to the
    same category set (the union of train + test categories) so that
    LightGBM's training-time and prediction-time category checks pass.
    """
    out = df.copy()
    for col in categorical_cols:
        if col not in out.columns:
            continue
        if category_unions is not None and col in category_unions:
            cat_dtype = pd.CategoricalDtype(categories=category_unions[col])
            out[col] = out[col].astype(cat_dtype)
        else:
            out[col] = out[col].astype("category")
    return out


def _build_category_unions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_cols: Sequence[str],
) -> dict[str, pd.Index]:
    """Pre-compute the union of categories per column across train + test.

    This is the fix for LightGBM's "train and valid dataset categorical_feature
    do not match" error: by aligning every fold's validation frame (and the
    test frame) to the same category set as the training data, the
    category-set comparison inside LightGBM always passes.
    """
    unions: dict[str, pd.Index] = {}
    for col in categorical_cols:
        if col not in train.columns or col not in test.columns:
            continue
        combined = pd.concat([train[col], test[col]], axis=0, ignore_index=True)
        unions[col] = pd.Index(combined.unique())
    return unions


def train_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: np.ndarray,
    feature_cols: Sequence[str],
    target_col: str,
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int | None = 100,
    categorical_cols: Sequence[str] = (),
    tracking_enabled: bool = True,
    run_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run N-fold LightGBM CV.

    Parameters
    ----------
    train : DataFrame
        Training data with the target column.
    test : DataFrame
        Test data (no target).
    folds : ndarray of shape (len(train),)
        Integer fold assignment per row. Values in 0..n_folds-1.
    feature_cols : sequence of str
        Column names to use as features.
    target_col : str
        Name of the target column in `train`.
    params : dict, optional
        LightGBM params. Defaults to `DEFAULT_LGBM_PARAMS`.
    num_boost_round : int
        Maximum boosting rounds per fold.
    early_stopping_rounds : int or None
        If not None, stop training if validation AUC doesn't improve
        for this many rounds.
    categorical_cols : sequence of str
        Columns to mark as categorical for LightGBM native handling.
    tracking_enabled : bool
        If True, log the run to MLflow.
    run_name : str, optional
        MLflow run name.

    Returns
    -------
    oof : np.ndarray of shape (len(train),)
        Out-of-fold predicted probabilities.
    test_pred : np.ndarray of shape (len(test),)
        Mean-across-folds predicted probabilities on `test`.
    metrics : dict
        {"cv_auc_mean": float, "cv_auc_std": float, "fold_aucs": list[float]}
    """
    if params is None:
        params = DEFAULT_LGBM_PARAMS.copy()
    else:
        merged = DEFAULT_LGBM_PARAMS.copy()
        merged.update(params)
        params = merged

    fold_ids = np.unique(folds)
    n_folds = len(fold_ids)
    y_train = train[target_col].to_numpy()

    # Align every DataFrame we'll feed to LightGBM to a shared category
    # set so the categorical_feature check never trips.
    cat_unions = _build_category_unions(train, test, categorical_cols)
    cat_cols = list(categorical_cols)
    test_aligned = _to_categorical(
        test.loc[:, list(feature_cols)], cat_cols, cat_unions
    )

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

    with _maybe_tracking(tracking_enabled, run_name):
        if tracking_enabled:
            tracking.log_params({**params, "num_boost_round": num_boost_round})

        for k in fold_ids:
            tr_mask = folds != k
            va_mask = folds == k

            train_X = _to_categorical(
                train.loc[tr_mask, list(feature_cols)], cat_cols, cat_unions
            )
            valid_X = _to_categorical(
                train.loc[va_mask, list(feature_cols)], cat_cols, cat_unions
            )

            train_set = lgb.Dataset(
                train_X,
                label=y_train[tr_mask],
                categorical_feature=cat_cols,
            )
            valid_set = lgb.Dataset(
                valid_X,
                label=y_train[va_mask],
                categorical_feature=cat_cols,
                reference=train_set,
            )

            callbacks: list = []
            if early_stopping_rounds is not None:
                callbacks.append(
                    lgb.early_stopping(early_stopping_rounds, verbose=False)
                )
            callbacks.append(lgb.log_evaluation(period=0))

            booster = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=num_boost_round,
                valid_sets=[valid_set],
                callbacks=callbacks,
            )

            valid_pred = booster.predict(valid_X, categorical_feature=cat_cols)
            oof[va_mask] = valid_pred
            test_pred += booster.predict(test_aligned, categorical_feature=cat_cols) / n_folds

            fold_auc = roc_auc_score(y_train[va_mask], valid_pred)
            fold_aucs.append(float(fold_auc))
            if tracking_enabled:
                tracking.log_metric(f"fold_{int(k)}_auc", fold_auc)

        cv_mean = float(np.mean(fold_aucs))
        cv_std = float(np.std(fold_aucs))
        metrics = {
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std,
            "fold_aucs": fold_aucs,
        }

        if tracking_enabled:
            tracking.log_metrics({"cv_auc_mean": cv_mean, "cv_auc_std": cv_std})
            tracking.log_numpy_array(oof, name="oof.npy", artifact_path="oof")
            tracking.log_numpy_array(test_pred, name="test_pred.npy", artifact_path="oof")

    return oof, test_pred, metrics
