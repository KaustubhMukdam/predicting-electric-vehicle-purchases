"""Optuna hyperparameter sweep for the LightGBM trainer.

Search space
------------
We tune 9 LGBM hyperparameters over sensible ranges. Defaults match
`DEFAULT_LGBM_PARAMS` in `src/train_lgbm.py` so trial #0 reproduces
the v1 baseline.

Objective
---------
Mean 5-fold CV ROC-AUC. (Per-fold AUCs are computed by `train_lgbm`;
we take the mean.)

Tracking
--------
Each trial is logged as a separate MLflow run with its params and
score. A summary run (`<sweep_run_name>_best`) is also created with
the best params, the best score, and the full study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src import tracking
from src.train_lgbm import DEFAULT_LGBM_PARAMS, train_lgbm


# Optuna logs a lot at INFO; quiet it down so our MLflow run output
# stays readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class LGBMSearchSpace:
    """Defines what we tune and over what range."""

    # name -> (low, high) for `optuna.suggest_float` / `suggest_int`.
    # `log` is a flag we honor in `objective` (learning_rate uses log scale).
    ranges: dict = field(default_factory=lambda: {
        "learning_rate": (0.01, 0.1),     # log scale
        "num_leaves": (15, 255),          # int
        "max_depth": (4, 12),             # int; -1 (no cap) handled outside
        "min_data_in_leaf": (20, 300),    # int
        "feature_fraction": (0.6, 1.0),
        "bagging_fraction": (0.6, 1.0),
        "bagging_freq": (1, 10),          # int
        "lambda_l1": (1e-3, 10.0),        # log scale
        "lambda_l2": (1e-3, 10.0),        # log scale
    })
    log_scale: frozenset = field(default_factory=lambda: frozenset({
        "learning_rate", "lambda_l1", "lambda_l2",
    }))
    int_params: frozenset = field(default_factory=lambda: frozenset({
        "num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq",
    }))

    @property
    def defaults(self) -> dict:
        """Default values for each tunable — pulled from DEFAULT_LGBM_PARAMS
        with a small amount of clipping (e.g., max_depth=-1 -> 8, since
        Optuna requires a finite integer range)."""
        # max_depth: LGBM default is -1 (no cap). We tune over a finite
        # range, so map -1 -> max of the range.
        defaults = {
            "learning_rate": DEFAULT_LGBM_PARAMS["learning_rate"],
            "num_leaves": DEFAULT_LGBM_PARAMS["num_leaves"],
            "max_depth": 8 if DEFAULT_LGBM_PARAMS["max_depth"] == -1
                else DEFAULT_LGBM_PARAMS["max_depth"],
            "min_data_in_leaf": DEFAULT_LGBM_PARAMS["min_data_in_leaf"],
            "feature_fraction": DEFAULT_LGBM_PARAMS["feature_fraction"],
            "bagging_fraction": DEFAULT_LGBM_PARAMS["bagging_fraction"],
            "bagging_freq": DEFAULT_LGBM_PARAMS["bagging_freq"],
            "lambda_l1": DEFAULT_LGBM_PARAMS["lambda_l1"]
                if DEFAULT_LGBM_PARAMS["lambda_l1"] > 0 else 0.1,
            "lambda_l2": DEFAULT_LGBM_PARAMS["lambda_l2"]
                if DEFAULT_LGBM_PARAMS["lambda_l2"] > 0 else 0.1,
        }
        return defaults


def objective(
    params: dict,
    train: "pd.DataFrame",
    test: "pd.DataFrame",
    folds: np.ndarray,
    feature_cols: Sequence[str],
    target_col: str,
    categorical_cols: Sequence[str],
    num_boost_round: int = 1500,
    early_stopping_rounds: int | None = 100,
) -> float:
    """Score a single param set via 5-fold CV.

    Returns the mean fold AUC. Lower is worse; Optuna maximizes by
    default.
    """
    # Sanitize params: max_depth=-1 is LGBM's "no cap" sentinel, but
    # we tuned over a finite range. If a trial asks for depth=8 but
    # the user has set the default -1, fall back to -1.
    full_params = DEFAULT_LGBM_PARAMS.copy()
    full_params.update(params)
    if full_params.get("max_depth") is not None and full_params["max_depth"] >= 12:
        full_params["max_depth"] = -1  # let LGBM decide

    _, _, metrics = train_lgbm(
        train=train,
        test=test,
        folds=folds,
        feature_cols=feature_cols,
        target_col=target_col,
        params=full_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        categorical_cols=categorical_cols,
        tracking_enabled=False,  # sweep logs each trial separately
    )
    return metrics["cv_auc_mean"]


def _suggest(trial: optuna.Trial, space: LGBMSearchSpace, name: str):
    lo, hi = space.ranges[name]
    if name in space.int_params:
        return trial.suggest_int(name, int(lo), int(hi))
    if name in space.log_scale:
        return trial.suggest_float(name, float(lo), float(hi), log=True)
    return trial.suggest_float(name, float(lo), float(hi))


def run_optuna_sweep(
    train: "pd.DataFrame",
    test: "pd.DataFrame",
    folds: np.ndarray,
    feature_cols: Sequence[str],
    target_col: str,
    categorical_cols: Sequence[str],
    n_trials: int = 25,
    num_boost_round: int = 1500,
    early_stopping_rounds: int | None = 100,
    tracking_enabled: bool = True,
    sweep_run_name: str = "lgbm_optuna_sweep",
    seed: int = 42,
) -> tuple[dict, float, optuna.Study]:
    """Run an Optuna sweep over the LGBM search space.

    Returns
    -------
    best_params : dict
        The param set with the highest mean CV AUC.
    best_score : float
        The corresponding mean CV AUC.
    study : optuna.Study
        The full Optuna study (for inspection / re-use).
    """
    space = LGBMSearchSpace()
    sampler = TPESampler(seed=seed, multivariate=True)

    # Suppress the default per-trial log; we'll handle our own.
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    def _objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, space, name) for name in space.ranges}
        if tracking_enabled:
            with tracking.start_run(
                run_name=f"{sweep_run_name}_trial_{trial.number}"
            ):
                tracking.log_params(params)
                score = objective(
                    params=params,
                    train=train, test=test, folds=folds,
                    feature_cols=feature_cols, target_col=target_col,
                    categorical_cols=categorical_cols,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                )
                tracking.log_metric("cv_auc_mean", score)
                return score
        return objective(
            params=params,
            train=train, test=test, folds=folds,
            feature_cols=feature_cols, target_col=target_col,
            categorical_cols=categorical_cols,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_score = float(study.best_value)

    if tracking_enabled:
        with tracking.start_run(run_name=f"{sweep_run_name}_best"):
            tracking.log_params(best_params)
            tracking.log_metric("cv_auc_mean", best_score)
            tracking.log_metric("n_trials", n_trials)

    return best_params, best_score, study
