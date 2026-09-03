# project_context.md

> Single source of truth. Paste at the top of every AI chat.

## Project
- **Name:** predicting-electric-vehicle-purchases
- **Status:** in-progress (Phase 0)
- **Started:** 2026-09-02

## One-liner
Binary classification on a Kaggle Playground Series dataset — predict whether a person will buy an EV based on demographics, commute, income, charging infrastructure, and attitudinal features. Maximize ROC-AUC.

## Stack
- **Language:** Python 3.11
- **Data:** pandas, numpy
- **ML:** scikit-learn (CV, metrics), LightGBM (model)
- **Tracking:** MLflow (local SQLite file backend; `/kaggle/working/mlruns/` on Kaggle)
- **Tuning (later):** Optuna
- **Tests:** pytest
- **Notebooks:** Jupyter (run locally or on Kaggle web)

## Key decisions made
- Tier 1 only for v1: single LightGBM model with minimal engineered features. XGB / CatBoost / stacking deferred.
- ROC-AUC is the competition metric. We optimize for ranking quality, not calibration. Loss = `binary`; predictions stored as raw probabilities, never thresholded.
- Stratified 5-fold CV with `random_state=42` for reproducibility across runs.
- Categorical features are passed to LightGBM via its native `categorical_feature` parameter (no one-hot in v1). `Range_Anxiety_Level` is the exception: it has a natural order so it is ordinal-encoded.
- Tracking via MLflow (file backend) instead of W&B to avoid external accounts and to keep everything in-repo. On Kaggle, runs land in `/kaggle/working/mlruns/` and are zipped as a notebook output for download.
- All tests use the real `data/train.csv` and `data/test.csv` (no synthetic fixtures) so we catch real-data quirks.

## Current focus
Phase 0 — project skeleton (requirements, pytest config, .gitignore, empty `src/` and `tests/` placeholders). No code yet.

## Known issues / blockers
- None. EDA completed in plan mode: 668,665 train rows, 286,571 test rows, no missing values, no duplicates, target imbalance 82.5% / 17.5%. Strong signals: `Range_Anxiety_Level`, `Subsidy_Available`, `Environmental_Concern_Level`, `Annual_Income_USD`. Weak signals: `Gender`, `Number_of_Cars_Owned`, commute distance.
- Potential issue: LightGBM native categorical handling requires consistent category sets between train and test. Loader must validate this in Phase 2 (data loader).

## What this project is NOT doing
- Not building a deployable web app or API.
- Not doing threshold tuning, calibration, or probability shrinkage — ROC-AUC is threshold-free.
- Not stacking, pseudo-labeling, or DNNs in v1.
- Not pushing to git. All git operations are the user's responsibility (per their explicit instruction).
- Not installing dependencies outside the Linux `.venv`.

## Competition reference
- Competition: [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9)
- Metric: ROC-AUC
- Submission format: `id, Will_Buy_EV` (probability in [0, 1])
