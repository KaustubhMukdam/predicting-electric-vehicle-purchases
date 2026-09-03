# Model Card — Predicting Electric Vehicle Purchases

## Model overview
- **Type:** LightGBM binary classifier
- **Task:** Predict `Will_Buy_EV` (Yes/No) from 13 raw + 6 engineered features.
- **Training date:** 2026-09-03
- **Framework:** LightGBM 4.7.0
- **Tracking:** MLflow run `phase11_full_data_v1` (local file backend at `mlruns/`)
- **Pipeline run:** `scripts/train_full.py` (8.5 min wall time on local CPU)

## Training data
- **Source:** Kaggle PS S6E9 `train.csv` (synthetic)
- **Size:** 668,665 rows × 19 feature columns (after dropping `id`)
- **Target variable:** `Will_Buy_EV` — 82.5% No, 17.5% Yes
- **Preprocessing:** Ordinal-encoded `Range_Anxiety_Level`; added 6 engineered features; cast `Environmental_Concern_Level` from float to int. No imputation, no scaling.
- **Split:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — each fold has 133,733 rows and a positive rate of 0.1746 (matches overall).

## Features used
**Raw numeric (7):** `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`.

**Raw categorical (6):** `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level` (passed to LightGBM as native categoricals; ordinal-encoded version also added as `Anxiety_ord` for interaction features).

**Engineered (6):** `Anxiety_ord`, `Stations_Total`, `Income_per_Age`, `Env_x_Subsidy`, `Subsidy_x_Income`, `Anxiety_x_Stations_Total`.

## Performance

| Metric | Train (mean of 5 folds) | OOF (pooled, 668,665 rows) | Submission |
|---|---|---|---|
| ROC-AUC | 0.94181 (std 0.00080) | 0.94180 | 286,571 rows |
| Per-fold ROC-AUC | [0.94053, 0.94160, 0.94294, 0.94232, 0.94168] | n/a | n/a |
| Will_Buy_EV range | n/a | n/a | [0.0001, 0.9577] |
| Will_Buy_EV mean | n/a | n/a | 0.1747 (base rate: 0.1746) |

The OOF AUC of 0.9418 is well above the constant baseline (0.5) and the logistic-regression target (0.86). It is also stable across folds (std 0.0008 = 0.08 percentage points).

## Hyperparameters

```python
{
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
# num_boost_round: 1500 (capped; early stopping at 100)
```

## What it does well
- **Ranking quality**: 0.9418 OOF AUC means a randomly chosen EV buyer is ranked above a randomly chosen non-buyer 94.18% of the time. This is competitive on the PS-S6E9 leaderboard (estimated top-50% range, pending submission).
- **Stability**: The 0.0008 std across folds means the model is not overfitting to any particular subset of the data. New data from the same distribution should produce similar AUC.
- **Handles the class imbalance**: No special weighting or resampling was needed — ROC-AUC is threshold-free, and LightGBM's histogram splits handle the 82.5/17.5 split natively.

## Known limitations
- **Synthetic data**: This is a Kaggle Playground Series dataset, generated from a real survey-style dataset. The model may not generalize to actual EV purchase behavior.
- **Strong-signal features dominate**: 4 of the 19 features (`Range_Anxiety_Level`, `Subsidy_Available`, `Environmental_Concern_Level`, `Annual_Income_USD`) carry most of the signal. If any of these were missing or mismeasured in production, performance would degrade sharply.
- **Weak-signal noise**: `Gender`, `Number_of_Cars_Owned`, and `Age` add mostly noise. They are not harmful but are not helpful either.
- **Income is bounded**: The synthetic income distribution is bounded at $30k-188k. Real income distributions have heavier tails; the model's behavior on out-of-distribution incomes is unverified.
- **No probability calibration**: The raw scores are well-ranked but not calibrated. If a downstream consumer needs a probability (e.g., "60% likely to convert"), they should calibrate via isotonic regression on a held-out set.

## Bias and fairness
- **Demographic parity checked:** No. The dataset includes `Gender` but the predictive task is not gated on demographic fairness. EV purchase prediction is not a protected-class decision in the usual sense.
- **Groups evaluated:** None. Future work could include per-gender AUC to detect uneven performance.

## Intended use
- Kaggle competition submission only.
- Learning artifact for a single-user portfolio project.

## Out-of-scope use
- Real-world EV marketing or subsidy targeting. The dataset is synthetic and the model has not been validated on real purchase behavior.
- Any decision that affects individuals (lending, insurance, incentive allocation) without further validation on real, representative data.

## Artifacts produced
- `submissions/submission_lgbm_v1.csv` — 286,571 rows, ready for Kaggle upload
- `submissions/run_metrics_v1.json` — run metrics in machine-readable form
- `mlruns/` — MLflow local tracking directory (params, metrics, OOF/test artifacts)
- OOF and test predictions logged as `oof/oof.npy` and `oof/test_pred.npy` inside the MLflow run
