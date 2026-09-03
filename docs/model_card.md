# Model Card — Predicting Electric Vehicle Purchases

> Populated after the first training run. Template below.

## Model overview
- **Type:** LightGBM binary classifier
- **Task:** Predict `Will_Buy_EV` (Yes/No) from 13 raw + 6 engineered features.
- **Training date:** TBD
- **Framework:** LightGBM 4.3.x
- **Tracking:** MLflow run ID TBD

## Training data
- **Source:** Kaggle PS S6E9 train.csv (synthetic)
- **Size:** 668,665 rows × 14 features (after dropping `id`)
- **Target variable:** `Will_Buy_EV` — 82.5% No, 17.5% Yes
- **Preprocessing:** Ordinal-encoded `Range_Anxiety_Level`; added 6 engineered features; cast `Environmental_Concern_Level` from float to int. No imputation, no scaling.
- **Split:** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## Features used
**Raw numeric (7):** `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`.

**Raw categorical (6):** `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level` (passed to LightGBM as native categoricals; ordinal-encoded version also added as `Anxiety_ord` for interaction features).

**Engineered (6):** `Anxiety_ord`, `Stations_Total`, `Income_per_Age`, `Env_x_Subsidy`, `Subsidy_x_Income`, `Anxiety_x_Stations_Total`.

## Performance
*(Template — fill after first run.)*

| Metric | Train (mean of 5 folds) | OOF (held-out) | Kaggle Public LB |
|---|---|---|---|
| ROC-AUC | TBD | TBD | TBD |
| PR-AUC | TBD | TBD | — |
| Log loss | TBD | TBD | — |

## Hyperparameters
*(Template — fill after first run.)*

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
```

## What it does well
*(To be filled.)*

## Known limitations
*(To be filled. Expected items:*
- *Synthetic data — model may not generalize to real EV purchase behavior.*
- *Some features have very weak signal (Gender, Number_of_Cars_Owned) — they add noise more than information.*
- *Income is bounded in the synthetic generation; real income distributions have heavier tails.)*

## Bias and fairness
- **Demographic parity checked:** No. The dataset includes `Gender` but the predictive task is not gated on demographic fairness. EV purchase prediction is not a protected-class decision in the usual sense.
- **Groups evaluated:** None yet. Future work could include per-gender AUC to detect uneven performance.

## Intended use
- Kaggle competition submission only.
- Learning artifact for a single-user portfolio project.

## Out-of-scope use
- Real-world EV marketing or subsidy targeting. The dataset is synthetic and the model has not been validated on real purchase behavior.
- Any decision that affects individuals (lending, insurance, incentive allocation) without further validation on real, representative data.
