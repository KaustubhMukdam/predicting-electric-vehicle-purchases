# Evaluation — Predicting Electric Vehicle Purchases

## Why these metrics

### Primary metric: ROC-AUC
- **What it measures:** The probability that a randomly chosen positive example is ranked above a randomly chosen negative example by the model. Equivalent to the area under the ROC curve.
- **Why:** The competition uses it. It is threshold-free, so it doesn't punish us for outputting well-calibrated probabilities vs raw scores, and it doesn't require us to pick a decision threshold.
- **Range:** 0.5 (random) to 1.0 (perfect). 0.0 means perfectly anti-correlated.
- **What we report:** Mean 5-fold CV ROC-AUC. Std across folds. Per-fold AUCs (in MLflow).

### Secondary metric: PR-AUC (Precision-Recall AUC)
- **What it measures:** Area under the precision-recall curve.
- **Why:** With 17.5% positive class, PR-AUC is more sensitive to the model's behavior on the minority class. It complements ROC-AUC because ROC-AUC can look optimistic on imbalanced data.
- **When to look at it:** If our CV ROC-AUC is strong but the model is predicting the minority class poorly, PR-AUC will reveal it. Also useful if we later switch to a threshold-based evaluation.
- **Status:** Computed alongside ROC-AUC in `src/train_lgbm.py` and logged to MLflow. Not used as the primary optimization target.

## Baseline

| Baseline | Description | Expected ROC-AUC |
|---|---|---|
| Constant 0.1746 | Predict the base rate for every row (matches `sample_submission.csv`). | 0.5 |
| Logistic regression with raw features | Sanity-check pipeline. | ~0.86 (per plan-mode EDA) |
| **Target for v1 (LGBM Tier 1)** | Single LightGBM with minimal FE. | **≥ 0.93** |
| Stretch (LGBM tuned + ensemble) | Optuna + XGB + CatBoost rank-blend. | ~0.95 |

The constant baseline scores 0.5 ROC-AUC by definition. Anything that produces even a tiny amount of ranking signal will beat it. Our model needs to beat the logistic baseline by ≥0.07 AUC to be worth shipping as "v1".

## Results vs baseline
*(Filled in after first training run. Template below.)*

| Run | Model | Mean CV ROC-AUC | Std | PR-AUC | Notes |
|---|---|---|---|---|---|
| 0 | Constant 0.1746 | 0.5000 | 0.0000 | — | Sample submission |
| 1 | Logistic regression (raw features) | TBD | TBD | TBD | Sanity check |
| 2 | LightGBM v1 (minimal FE) | TBD | TBD | TBD | First real model |

## Error analysis (planned for after first run)
After the v1 LGBM trains, we will:
1. Plot the OOF prediction distribution split by class. If the two distributions are nearly identical, the model is barely using the signal. If they separate cleanly, we are in good shape.
2. Compute the confusion matrix at a few thresholds (0.1, 0.175, 0.3, 0.5) to see if the model's natural threshold lines up with the base rate.
3. Compute per-segment AUCs: one AUC per `Range_Anxiety_Level` value, per `Subsidy_Available` value, per `City_Type`. This tells us where the model is strong vs weak.
4. Look at the top-N highest-confidence wrong predictions (false positives at high threshold, false negatives at low threshold). These are the cases to dig into if a v2 model is needed.

## What the numbers actually mean
For business interpretation (not needed for the leaderboard but useful to write down):
- **ROC-AUC of 0.94** = if you pick a random EV buyer and a random non-buyer, the model ranks the buyer higher 94% of the time.
- **At a threshold of 0.5**, we expect roughly the right precision/recall split for a top-of-funnel marketing list.
- **At a threshold of 0.175 (the base rate)**, we expect the model's "top half" of predictions to be where most of the actual buyers are.

The Kaggle leaderboard ranks submissions by ROC-AUC, so we never need to set a threshold. The threshold discussion above is purely for our own understanding of what the model is doing.
