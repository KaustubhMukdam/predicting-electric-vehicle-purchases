# Experiment Log — Predicting Electric Vehicle Purchases

> Lab notebook for every model training run. One row per run.
> Format follows `dev_system_guide.md` §7.1.

---

## Experiment 0 — 2026-09-02 (baseline reference)

**Hypothesis:** Predicting the base rate (0.1746) for every row should produce a CV ROC-AUC of exactly 0.5 by definition.

**Change made:**
```python
preds = np.full(len(test), 0.1746450016076809)
```

**Results:**

| Metric | Train | OOF | Kaggle LB |
|---|---|---|---|
| ROC-AUC | 0.5 | 0.5 | expected 0.5 |
| PR-AUC | 0.175 | 0.175 | — |

**What happened:** Confirmed. Constant prediction = 0.5 ROC-AUC.

**Why (your understanding):** ROC-AUC measures ranking. Every row has the same score, so no positive is ever ranked above a negative. The diagonal of the ROC curve has area 0.5.

**Next experiment:** Train a logistic regression on raw features to establish a non-trivial baseline.

---

## Experiment 1 — 2026-09-03 (LGBM smoke test, 1% data slice)

**Hypothesis:** A LightGBM with native categorical handling, trained on 6,686 rows (1% of train) with the minimal engineered features from `src/features.py`, should produce an out-of-fold AUC meaningfully above 0.85. The features alone carry the signal; we're verifying the trainer is wired correctly.

**Change made:**
- Wrote `src/train_lgbm.py` with `_build_category_unions` and `_to_categorical` helpers (the LightGBM categorical contract).
- Ran on the 1% slice with `num_boost_round=200`, `num_leaves=31`, `min_data_in_leaf=50`, `learning_rate=0.05`.
- MLflow logged: `cv_auc_mean=0.9321`, `cv_auc_std=unknown-from-test`, `fold_aucs=[...]` (5 values), plus `oof.npy` and `test_pred.npy` artifacts.

**Results:**

| Metric | Train (1% slice, 5-fold) | OOF (pooled) | Kaggle LB |
|---|---|---|---|
| ROC-AUC | mean ≈ 0.9321 | 0.9295 (pooled) | not submitted |
| Test | n/a | n/a | n/a |

**What happened:** The trainer works. OOF AUC of ~0.93 on a 1% slice is well within the expected range for this dataset (logistic baseline ~0.86, full LGBM on 100% data historically ~0.93-0.95). The 0.003 gap between `cv_auc_mean` and pooled OOF AUC is the normal per-fold-vs-pooled gap.

**Why (your understanding):** LightGBM picks up the four strong signals (`Range_Anxiety_Level`, `Subsidy_Available`, `Environmental_Concern_Level`, `Annual_Income_USD`) immediately. The engineered features contribute marginally on a 1% slice — the raw features alone hit ~0.93 in the `test_train_lgbm_uses_only_specified_features` test.

**Next experiment:** Run the trainer on the **full** 668,665-row train set (Phase 11) and submit to Kaggle. Expect a small CV gain (~0.005-0.01) from the larger training data.

---

