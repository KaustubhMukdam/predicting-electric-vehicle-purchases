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

*(Append a new section below for each subsequent run. Do not overwrite prior sections.)*
