# Learnings

> Personal knowledge base built from real work. Update at the end of every session (5 min max).
> Format follows `dev_system_guide.md` §6.1.

---

## 2026-09-02 — Predicting Electric Vehicle Purchases (project start)

### What I learned
**Kaggle Playground Series datasets are clean by design.** The PS-S6E9 EV dataset had zero missing values, zero duplicates, and consistent category sets between train and test. This means the data-loader can skip the usual defensive code (imputation, dtype coercion for nulls) and focus on schema validation instead.

**ROC-AUC as a competition metric is a strong signal to use LightGBM with `metric='auc'`.** LightGBM's default split criterion (gini / entropy) does not directly optimize AUC, but it ranks well empirically and supports `metric='auc'` as a tracked eval metric. Calibration is irrelevant — what matters is the *ranking* of predictions.

**Synthetic Playground data has rounding artifacts.** Annual_Income_USD had 13,214 unique values (out of 668,665 rows). That looks like real data, but Daily_Commute_km has only 805 unique values — suspiciously round. Worth noting for any model that assumes continuous distributions.

### Code snippet that clicked
```python
# Map binary categorical target to 0/1 in one line, no sklearn LabelEncoder needed
train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)
```

This is cleaner than `LabelEncoder().fit_transform(train['Will_Buy_EV'])` because the mapping is explicit and the `0`/`1` values are documented in the code.

### What confused me today
**Should I ordinal-encode Range_Anxiety_Level or pass it as a native LightGBM categorical?** Both are valid. I went with native categorical for consistency with the other string columns, and added a separate `Anxiety_ord` column for use in the interaction features. LightGBM will see both — slight redundancy but the interaction features need the numeric form.

### How I solved it
Decided to pass all 6 string categoricals to LightGBM's native handler, and compute `Anxiety_ord` only for the engineered interaction features. This separates concerns: LightGBM handles the raw signal, the engineered features encode domain priors.

### What I'd do differently
- Run a quick sklearn `LogisticRegression` baseline first (1 line of code) to confirm the pipeline produces a sensible AUC before committing to the full LGBM infrastructure.
- Add a `make_pipeline.py` script that wires `load_data → build_features → make_folds → train_lgbm → make_submission` into one call, so the notebook is just `pipeline.run()`.
