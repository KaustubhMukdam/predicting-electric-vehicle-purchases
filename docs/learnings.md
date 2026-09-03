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

---

## 2026-09-03 — Phase 6 (LGBM trainer)

### What I learned
**LightGBM's native categorical handling is a contract, not a flag.** Passing `categorical_feature=[...]` to `lgb.Dataset` is half the contract. The other half is that *every* DataFrame fed to the trained booster (per-fold validation slices, the test set, future inference data) must use the same `category` dtype AND the same category *set*. If even one column has a category in val that the training fold didn't see (or vice versa), the booster raises `ValueError: train and valid dataset categorical_feature do not match`. The fix is to pre-compute the union of categories across train + test once, then reindex every DataFrame to that union before it reaches LightGBM.

**Two AUC numbers from the same OOF array are not the same number.** Per-fold AUC (mean of `roc_auc_score(y[fold_k], oof[fold_k])` for k = 0..4) and pooled AUC (`roc_auc_score(y, oof)`) measure subtly different things — pooled AUC is one ranking across all 668k rows, per-fold mean is the average of 5 separate rankings on different positive-rate slices. They differ by ~0.002-0.003 on this dataset, which is normal. Tests should not assert equality, only an empirical tolerance.

**`MlflowClient.list_artifacts(run_id)` is non-recursive.** The method returns only files at the run's artifact root. To inspect files logged under a subdir like `oof/`, you have to call `list_artifacts(run_id, path="oof")` explicitly. Tests that union both sets cover both cases.

### Code snippet that clicked
```python
# Pre-compute category unions so train/val/test share a category set
def _build_category_unions(train, test, categorical_cols):
    unions = {}
    for col in categorical_cols:
        if col in train.columns and col in test.columns:
            combined = pd.concat([train[col], test[col]], axis=0, ignore_index=True)
            unions[col] = pd.Index(combined.unique())
    return unions

# Then before every LightGBM call:
df_aligned = df.copy()
for col in categorical_cols:
    df_aligned[col] = df_aligned[col].astype(
        pd.CategoricalDtype(categories=unions[col])
    )
```

This is the canonical recipe. Without it, you're one CV-fold partition away from a hard crash.

### What confused me today
Whether to use `lgb.Dataset` for the prediction-time data or just a raw DataFrame. The booster's `predict()` accepts either, but the categorical contract only holds for DataFrames that have been explicitly cast to `category` dtype. The cleanest separation: build `lgb.Dataset` only for training/validation (where LightGBM needs to compute bin edges), and use raw aligned DataFrames for prediction.

### How I solved it
Three real bugs hit, all caught by tests, all from the same family: "LightGBM/Mlflow contracts are stricter than they look". Refactored the trainer to compute category unions up front and apply them in one place (`_to_categorical`). The rest of the trainer stayed roughly the same.

### What I'd do differently
- For any library that has a "do X to my data" contract (LightGBM categoricals, sklearn's `check_is_fitted`, MLflow's experiment lookups), write **one** focused test that proves the contract before writing the bulk of the integration. A 5-line contract test catches 80% of integration bugs at zero cost.
- When asserting equivalence between two metrics, run both computations once during development, observe the empirical gap, and bake that gap into the test tolerance. Don't write `< 1e-6` and hope.

### What I learned
**MLflow's API has shifted across versions.** `mlflow.set_experiment` returns an `Experiment` object in 2.22.x, not a string id. Older tutorials and even some current Stack Overflow answers still describe the "returns the id" behavior. Always check the installed version's return type before wrapping.

**The "default experiment" assumption is a trap.** The default experiment (id `"0"`) only exists if no one has ever called `set_experiment`. The moment you wrap MLflow with a function that always calls `set_experiment(some_name)`, the default experiment is bypassed and any test that searches by id `"0"` will silently return 0 runs. The robust pattern is to always look up the experiment by name via `client.get_experiment_by_name(...)`.

**TDD with library wrappers: tests should use the library the way real callers will.** I wrote tests that hard-coded `experiment_ids=["0"]` because I was thinking about the default experiment. Real callers will know the name (`ev-purchase-lgbm`) and look it up. The tests are now realistic, and they double as usage examples for future code.

### Code snippet that clicked
```python
# Correct way to look up an experiment by name in MLflow 2.x
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("ev-purchase-lgbm")
runs = client.search_runs(experiment_ids=[exp.experiment_id])
```

### What confused me today
Whether the tracking URI was being respected when the test fixture set it. I worried that `_ensure_local_tracking_uri` would overwrite the test's `file:/tmp/...` URI. It doesn't — the function checks `mlflow.get_tracking_uri() in ("", None)` first. Lesson: the helper is a *default* setter, not an *override*.

### How I solved it
Fixed the four real bugs in this order: (1) created the missing `src/tracking.py`, (2) changed return type to string, (3) made tests look up the experiment by name, (4) swapped positional/kwarg in `get_metric_history`. Each fix came from reading the actual error message carefully.

### What I'd do differently
For library wrappers, write **one** end-to-end test that exercises the happy path with realistic calls (e.g., `get_or_create_experiment` → `start_run` → `log_params` → `log_metrics` → query via client) before writing per-function unit tests. That single test would have caught all 4 bugs at once.

