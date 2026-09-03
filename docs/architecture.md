# Architecture — Predicting Electric Vehicle Purchases

## System overview
A batch ML pipeline that reads two CSV files (train, test), engineers a small set of features, runs a single LightGBM model under stratified 5-fold cross-validation, tracks every run via MLflow, and writes a `submission.csv` whose format matches the Kaggle `sample_submission.csv`. The pipeline is implemented as small composable Python modules in `src/`, each with a paired pytest file, and is driven either directly from a Python script (local) or from a Jupyter notebook (Kaggle).

## Component diagram (ASCII)
```
data/train.csv
data/test.csv
data/sample_submission.csv
        |
        v
[src/data.py] load_data()
        |
        v
[src/features.py] build_features()
        |
        v
[src/cv.py] make_folds()
        |
        +------------------+
        |                  |
        v                  v
[src/train_lgbm.py]    (OOF preds, test preds, fold AUCs)
        |
        v
[src/tracking.py] MLflow.log_*()
        |
        v
[src/predict.py] make_submission()
        |
        v
submissions/submission_lgbm_v1.csv   (or /kaggle/working/submission.csv on Kaggle)
```

## Data flow
1. **`src/data.py:load_data(train_path, test_path)`** reads the two CSVs, validates the schema (expected columns, dtypes), maps the target column (`Will_Buy_EV: {No: 0, Yes: 1}`), and returns `(train_df, test_df)`.
2. **`src/features.py:build_features(df)`** adds engineered columns in place (or returns a copy): `Anxiety_ord`, `Stations_Total`, `Income_per_Age`, `Env_x_Subsidy`, `Subsidy_x_Income`, `Anxiety_x_Stations_Total`. Pure function — same input → same output, no fit state.
3. **`src/cv.py:make_folds(y, n_splits=5, seed=42)`** returns an array of fold indices (0..n_splits-1) of the same length as `y`, stratified on the target.
4. **`src/train_lgbm.py:train_lgbm(train, test, folds, params, tracking=True)`** iterates the folds, fits LightGBM on the train portion, predicts on the validation portion (contributing to OOF) and on the full test set (averaged across folds). Returns `(oof_preds, test_preds, cv_metrics)`. If `tracking=True`, it logs params, fold AUCs, mean AUC, std, the OOF array as an artifact, and the test predictions as an artifact via `src/tracking.py`.
5. **`src/predict.py:make_submission(test_ids, test_preds, template_path, out_path)`** reads the `sample_submission.csv` template, replaces its `Will_Buy_EV` column with the predicted probabilities, and writes to `out_path`. Asserts the id column is preserved and probabilities are in [0, 1].
6. The Jupyter notebook (`notebooks/train.ipynb`) wires all of the above into a single end-to-end run and prints a summary.

## Key interfaces
- **All module functions are pure where possible.** No global state, no side effects on import. The only side effect is MLflow logging (controlled by `tracking=True`) and file writes (controlled by explicit `out_path`).
- **The CV split is computed once and passed into the trainer** rather than recomputed inside it. This makes the folds inspectable in tests and reproducible across models.
- **Test data is never used for fitting anything that produces learned state.** No scaler, no target encoder, no imputer in v1. We rely on LightGBM's robustness to scale and on the data having no missing values.

## Module dependency graph
```
config.py        (no deps)
utils.py         (no deps)
data.py          -> pandas, config
features.py      -> pandas, config
cv.py            -> numpy, sklearn, config
tracking.py      -> mlflow
train_lgbm.py    -> lightgbm, sklearn, tracking, config
predict.py       -> pandas
```

No circular deps. `config.py` is leaf-level — it exports constants only.

## Security considerations
- [x] No secrets in code. MLflow file backend, no remote URI.
- [x] No PII in this dataset beyond what is already public on Kaggle (Playground Series is synthetic).
- [x] Submission file does not contain any train data — it only contains `id` and predicted probability.
- [x] `data/` is gitignored so the CSVs never enter the repo.
