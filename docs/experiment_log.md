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

## Experiment 2 — 2026-09-03 (Submission pipeline unit tests, full 286k scale)

**Hypothesis:** `make_submission` should produce a Kaggle-valid file at any scale, with id column preserved exactly, target column replaced by predictions, and the 286,571-row template round-tripped cleanly.

**Change made:** Wrote 16 tests in `tests/test_predict.py`:
- 4 happy-path tests (file written, shape, id preserved, predictions in target column).
- 5 validation tests (length mismatch, out-of-range, NaN preds, NaN ids, missing template, missing target column).
- 2 idempotence/integration tests (idempotent, parent dir created).
- 5 production-scale tests (round-trip on 286k template, unique preds at scale, extra columns preserved, id dtype preserved, file integrity check).

**Results:** 16/16 green. The first run had one red: a smoke test that mixed 3-row data with the 286k template, hit a pandas length mismatch, and was fixed by splitting smoke (3-row) from scale (286k).

**What happened:** Confirmed `make_submission` works at production scale. The output CSV has the same row count, same id range, same id dtype, and the target column contains exactly the predictions passed in. Extra template columns (e.g., `extra_meta`) are preserved unchanged.

**Why (your understanding):** Validation happens before any I/O. `template.copy()` then column-assignment preserves the template's column order and any extra columns. The submission is byte-identical between two consecutive calls with the same inputs.

**Next experiment:** Wire `make_submission` into the E2E pipeline test (Phase 8), then build the Kaggle notebook (Phase 10) that runs `load_data → build_features → make_folds → train_lgbm → make_submission` end-to-end and writes `/kaggle/working/submission.csv`.

---

## Experiment 3 — 2026-09-03 (E2E pipeline on 1% train slice, full test set)

**Hypothesis:** Wiring `load_data → build_features → make_folds → train_lgbm → make_submission` into a single E2E test on a 1% train slice should produce a valid submission file in under 2 minutes and an OOF AUC well above the 0.85 floor.

**Change made:** Wrote `tests/test_pipeline_e2e.py` with 8 assertions (file exists, shape, id preservation, AUC floor, metrics keys, prediction validity, etc.) wired through a module-scoped fixture that runs the full pipeline once and reuses the artifacts.

**Results:** 8/8 green, total run time ~85 s.
- OOF AUC on the 1% slice: ~0.93 (matches Experiment 1).
- Submission file: 286,571 rows × 2 columns, id column matches the test set exactly, `Will_Buy_EV` column contains the trainer's predictions in the same order.

**What happened:** E2E green on first red→fix cycle. The only bug was the E2E fixture slicing the test set instead of the train set — same shape mismatch as the Phase 7 smoke test, same fix pattern (slice the right thing). The `src/` library had no bugs.

**Why (your understanding):** Each module's contract held under integration. The LightGBM categorical contract, the submission validation order, the CV fold shape — all survived the wiring. This is the strongest signal we have so far that the library is correct.

**Next experiment:** Build the Kaggle notebook (Phase 10) that runs the same pipeline at full data scale (668k train rows, 286k test rows, ~5-10 min on Kaggle CPU) and writes `/kaggle/working/submission.csv`. Expected OOF AUC: 0.93-0.95.

---

## Experiment 4 — 2026-09-03 (Full-data run, 668,665 train rows)

**Hypothesis:** Running the v1 LightGBM on the full training set (no slicing) should produce an OOF AUC of 0.93-0.95 and a stable fold std (< 0.005). The 1% smoke test in Experiment 1 hit ~0.93; the full run should match or slightly exceed it because of the larger training data per fold.

**Change made:** Wrote `scripts/train_full.py` (production mirror of `train.ipynb`). Ran it on the full data with the default LGBM params, `num_boost_round=1500`, `early_stopping_rounds=100`, 5-fold CV. Tracked via MLflow under run name `phase11_full_data_v1`.

**Results:**

| Metric | Value |
|---|---|
| Mean CV AUC (5 folds) | **0.94181** |
| Std CV AUC | 0.00080 |
| Per-fold AUC | [0.94053, 0.94160, 0.94294, 0.94232, 0.94168] |
| Pooled OOF AUC (all 668,665 rows) | 0.94180 |
| Runtime | 509 s (~8.5 min on local CPU) |
| Submission file | `submissions/submission_lgbm_v1.csv` (286,571 rows) |
| Will_Buy_EV range | [0.0001, 0.9577] |
| Will_Buy_EV mean | 0.1747 (base rate: 0.1746) |

**What happened:** Beat the 0.93 floor by ~1.2 percentage points. The model is stable across folds (std 0.0008). The 0.001 gap between mean-of-folds (0.94181) and pooled OOF (0.94180) confirms the per-fold-vs-pooled gap is negligible at scale. Submission mean (0.1747) matches the train base rate (0.1746), as expected from a well-calibrated ranking model.

**Why (your understanding):** The full 668k rows give each fold's training set ~535k rows (vs ~5.3k in the 1% smoke test). That's a 100x increase in data per fold, which the LGBM converts directly into better splits and better OOF predictions. The 0.93 → 0.94 jump comes from the model being able to learn finer-grained interaction effects that the 1% slice didn't have enough samples to estimate.

**Next experiment:** Submit to Kaggle and record the public leaderboard score. Then start the Optuna sweep (Phase 12) to see if the OOF AUC can be pushed to 0.95+ via hyperparameter tuning.

---

