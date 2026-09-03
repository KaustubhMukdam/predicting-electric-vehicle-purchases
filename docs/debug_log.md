# Debug Log

> Tracks every non-trivial bug solved. Prevents solving the same bug twice.
> Format follows `dev_system_guide.md` §6.2.

---

## 2026-09-03 — `ModuleNotFoundError: No module named 'src.tracking'` (9 tests red)

**Project:** predicting-electric-vehicle-purchases (Phase 5)
**Error message:** `ModuleNotFoundError: No module named 'src.tracking'` across all 9 tests in `tests/test_tracking.py`.

**Root cause:** I wrote the tests for Phase 5 but never wrote `src/tracking.py`. Tests red because the module didn't exist. The tracking wrapper is what `train_lgbm.py` will need in Phase 6, so this was a real gap, not a cosmetic one.

**Fix:** Created `src/tracking.py` with the full wrapper API the tests required: `set_tracking_uri`, `_ensure_local_tracking_uri`, `get_or_create_experiment`, `start_run` (context manager), `log_params`, `log_metrics`, `log_metric`, `log_artifact`, `log_numpy_array`.

**Time lost:** ~1 min
**How I found it:** The traceback named the missing module directly.
**Pattern to remember:** TDD step ordering matters — the standard cycle is red → green → refactor. Writing 9 red tests without a module to satisfy them is a half-cycle, not a phase. Future me: when tests are red because the module doesn't exist, the next action is **always** to write the module, not to keep writing more tests.

---

## 2026-09-03 — `mlflow.set_experiment` returns an `Experiment` object, not a string id

**Project:** predicting-electric-vehicle-purchases (Phase 5)
**Error message:** `AssertionError: assert <Experiment: artifact_location='...', experiment_id='746153913683405565', ...> == <Experiment: ...>` — same object, but `==` returned False.

**Root cause:** `mlflow.set_experiment` returns an `Experiment` entity, not a string id. The MLflow `Experiment` class does not implement `__eq__`, so two `Experiment` objects with the same id compare unequal. My `get_or_create_experiment` returned whatever `set_experiment` returned and was annotated as `str`, but it was actually an `Experiment`.

**Fix:** Changed `get_or_create_experiment` to return `experiment.experiment_id` (a string). Tightened the test to assert `isinstance(exp_id, str) and exp_id` so this can't regress.

**Time lost:** ~3 min
**How I found it:** The full diff in the assertion error showed both sides as `<Experiment: ...>` — the same id, but no equality defined.
**Pattern to remember:** Always check return types of library calls against the current docs, not against what they "should" return. MLflow's API has shifted between versions; the docstring said "returns id since 2.0" but in 2.22 it returns the full object.

---

## 2026-09-03 — Test hard-coded experiment id `"0"` instead of looking it up

**Project:** predicting-electric-vehicle-purchases (Phase 5)
**Error message:** `assert 0 == 1 ... where 0 = len([])` — `client.search_runs(experiment_ids=["0"])` returned 0 runs.

**Root cause:** My tests assumed that if you don't pass an experiment name to `start_run`, the run lands in experiment id `"0"` (the default). But `get_or_create_experiment` creates/uses the experiment named `ev-purchase-lgbm`, which gets an auto-generated id (e.g., `746153913683405565`). The default experiment only exists if no experiment is ever explicitly set.

**Fix:** Replaced `experiment_ids=["0"]` with `experiment_ids=[client.get_experiment_by_name("ev-purchase-lgbm").experiment_id]` in 4 tests. This is also the correct usage pattern for callers who don't know the auto-generated id ahead of time.

**Time lost:** ~5 min (after the first test failed, I had to fix 4 more in the same file)
**How I found it:** The captured stderr showed `INFO mlflow.tracking.fluent: Experiment with name 'ev-purchase-lgbm' does not exist. Creating a new experiment.` — i.e. the experiment existed, just not at id `"0"`.
**Pattern to remember:** When using a wrapper that abstracts the tracking URI, look up the experiment by name (`get_experiment_by_name`) rather than guessing at the id. The default-experiment assumption only holds when no experiment has been explicitly created.

---

## 2026-09-03 — `MlflowClient.get_metric_history` signature: `(run_id, key)`, not `(key, run_id=...)`

**Project:** predicting-electric-vehicle-purchases (Phase 5)
**Error message:** `TypeError: MlflowClient.get_metric_history() got multiple values for argument 'run_id'`.

**Root cause:** I wrote `client.get_metric_history("train_loss", run_id=run_id)`. The actual signature is `get_metric_history(run_id, key)`. So Python saw `run_id` passed both positionally (as the first arg) and as a keyword.

**Fix:** Swapped to `client.get_metric_history(run_id, "train_loss")`.

**Time lost:** ~30 sec
**How I found it:** Error message was self-explanatory.
**Pattern to remember:** When the kwarg name and the positional name collide, Python raises this specific `TypeError`. Always check the signature when in doubt — `help(MlflowClient.get_metric_history)` is faster than guessing.

---

## 2026-09-03 — `ValueError: train and valid dataset categorical_feature do not match` (9 tests red in `train_lgbm`)

**Project:** predicting-electric-vehicle-purchases (Phase 6)
**Error message:**
```
ValueError: train and valid dataset categorical_feature do not match.
```
Raised inside `lgb.basic._data_from_pandas` when `booster.predict()` is called on a raw DataFrame (not a `lgb.Dataset`) for the validation fold.

**Root cause:** LightGBM requires the prediction-time DataFrame to declare its categorical columns with the same `category` dtype (and the same category *set*) as the training-time DataFrame. My first version cast columns to `category` inside the training DataFrame, but the per-fold validation slice (and the test DataFrame) had plain `object` dtype, so LightGBM couldn't reconcile the categorical-feature contract.

**Fix:** Added `_build_category_unions()` — pre-compute the union of `unique()` values per categorical column across `train + test` once, up front. Then a `_to_categorical()` helper reindexes every DataFrame (training, per-fold validation, test) to that shared `CategoricalDtype` before it ever reaches LightGBM. This is the canonical pattern for LightGBM native categoricals when the train and test category sets can drift across CV folds.

**Time lost:** ~10 min (one-shot once I traced the error to the right library call)
**How I found it:** Traceback pointed straight at `_data_from_pandas`. The fix is the standard recipe for this error.
**Pattern to remember:** LightGBM's native categorical handling looks like a one-liner (`categorical_feature=[...]`) but it's a contract. Train, val, and test must all use the same `category` dtype, AND the same category set. Pre-compute the union of categories across train + test and reindex every DataFrame to it.

---

## 2026-09-03 — `cv_auc_mean` ≠ pooled `roc_auc_score(y, oof)` (test asserted `1e-6`)

**Project:** predicting-electric-vehicle-purchases (Phase 6)
**Error message:**
```
assert 0.002628074452555529 < 1e-06
abs((0.932130353043491 - 0.9295022785909355))
```

**Root cause:** I compute `cv_auc_mean = mean(fold_aucs)` in the trainer, where each `fold_auc` is `roc_auc_score(y[fold_k], oof[fold_k])` — i.e., AUC computed on a single fold. The test separately computed `roc_auc_score(y, oof)` — i.e., AUC computed on the pooled OOF predictions. These are mathematically different: each fold has its own positive rate and its own model, so the pooled ranking is a different ranking than the per-fold ranking averaged. A 0.002-0.003 difference is normal and expected.

**Fix:** Test was wrong, code was right. Loosened the assertion from `< 1e-6` to `< 0.01` (one percentage point, which accommodates the natural fold-pool gap and small early-stopping variance).

**Time lost:** ~1 min
**How I found it:** The diff (0.0026) was a clean signal — not noise, not a code bug, just a mathematical mismatch I had mis-asserted.
**Pattern to remember:** When a test compares two metrics that are *related but not equal* (e.g., per-fold mean vs pooled), set the tolerance from the data, not from wishful thinking. The right tolerance is the empirical gap (≈ 0.005 here), not zero.

---

## 2026-09-03 — `client.list_artifacts()` only returns root-level entries

**Project:** predicting-electric-vehicle-purchases (Phase 6)
**Error message:** `assert any(p.endswith("oof.npy") for p in artifacts) -> False` even though the file was logged to `artifact_path="oof"`.

**Root cause:** `MlflowClient.list_artifacts(run_id)` returns only the files at the root of the run's artifact directory, not recursively. My `oof.npy` was logged under the subdirectory `oof/`, so `list_artifacts` only returned the directory entry `oof` itself (not `oof/oof.npy`). I needed to call `list_artifacts(run_id, path="oof")` to list files inside that subdir.

**Fix:** Test now queries both root artifacts and the `oof/` subdir, then unions the two sets. Also widened the suffix check to match any path ending in `oof.npy` (whether the path is `oof.npy` at root or `oof/oof.npy` nested).

**Time lost:** ~2 min
**How I found it:** Read the MLflow docs for `list_artifacts` — confirmed the non-recursive default. The fix was two lines.
**Pattern to remember:** `list_artifacts(run_id)` is shallow. To inspect nested artifacts, pass `path="subdir"`. For tests, query both and union the results.

---

## 2026-09-02 — `IntCastingNaNError` in `build_features` on synthetic test data

**Project:** predicting-electric-vehicle-purchases
**Error message:**
```
pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer.
Replace or remove non-finite values or cast to an integer type that supports these values (e.g. 'Int64')
```

**Root cause:** The test `test_build_features_preserves_raw_columns` built a DataFrame with `df[col] = [1] for col in raw_cols` — every value was the integer 1, including `Range_Anxiety_Level = 1`. The features module does `out["Anxiety_ord"] = out["Range_Anxiety_Level"].map({"High": 0, "Medium": 1, "Low": 2}).astype("int64")` — `1` is not in the map, so it returned NaN, and the subsequent `.astype("int64")` raised.

**Fix:**
- Test was wrong, code was right. Rewrote the test fixture to use realistic values (`Range_Anxiety_Level = "Low"`).
- Lesson: when building tiny synthetic test DataFrames, mirror the real dtype AND the real categorical levels. Defaults like `1` for a string-typed column are silently catastrophic.

**Time lost:** ~5 min
**How I found it:** Traceback pointed at `_astype_float_to_int_nansafe`. Recognized the symptom from the map → astype chain.
**Pattern to remember:** `Series.astype("int64")` after a `.map()` is a footgun if the map can miss. Either validate the map result (`assert no NaN`) or use a defensive default.

---

## 2026-09-02 — Module-scoped fixture poisoned by `load_data` schema check

**Project:** predicting-electric-vehicle-purchases
**Error message:**
```
ValueError: test CSV schema mismatch. missing=[], extra=['Will_Buy_EV']
```

**Root cause:** The `train_df` fixture in `test_features.py` called `load_data(TRAIN_CSV, TRAIN_CSV)` to get a single DataFrame — passing the train CSV as both arguments. `load_data` validates that the test CSV does NOT contain the target column; train does. So it errored.

**Fix:** Fixture now reads the CSV directly with `pd.read_csv(TRAIN_CSV)` — no schema check needed since `build_features` doesn't care about the target.

**Time lost:** ~3 min
**How I found it:** Error pointed to the `_validate_schema` call, and the fixture was the only caller doing the weird thing.
**Pattern to remember:** Don't re-use a validating loader for test data. For unit tests, use `pd.read_csv` directly. Reserve the loader for the integration path.

---

## 2026-09-02 — Sanity bound too tight on `Income_per_Age`

**Project:** predicting-electric-vehicle-purchases
**Error message:** `assert 500 < np.float64(434.78...)` — left side of `500 < min` failed.

**Root cause:** I asserted `500 < ipa.min() < ipa.max() < 10_000` based on a guess at the income-to-age distribution. Real data: `Annual_Income_USD` floor is 30,000 and `Age` ceiling is 69, so `Income_per_Age` minimum is `30_000 / 69 = 434.78`.

**Fix:** Loosened lower bound to `400 < ipa.min()`.

**Time lost:** ~1 min
**How I found it:** Trivially from the assertion message.
**Pattern to remember:** Sanity bounds in tests should be derived from the actual data spec (min/max values), not from a hand-wavy guess. Or — better — compute them from the data once and assert they're consistent with the spec.

