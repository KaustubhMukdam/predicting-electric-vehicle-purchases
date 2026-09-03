# Debug Log

> Tracks every non-trivial bug solved. Prevents solving the same bug twice.
> Format follows `dev_system_guide.md` §6.2.

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

