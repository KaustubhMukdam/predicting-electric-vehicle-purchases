# Folder Structure — Predicting Electric Vehicle Purchases

```
predicting-electric-vehicle-purchases/
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
│
├── data/                              # gitignored — Kaggle-provided CSVs
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── docs/                              # all project documentation
│   ├── project_context.md             # paste at top of every AI chat
│   ├── PRD.md
│   ├── tech_stack.md
│   ├── architecture.md
│   ├── folder_structure.md            # this file
│   ├── tasks.md                       # atomic task list
│   ├── data_doc.md                    # dataset schema + EDA
│   ├── experiment_log.md              # one row per training run
│   ├── model_card.md                  # final model documentation
│   ├── eval.md                        # evaluation methodology
│   ├── learnings.md                   # personal learnings (per dev guide §6.1)
│   └── debug_log.md                   # bugs hit and how they were fixed (§6.2)
│
├── src/                               # all production code
│   ├── __init__.py
│   ├── config.py                      # paths, target map, categorical lists, constants
│   ├── utils.py                       # seed_everything, small helpers
│   ├── data.py                        # load_data(train_path, test_path) -> (df, df)
│   ├── features.py                    # build_features(df) -> df
│   ├── cv.py                          # make_folds(y, n_splits, seed) -> np.ndarray
│   ├── tracking.py                    # mlflow wrapper: start_run, log_params, log_metrics, log_artifact
│   ├── train_lgbm.py                  # train_lgbm(...) -> oof, test_pred, cv_metrics
│   └── predict.py                     # make_submission(...) -> writes submission.csv
│
├── tests/                             # pytest, one file per src/ module + e2e
│   ├── __init__.py
│   ├── conftest.py                    # shared fixtures (paths, data loaders)
│   ├── test_smoke.py                  # all modules importable
│   ├── test_config.py
│   ├── test_utils.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_cv.py
│   ├── test_tracking.py
│   ├── test_train_lgbm.py
│   ├── test_predict.py
│   └── test_pipeline_e2e.py           # runs the whole pipeline on a 1% sample
│
├── notebooks/                         # Jupyter notebooks
│   ├── .gitkeep
│   ├── EDA.ipynb                      # exploratory data analysis
│   └── train.ipynb                    # end-to-end pipeline runner (Kaggle-ready)
│
├── oof/                               # gitignored — out-of-fold predictions (.npy)
│   └── .gitkeep
│
├── submissions/                       # gitignored — final .csv files
│   └── .gitkeep
│
└── mlruns/                            # gitignored — MLflow local tracking dir (if used locally)
    └── .gitkeep
```

## Naming conventions
- **Python files:** `snake_case` (`train_lgbm.py`, `make_folds`).
- **Python functions/variables:** `snake_case`.
- **Python classes:** `PascalCase` (used sparingly — only if a class earns its keep).
- **Constants:** `UPPER_SNAKE_CASE` in `config.py`.
- **Test files:** mirror the module name (`test_<module>.py`).
- **Test functions:** `test_<thing>_<expected_behavior>` (`test_load_data_returns_dataframe_with_target_mapped`).
- **Doc files:** `snake_case.md` matching their purpose.
- **Notebooks:** `PascalCase.ipynb` (EDA.ipynb, Train.ipynb).

## Why this structure
- **`src/` is importable from anywhere** because we add the project root to `sys.path` in the test conftest and in the notebooks. No need for a `setup.py` for a single-project repo.
- **One function per module, no premature `__init__` exports.** Each `src/*.py` exposes one main public function (`load_data`, `build_features`, etc.) that the notebook wires together. This forces a clean dependency graph and makes tests trivial.
- **`oof/`, `submissions/`, `mlruns/` are gitignored from day one** so generated artifacts never accidentally get committed.
- **Tests mirror `src/` exactly** so a reader can find the test for any function in one hop.
- **`docs/` is a flat directory, not nested.** Eleven files at the top level is the threshold where nesting starts to help — we have 12, but the entries are independent enough that flat is still easier to scan.
