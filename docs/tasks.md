# Tasks — Predicting Electric Vehicle Purchases

> Atomic task list. One task = one thing finishable in a single session. Replaces Notion/Jira for this solo project.

## In progress
- [ ] Phase 11 — first real Kaggle run + populate `model_card.md`

## Up next (in order)
- [ ] Phase 12 — Optuna hyperparameter sweep (gated)
- [ ] Phase 13 — XGBoost / CatBoost diversity models (gated)
- [ ] Phase 14 — Rank-averaged ensemble (gated)

## Done
- [x] Phase 0 — project skeleton
- [x] Phase 1 — `config.py` + `utils.py`
- [x] Phase 2 — `data.py`
- [x] Phase 3 — `features.py`
- [x] Phase 4 — `cv.py`
- [x] Phase 5 — `tracking.py`
- [x] Phase 6 — `train_lgbm.py`
- [x] Phase 7 — `predict.py`
- [x] Phase 8 — `test_pipeline_e2e.py` (8/8 tests green, 93/93 across the project)
- [x] Phase 9 — `notebooks/EDA.ipynb` (8 code cells, 0 syntax errors)
- [x] Phase 10 — `notebooks/train.ipynb` (7 code cells, 0 syntax errors)
- [x] All 12 docs in `docs/` updated through Phase 10
- [x] `experiment_log.md` has Experiments 0, 1, 2, 3

## Backlog (post-MVP, each its own phase)
- [ ] Optuna sweep (50–100 trials) on LGBM hyperparameters
- [ ] XGBoost model script + comparison
- [ ] CatBoost model script + comparison
- [ ] Rank-averaged ensemble of LGBM + XGB + CatBoost
- [ ] Stacking with logistic meta-learner
- [ ] TabPFN / simple MLP for tabular diversity
- [ ] Pseudo-labeling on high-confidence test rows

## Done
- [x] Dataset EDA (plan mode): rows, columns, missing, duplicates, class balance, per-feature signal
- [x] Locked stack: pandas, numpy, scikit-learn, LightGBM, MLflow, pytest
- [x] Locked phasing: Tier 1 single LGBM, minimal FE, MLflow file backend, Kaggle runtime
- [x] Locked process: phase-by-phase, TDD, no git ops by agent, Linux `.venv`
- [x] All 10 ML docs written in `docs/`

## Blocked
- (none)

## Ideas / backlog
- Add a `Makefile` or `justfile` with `make test`, `make train`, `make submit` targets
- Cache the OOF predictions on disk so a failed notebook run doesn't re-train
- Per-segment AUC plots in the EDA notebook
