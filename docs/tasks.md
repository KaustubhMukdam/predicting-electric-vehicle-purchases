# Tasks — Predicting Electric Vehicle Purchases

> Atomic task list. One task = one thing finishable in a single session. Replaces Notion/Jira for this solo project.

## In progress
- [ ] Phase 0 — project skeleton: `requirements.txt`, `pytest.ini`, `conftest.py`, `.gitignore` updates, empty `src/` and `tests/` placeholders

## Up next (in order)
- [ ] Phase 1 — `config.py` + `utils.py` + their tests (TDD)
- [ ] Phase 2 — `data.py` + test (TDD): schema validation, target mapping, dtype coercion
- [ ] Phase 3 — `features.py` + test (TDD): 6 engineered features
- [ ] Phase 4 — `cv.py` + test (TDD): stratified KFold
- [ ] Phase 5 — `tracking.py` + test (TDD): MLflow wrapper
- [ ] Phase 6 — `train_lgbm.py` + test (TDD): 5-fold trainer with OOF + test preds
- [ ] Phase 7 — `predict.py` + test (TDD): submission file from template
- [ ] Phase 8 — `test_pipeline_e2e.py`: smoke-test the whole pipeline on 1% data
- [ ] Phase 9 — `notebooks/EDA.ipynb`: reproduce the EDA from plan mode
- [ ] Phase 10 — `notebooks/train.ipynb`: Kaggle-ready end-to-end runner
- [ ] Phase 11 — first real Kaggle run + populate `model_card.md` + first `experiment_log.md` row

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
