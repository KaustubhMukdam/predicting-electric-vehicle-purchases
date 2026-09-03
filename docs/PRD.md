# PRD — Predicting Electric Vehicle Purchases

## Problem statement
Predict, from a person's demographics, daily commute, income, current car ownership, nearby EV charging infrastructure, and attitudinal signals (environmental concern, range anxiety, openness to subsidy), whether they will purchase an electric vehicle. This is a Kaggle Playground Series competition (synthetic dataset, ~700k train rows, 13 features, 1 binary target). The business interpretation: identify high-intent EV buyers for targeted marketing, incentive allocation, or charging-infrastructure planning.

## Target users
- **Primary:** Me (Kaggle competitor). Building a strong, reproducible baseline pipeline with test coverage and experiment tracking.
- **Secondary:** Anyone reading the repo — the docs should make the approach understandable in 15 minutes without talking to me.

## Core features (MVP)
- [x] Reproducible data loading with deterministic schema validation.
- [x] Minimal feature engineering (ordinal anxiety, charging totals, income-per-age, two interactions).
- [x] Stratified 5-fold CV with a single LightGBM model using native categorical handling.
- [x] MLflow tracking of every run (params, fold AUCs, mean AUC, std, artifacts).
- [x] Submission file generated from a sample_submission template (id column preserved, no leaked columns).
- [x] Pytest suite covering each module + an end-to-end smoke test on a 1% data slice.

## Nice-to-have (post-MVP, gated by separate phases)
- [ ] Optuna hyperparameter sweep (50–100 trials).
- [ ] XGBoost and CatBoost diversity models.
- [ ] Rank-averaged ensemble of all models.
- [ ] Stacking with a logistic meta-learner.
- [ ] TabPFN / TabNet / simple MLP for tabular diversity.
- [ ] Pseudo-labeling on high-confidence test rows.
- [ ] Kaggle notebook (`.ipynb`) that runs the full pipeline and writes `/kaggle/working/submission.csv`.

## Non-goals
- Not optimizing for the absolute top of the leaderboard — that requires heavy compute and is not the learning goal here.
- Not training on the test set in any way that leaks labels (no semi-supervised methods unless explicitly green-lit).
- Not building a model card for production deployment. The `model_card.md` is a learning artifact, not an MLOps deliverable.
- Not building a UI, REST API, or CLI for the model.

## Success metrics
- **Primary:** Mean 5-fold CV ROC-AUC on the training set. Target for v1: ≥ 0.93. (Logistic baseline ~0.86; strong LGBM on this kind of synthetic data historically hits 0.93–0.95.)
- **Secondary:**
  - CV AUC std across folds ≤ 0.005 (model is stable, not just lucky).
  - End-to-end test runs cleanly on a 1% sample in under 60 seconds.
  - Every training run appears in MLflow with full params + metrics + a logged submission artifact.
  - `experiment_log.md` has one row per run with hypothesis, change, result, next experiment.

## Constraints
- **Compute:** Free tier only. Local CPU is sufficient for Tier 1; Kaggle free GPU notebook (30 hr/week) is the recommended runtime.
- **Storage:** Kaggle ephemeral. MLflow runs go to `/kaggle/working/mlruns/` and are zipped as a notebook output.
- **Time budget:** v1 (Tier 1) should be shippable in a focused afternoon. Each subsequent tier is its own phase.
- **Tech:** Python only. No R, no Julia. Dependencies pinned in `requirements.txt` and installed in the Linux `.venv`.
- **Process:** Phase-by-phase execution with TDD per phase. No code without a failing test first. No git operations by the agent.
