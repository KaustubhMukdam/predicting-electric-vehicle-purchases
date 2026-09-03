# Tech Stack — Predicting Electric Vehicle Purchases

## Language
| Technology | Version | Why chosen |
|------------|---------|------------|
| Python | 3.11 | Best balance of library support and stdlib features; matches Kaggle default kernel. |

## Data
| Library | Version | Why chosen |
|---------|---------|------------|
| pandas | 2.2.x | Standard for tabular data. `pd.read_csv`, `DataFrame.merge`, `groupby` cover everything. |
| numpy | 1.26.x | Underlies pandas; needed for array ops in feature engineering. |

## Machine learning
| Library | Version | Why chosen |
|---------|---------|------------|
| scikit-learn | 1.4.x | Stratified KFold, `roc_auc_score`, `train_test_split` — all we need for v1 CV plumbing. |
| LightGBM | 4.3.x | Fastest GBM on tabular data with native categorical support (avoids one-hot blowup). Histogram-based, low memory, handles 700k rows on CPU in seconds. |

## Experiment tracking
| Library | Version | Why chosen |
|---------|---------|------------|
| MLflow | 2.13.x | File-based backend — no server, no external account. Same API locally and on Kaggle. Tracks params, metrics, artifacts. UI via `mlflow ui`. |

## Hyperparameter tuning (later)
| Library | Version | Why chosen |
|---------|---------|------------|
| Optuna | 3.6.x | Modern, lightweight, TPE sampler by default. Defer to a separate phase. |

## Testing
| Library | Version | Why chosen |
|---------|---------|------------|
| pytest | 8.x | Simple, fast, great fixture system, ubiquitous in Python data projects. |

## Environment
| Tool | Why chosen |
|------|------------|
| Linux `.venv` | Matches the user's environment. Keeps system Python clean. |

## Why NOT chosen
- **XGBoost / CatBoost in v1:** Adds two more model scripts to maintain before we've proven the pipeline works. Deferred to a separate phase.
- **PyTorch / TabNet / TabPFN in v1:** Overkill for a 13-feature tabular problem with a single LGBM expected to land ~0.94 AUC.
- **Weights & Biases / Neptune:** External accounts, network dependency. MLflow file backend is local-only.
- **Hydra / OmegaConf config:** YAGNI for a single-model pipeline. A plain `config.py` with constants is enough.
- **DVC:** Data is small (~50 MB) and versioned only via the user pushing to git. DVC adds infrastructure we don't need.

## Known tradeoffs
- **LightGBM native categoricals vs one-hot:** Native handling often matches or beats one-hot on tree models, but requires careful category-set alignment between train and test. The data loader will validate this in Phase 2.
- **MLflow file backend:** Runs don't sync anywhere. On Kaggle, the `mlruns/` directory must be zipped and downloaded at the end of the session, or the history is lost. This is acceptable for a single-user competition project but would be a problem in a team setting.
- **No pinned exact versions in `requirements.txt` initially:** We'll use compatible release specifiers (`~=`) and let `pip install` resolve. We can pin exact versions once the pipeline stabilizes.
