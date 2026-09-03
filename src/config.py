"""Project-wide constants and configuration.

This module is intentionally side-effect free: importing it must not
read files, set MLflow URIs, or do any I/O. The only exception is
path resolution, which uses pathlib and does not touch the filesystem.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Default data location: local repo layout.
# On Kaggle, src/data.py will detect /kaggle/input and override these.
DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_CSV: Path = DATA_DIR / "train.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV: Path = DATA_DIR / "sample_submission.csv"

# Output locations (gitignored).
OOF_DIR: Path = PROJECT_ROOT / "oof"
SUBMISSIONS_DIR: Path = PROJECT_ROOT / "submissions"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TARGET_COL: str = "Will_Buy_EV"
ID_COL: str = "id"

# Target value mapping. Yes -> 1, No -> 0. The loader reverses this if
# the CSV happens to use 0/1 already.
TARGET_MAP: dict[str, int] = {"No": 0, "Yes": 1}
TARGET_INVERSE_MAP: dict[int, str] = {v: k for k, v in TARGET_MAP.items()}

# Raw numeric columns.
NUMERIC_COLS: list[str] = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

# Raw categorical columns (string dtype in the CSV).
CATEGORICAL_COLS: list[str] = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

# Engineered feature names (defined here so features.py and tests can agree).
ENGINEERED_COLS: list[str] = [
    "Anxiety_ord",
    "Stations_Total",
    "Income_per_Age",
    "Env_x_Subsidy",
    "Subsidy_x_Income",
    "Anxiety_x_Stations_Total",
]

# All feature columns after build_features() — used by the trainer and tests.
FEATURE_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_COLS

# ---------------------------------------------------------------------------
# Modeling defaults
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_FOLDS: int = 5

# MLflow experiment name.
MLFLOW_EXPERIMENT_NAME: str = "ev-purchase-lgbm"
