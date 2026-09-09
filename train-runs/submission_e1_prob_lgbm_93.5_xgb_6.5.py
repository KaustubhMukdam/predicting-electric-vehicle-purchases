"""
E1 — Probability Blend
LightGBM V2: 93.5%
XGBoost V3:   6.5%

Creates a Kaggle-ready submission CSV.

Expected files:
- test_pred_lgbm_v2.npy
- submission_xgb_v3.csv

If your XGBoost V3 test predictions are stored as test_pred_xgb_v3.npy,
the script will use that instead.
"""

import numpy as np
import pandas as pd
from pathlib import Path

LGBM_WEIGHT = 0.935
XGB_WEIGHT = 0.065

# Change these paths if necessary.
LGBM_TEST_PATH = Path("test_pred_lgbm_v2.npy")
XGB_TEST_PATH = Path("test_pred_xgb_v3.npy")
XGB_SUBMISSION_PATH = Path("submission_xgb_v3.csv")

OUTPUT_PATH = Path(
    "submission_e1_prob_lgbm_93.5_xgb_6.5.csv"
)

# -------------------------
# Load LightGBM predictions
# -------------------------
lgbm_pred = np.load(LGBM_TEST_PATH)

# -------------------------
# Load XGBoost predictions
# -------------------------
if XGB_TEST_PATH.exists():
    xgb_pred = np.load(XGB_TEST_PATH)
    print(f"Loaded XGB predictions from: {XGB_TEST_PATH}")
else:
    # Fall back to the already-created XGB V3 submission.
    xgb_sub = pd.read_csv(XGB_SUBMISSION_PATH)

    if "Will_Buy_EV" not in xgb_sub.columns:
        raise ValueError(
            f"'Will_Buy_EV' column not found in {XGB_SUBMISSION_PATH}. "
            f"Columns found: {list(xgb_sub.columns)}"
        )

    xgb_pred = xgb_sub["Will_Buy_EV"].to_numpy()
    print(f"Loaded XGB predictions from submission: {XGB_SUBMISSION_PATH}")

# -------------------------
# Validate dimensions
# -------------------------
if len(lgbm_pred) != len(xgb_pred):
    raise ValueError(
        f"Prediction length mismatch: "
        f"LGBM={len(lgbm_pred)}, XGB={len(xgb_pred)}"
    )

if not np.isfinite(lgbm_pred).all():
    raise ValueError("LightGBM predictions contain NaN/Inf values.")

if not np.isfinite(xgb_pred).all():
    raise ValueError("XGBoost predictions contain NaN/Inf values.")

# -------------------------
# Probability blending
# -------------------------
blend_pred = (
    LGBM_WEIGHT * lgbm_pred
    + XGB_WEIGHT * xgb_pred
)

# -------------------------
# Build submission
# -------------------------
# Use XGB V3 submission as the ID template because it already has
# the correct Kaggle test-row ordering.
xgb_sub = pd.read_csv(XGB_SUBMISSION_PATH)

if len(xgb_sub) != len(blend_pred):
    raise ValueError(
        f"Submission length mismatch: "
        f"submission={len(xgb_sub)}, predictions={len(blend_pred)}"
    )

submission = xgb_sub.copy()
submission["Will_Buy_EV"] = blend_pred

submission.to_csv(OUTPUT_PATH, index=False)

print("\nE1 Probability Blend")
print("--------------------")
print(f"LightGBM weight : {LGBM_WEIGHT:.1%}")
print(f"XGBoost weight  : {XGB_WEIGHT:.1%}")
print(f"Rows            : {len(submission):,}")
print(f"Prediction mean : {blend_pred.mean():.8f}")
print(f"Prediction min  : {blend_pred.min():.8f}")
print(f"Prediction max  : {blend_pred.max():.8f}")
print(f"\nSaved submission: {OUTPUT_PATH}")
