"""
E2 — Rank Blending
LightGBM V2: 92.8%
XGBoost V3:    7.2%

Rank blending converts each model's test predictions to their relative
percentile/rank before combining them. This is useful for ROC-AUC because
AUC depends on ordering rather than probability calibration.

Creates a Kaggle-ready submission CSV.

Expected files:
- test_pred_lgbm_v2.npy
- test_pred_xgb_v3.npy

Fallback:
- submission_xgb_v3.csv can be used if test_pred_xgb_v3.npy is unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path

LGBM_WEIGHT = 0.928
XGB_WEIGHT = 0.072

# Change these paths if necessary.
LGBM_TEST_PATH = Path("test_pred_lgbm_v2.npy")
XGB_TEST_PATH = Path("test_pred_xgb_v3.npy")
XGB_SUBMISSION_PATH = Path("submission_xgb_v3.csv")

OUTPUT_PATH = Path(
    "submission_e2_rank_lgbm_92.8_xgb_7.2.csv"
)


def percentile_rank(values):
    """
    Convert predictions into percentile ranks.

    pct=True gives values approximately in (0, 1], preserving the ordering.
    Average ranking is used for ties.
    """
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


# -------------------------
# Load LightGBM predictions
# -------------------------
if not LGBM_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Could not find LightGBM test predictions: {LGBM_TEST_PATH}"
    )

lgbm_pred = np.load(LGBM_TEST_PATH)

# -------------------------
# Load XGBoost predictions
# -------------------------
if XGB_TEST_PATH.exists():
    xgb_pred = np.load(XGB_TEST_PATH)
    print(f"Loaded XGB predictions from: {XGB_TEST_PATH}")
else:
    if not XGB_SUBMISSION_PATH.exists():
        raise FileNotFoundError(
            "Neither test_pred_xgb_v3.npy nor submission_xgb_v3.csv was found."
        )

    xgb_sub = pd.read_csv(XGB_SUBMISSION_PATH)

    if "Will_Buy_EV" not in xgb_sub.columns:
        raise ValueError(
            f"'Will_Buy_EV' column not found in {XGB_SUBMISSION_PATH}. "
            f"Columns found: {list(xgb_sub.columns)}"
        )

    xgb_pred = xgb_sub["Will_Buy_EV"].to_numpy()
    print(f"Loaded XGB predictions from: {XGB_SUBMISSION_PATH}")

# -------------------------
# Validate predictions
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
# Convert predictions to ranks
# -------------------------
lgbm_rank = percentile_rank(lgbm_pred)
xgb_rank = percentile_rank(xgb_pred)

# -------------------------
# Rank blending
# -------------------------
blend_pred = (
    LGBM_WEIGHT * lgbm_rank
    + XGB_WEIGHT * xgb_rank
)

# -------------------------
# Build submission
# -------------------------
# XGB V3 submission supplies the correct test IDs/order.
if not XGB_SUBMISSION_PATH.exists():
    raise FileNotFoundError(
        f"Need {XGB_SUBMISSION_PATH} as the submission ID template."
    )

xgb_sub = pd.read_csv(XGB_SUBMISSION_PATH)

if len(xgb_sub) != len(blend_pred):
    raise ValueError(
        f"Submission length mismatch: "
        f"submission={len(xgb_sub)}, predictions={len(blend_pred)}"
    )

submission = xgb_sub.copy()
submission["Will_Buy_EV"] = blend_pred

submission.to_csv(OUTPUT_PATH, index=False)

print("\nE2 Rank Blend")
print("-------------")
print(f"LightGBM weight : {LGBM_WEIGHT:.1%}")
print(f"XGBoost weight  : {XGB_WEIGHT:.1%}")
print(f"Rows            : {len(submission):,}")
print(f"Rank blend mean : {blend_pred.mean():.8f}")
print(f"Rank blend min  : {blend_pred.min():.8f}")
print(f"Rank blend max  : {blend_pred.max():.8f}")
print(f"\nSaved submission: {OUTPUT_PATH}")
