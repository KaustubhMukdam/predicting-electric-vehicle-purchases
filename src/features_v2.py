"""V2 feature engineering for EV purchase prediction.

V2 is based on the interaction EDA performed in 01_interaction_eda.ipynb.

Design goals:
- Preserve all original columns.
- Do not mutate the input DataFrame.
- Keep V1 features for direct comparison.
- Add only high-value interaction candidates.
- Same input -> same output.
"""

from __future__ import annotations

import pandas as pd


ANXIETY_ORD_MAP: dict[str, int] = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build V2 features."""

    out = df.copy()

    # ============================================================
    # V1 FEATURES
    # ============================================================

    # Range anxiety ordinal
    out["Anxiety_ord"] = (
        out["Range_Anxiety_Level"]
        .map(ANXIETY_ORD_MAP)
        .astype("int64")
    )

    # Charging station totals
    out["Stations_Total"] = (
        out["Charging_Stations_Near_Home"]
        + out["Charging_Stations_Near_Work"]
    ).astype("int64")

    out["Stations_Difference"] = (
        out["Charging_Stations_Near_Home"]
        - out["Charging_Stations_Near_Work"]
    ).astype("int64")

    # Ratios
    out["Income_per_Age"] = (
        out["Annual_Income_USD"] / out["Age"]
    )

    out["Income_per_Commute"] = (
        out["Annual_Income_USD"]
        / out["Daily_Commute_km"]
    )

    # ============================================================
    # BINARY FLAGS
    # ============================================================

    subsidy_flag = (
        out["Subsidy_Available"] == "Yes"
    ).astype("int8")

    home_charge_flag = (
        out["Home_Charging_Possible"] == "Yes"
    ).astype("int8")

    # ============================================================
    # NUMERICAL × BINARY
    # ============================================================

    out["Environmental_x_Subsidy"] = (
        out["Environmental_Concern_Level"]
        * subsidy_flag
    )

    out["Income_x_Subsidy"] = (
        out["Annual_Income_USD"]
        * subsidy_flag
    )

    out["Environmental_x_HomeCharging"] = (
        out["Environmental_Concern_Level"]
        * home_charge_flag
    )

    out["Income_x_HomeCharging"] = (
        out["Annual_Income_USD"]
        * home_charge_flag
    )

    # ============================================================
    # NUMERICAL × ORDINAL
    # ============================================================

    out["Environmental_x_Anxiety"] = (
        out["Environmental_Concern_Level"]
        * out["Anxiety_ord"]
    )

    out["Income_x_Anxiety"] = (
        out["Annual_Income_USD"]
        * out["Anxiety_ord"]
    )

    # ============================================================
    # NUMERICAL × NUMERICAL
    # ============================================================

    out["Income_x_Environmental"] = (
        out["Annual_Income_USD"]
        * out["Environmental_Concern_Level"]
    )

    # ============================================================
    # CATEGORICAL × CATEGORICAL
    # ============================================================

    out["Subsidy_x_Anxiety"] = (
        out["Subsidy_Available"].astype(str)
        + "_"
        + out["Range_Anxiety_Level"].astype(str)
    )

    out["Subsidy_x_HomeCharging"] = (
        out["Subsidy_Available"].astype(str)
        + "_"
        + out["Home_Charging_Possible"].astype(str)
    )

    out["Subsidy_x_City"] = (
        out["Subsidy_Available"].astype(str)
        + "_"
        + out["City_Type"].astype(str)
    )

    out["Subsidy_x_CarType"] = (
        out["Subsidy_Available"].astype(str)
        + "_"
        + out["Current_Car_Type"].astype(str)
    )

    # ============================================================
    # SECOND-TIER ENVIRONMENTAL INTERACTIONS
    # ============================================================

    out["Environmental_x_City"] = (
        out["Environmental_Concern_Level"].astype(str)
        + "_"
        + out["City_Type"].astype(str)
    )

    return out