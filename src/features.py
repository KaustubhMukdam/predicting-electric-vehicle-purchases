"""Feature engineering.

A pure function: same input -> same output, no fit state, no mutation
of the input DataFrame. The function returns a *new* DataFrame with
all original columns preserved and the engineered columns appended.
"""

from __future__ import annotations

import pandas as pd

# Ordinal mapping for Range_Anxiety_Level. Higher value = lower anxiety.
# Captures the natural ordering: High < Medium < Low.
ANXIETY_ORD_MAP: dict[str, int] = {"High": 0, "Medium": 1, "Low": 2}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append engineered features to a copy of `df`.

    Engineered columns:
        - Anxiety_ord:             {High: 0, Medium: 1, Low: 2}
        - Stations_Total:          Home + Work charging stations
        - Income_per_Age:          Annual_Income_USD / Age
        - Env_x_Subsidy:           Environmental_Concern_Level * 1{Subsidy_Available == 'Yes'}
        - Subsidy_x_Income:        Annual_Income_USD * 1{Subsidy_Available == 'Yes'}
        - Anxiety_x_Stations_Total: Anxiety_ord * Stations_Total
    """
    out = df.copy()

    out["Anxiety_ord"] = out["Range_Anxiety_Level"].map(ANXIETY_ORD_MAP).astype("int64")

    out["Stations_Total"] = (
        out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    ).astype("int64")

    out["Income_per_Age"] = out["Annual_Income_USD"] / out["Age"]

    subsidy_flag = (out["Subsidy_Available"] == "Yes").astype("int64")
    out["Env_x_Subsidy"] = out["Environmental_Concern_Level"] * subsidy_flag
    out["Subsidy_x_Income"] = out["Annual_Income_USD"] * subsidy_flag

    out["Anxiety_x_Stations_Total"] = out["Anxiety_ord"] * out["Stations_Total"]

    return out
