"""Tests for src/features.py.

These use the real train data so we catch real-data quirks (e.g.,
income = 30_000 is a hard floor, age starts at 25).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ENGINEERED_COLS, TRAIN_CSV


pytestmark = pytest.mark.skipif(
    not TRAIN_CSV.exists(),
    reason="data/train.csv not present",
)


@pytest.fixture(scope="module")
def train_df():
    import pandas as pd

    # Load the train CSV directly — we don't need the loader's validation
    # here, we just need the data the same way the real pipeline would.
    return pd.read_csv(TRAIN_CSV)


def test_build_features_preserves_row_count():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [25, 50, 70],
            "Annual_Income_USD": [30000.0, 80000.0, 200000.0],
            "Daily_Commute_km": [5.0, 30.0, 100.0],
            "Number_of_Cars_Owned": [1, 2, 4],
            "Charging_Stations_Near_Home": [0, 5, 14],
            "Charging_Stations_Near_Work": [0, 10, 19],
            "Environmental_Concern_Level": [1, 3, 5],
            "Gender": ["Male", "Female", "Other"],
            "City_Type": ["Urban", "Suburban", "Rural"],
            "Current_Car_Type": ["Sedan", "SUV", "Truck"],
            "Home_Charging_Possible": ["Yes", "No", "Yes"],
            "Subsidy_Available": ["Yes", "No", "No"],
            "Range_Anxiety_Level": ["High", "Medium", "Low"],
        }
    )
    out = build_features(df)
    assert len(out) == 3


def test_build_features_adds_all_engineered_columns():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [25, 50, 70],
            "Annual_Income_USD": [30000.0, 80000.0, 200000.0],
            "Daily_Commute_km": [5.0, 30.0, 100.0],
            "Number_of_Cars_Owned": [1, 2, 4],
            "Charging_Stations_Near_Home": [0, 5, 14],
            "Charging_Stations_Near_Work": [0, 10, 19],
            "Environmental_Concern_Level": [1, 3, 5],
            "Gender": ["Male", "Female", "Other"],
            "City_Type": ["Urban", "Suburban", "Rural"],
            "Current_Car_Type": ["Sedan", "SUV", "Truck"],
            "Home_Charging_Possible": ["Yes", "No", "Yes"],
            "Subsidy_Available": ["Yes", "No", "No"],
            "Range_Anxiety_Level": ["High", "Medium", "Low"],
        }
    )
    out = build_features(df)
    for col in ENGINEERED_COLS:
        assert col in out.columns, f"{col} missing"


def test_anxiety_ord_mapping_is_correct():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [30, 40, 50],
            "Annual_Income_USD": [50000.0, 50000.0, 50000.0],
            "Daily_Commute_km": [10.0, 10.0, 10.0],
            "Number_of_Cars_Owned": [1, 1, 1],
            "Charging_Stations_Near_Home": [0, 0, 0],
            "Charging_Stations_Near_Work": [0, 0, 0],
            "Environmental_Concern_Level": [3, 3, 3],
            "Gender": ["Male", "Male", "Male"],
            "City_Type": ["Urban", "Urban", "Urban"],
            "Current_Car_Type": ["Sedan", "Sedan", "Sedan"],
            "Home_Charging_Possible": ["Yes", "Yes", "Yes"],
            "Subsidy_Available": ["No", "No", "No"],
            "Range_Anxiety_Level": ["High", "Medium", "Low"],
        }
    )
    out = build_features(df)
    assert out["Anxiety_ord"].tolist() == [0, 1, 2]


def test_stations_total_is_sum():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [30],
            "Annual_Income_USD": [50000.0],
            "Daily_Commute_km": [10.0],
            "Number_of_Cars_Owned": [1],
            "Charging_Stations_Near_Home": [3],
            "Charging_Stations_Near_Work": [7],
            "Environmental_Concern_Level": [3],
            "Gender": ["Male"],
            "City_Type": ["Urban"],
            "Current_Car_Type": ["Sedan"],
            "Home_Charging_Possible": ["Yes"],
            "Subsidy_Available": ["No"],
            "Range_Anxiety_Level": ["Low"],
        }
    )
    out = build_features(df)
    assert out["Stations_Total"].iloc[0] == 10


def test_income_per_age_is_ratio():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [25, 50],
            "Annual_Income_USD": [50000.0, 100000.0],
            "Daily_Commute_km": [10.0, 10.0],
            "Number_of_Cars_Owned": [1, 1],
            "Charging_Stations_Near_Home": [0, 0],
            "Charging_Stations_Near_Work": [0, 0],
            "Environmental_Concern_Level": [3, 3],
            "Gender": ["Male", "Male"],
            "City_Type": ["Urban", "Urban"],
            "Current_Car_Type": ["Sedan", "Sedan"],
            "Home_Charging_Possible": ["Yes", "Yes"],
            "Subsidy_Available": ["No", "No"],
            "Range_Anxiety_Level": ["Low", "Low"],
        }
    )
    out = build_features(df)
    assert out["Income_per_Age"].iloc[0] == pytest.approx(2000.0)
    assert out["Income_per_Age"].iloc[1] == pytest.approx(2000.0)


def test_subsidy_interactions_are_zero_when_no_subsidy():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [30, 40],
            "Annual_Income_USD": [50000.0, 90000.0],
            "Daily_Commute_km": [10.0, 10.0],
            "Number_of_Cars_Owned": [1, 1],
            "Charging_Stations_Near_Home": [2, 2],
            "Charging_Stations_Near_Work": [4, 4],
            "Environmental_Concern_Level": [3, 3],
            "Gender": ["Male", "Female"],
            "City_Type": ["Urban", "Rural"],
            "Current_Car_Type": ["Sedan", "SUV"],
            "Home_Charging_Possible": ["Yes", "No"],
            "Subsidy_Available": ["No", "No"],
            "Range_Anxiety_Level": ["Low", "High"],
        }
    )
    out = build_features(df)
    assert (out["Env_x_Subsidy"] == 0).all()
    assert (out["Subsidy_x_Income"] == 0).all()


def test_subsidy_interactions_use_raw_value_when_subsidy_yes():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [30, 40],
            "Annual_Income_USD": [50000.0, 90000.0],
            "Daily_Commute_km": [10.0, 10.0],
            "Number_of_Cars_Owned": [1, 1],
            "Charging_Stations_Near_Home": [2, 2],
            "Charging_Stations_Near_Work": [4, 4],
            "Environmental_Concern_Level": [3, 4],
            "Gender": ["Male", "Female"],
            "City_Type": ["Urban", "Rural"],
            "Current_Car_Type": ["Sedan", "SUV"],
            "Home_Charging_Possible": ["Yes", "No"],
            "Subsidy_Available": ["Yes", "Yes"],
            "Range_Anxiety_Level": ["Low", "High"],
        }
    )
    out = build_features(df)
    assert out["Env_x_Subsidy"].iloc[0] == 3
    assert out["Env_x_Subsidy"].iloc[1] == 4
    assert out["Subsidy_x_Income"].iloc[0] == pytest.approx(50000.0)
    assert out["Subsidy_x_Income"].iloc[1] == pytest.approx(90000.0)


def test_anxiety_x_stations_total_uses_ordinal():
    from src.features import build_features

    df = pd.DataFrame(
        {
            "Age": [30, 40, 50],
            "Annual_Income_USD": [50000.0, 50000.0, 50000.0],
            "Daily_Commute_km": [10.0, 10.0, 10.0],
            "Number_of_Cars_Owned": [1, 1, 1],
            "Charging_Stations_Near_Home": [2, 2, 2],
            "Charging_Stations_Near_Work": [3, 3, 3],
            "Environmental_Concern_Level": [3, 3, 3],
            "Gender": ["Male", "Male", "Male"],
            "City_Type": ["Urban", "Urban", "Urban"],
            "Current_Car_Type": ["Sedan", "Sedan", "Sedan"],
            "Home_Charging_Possible": ["Yes", "Yes", "Yes"],
            "Subsidy_Available": ["No", "No", "No"],
            "Range_Anxiety_Level": ["High", "Medium", "Low"],
        }
    )
    out = build_features(df)
    # Stations_Total = 5 for all rows; Anxiety_ord = 0, 1, 2
    assert out["Anxiety_x_Stations_Total"].tolist() == [0, 5, 10]


def test_build_features_preserves_raw_columns():
    from src.features import build_features

    raw_cols = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]
    df = pd.DataFrame(
        {
            "Age": [30],
            "Annual_Income_USD": [50000.0],
            "Daily_Commute_km": [10.0],
            "Number_of_Cars_Owned": [1],
            "Charging_Stations_Near_Home": [0],
            "Charging_Stations_Near_Work": [0],
            "Environmental_Concern_Level": [3],
            "Gender": ["Male"],
            "City_Type": ["Urban"],
            "Current_Car_Type": ["Sedan"],
            "Home_Charging_Possible": ["Yes"],
            "Subsidy_Available": ["No"],
            "Range_Anxiety_Level": ["Low"],
        }
    )
    out = build_features(df)
    for col in raw_cols:
        assert col in out.columns


def test_build_features_does_not_introduce_missing(train_df):
    from src.features import build_features

    out = build_features(train_df)
    assert out.isnull().sum().sum() == 0


def test_build_features_is_idempotent(train_df):
    from src.features import build_features

    a = build_features(train_df)
    b = build_features(train_df)
    pd.testing.assert_frame_equal(a, b)


def test_build_features_on_real_data_engineered_stats(train_df):
    """Sanity check on the real training data: the engineered columns
    should have plausible distributions (no inf, no absurd values)."""
    from src.features import build_features

    out = build_features(train_df)
    assert np.isfinite(out[ENGINEERED_COLS].to_numpy()).all()
    # Income_per_Age should be in the low thousands (~$430-$7.5k).
    ipa = out["Income_per_Age"]
    assert 400 < ipa.min() < ipa.max() < 10_000
