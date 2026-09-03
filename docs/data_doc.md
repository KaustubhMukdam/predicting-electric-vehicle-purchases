# Data Documentation — Predicting Electric Vehicle Purchases

## Source
- **Origin:** Kaggle Playground Series Season 6 Episode 9 — synthetic dataset, generated from a real survey-style dataset on EV purchase intent.
- **Link:** https://www.kaggle.com/competitions/playground-series-s6e9/data
- **License:** Competition rules — use allowed for the competition only.
- **Downloaded:** 2026-09-02

## Structure
- **Train rows:** 668,665
- **Test rows:** 286,571
- **Columns (train):** 15 (`id` + 13 features + 1 target)
- **Columns (test):** 14 (`id` + 13 features)
- **Target column:** `Will_Buy_EV` (Yes / No) — mapped to {0, 1} in `src/data.py`.

### Schema
| Column | Dtype | Cardinality / Range | Role |
|---|---|---|---|
| `id` | int64 | 0 to 668,664 (train); 668,665+ (test) | Identifier — not a feature |
| `Age` | int64 | 25 to 69 | Numeric feature |
| `Annual_Income_USD` | float64 | 30,000 to 188,549 (13,214 unique) | Numeric feature |
| `Daily_Commute_km` | float64 | 5.0 to 98.7 (805 unique) | Numeric feature |
| `Number_of_Cars_Owned` | int64 | {1, 2, 3, 4} | Numeric (low-cardinality) feature |
| `Charging_Stations_Near_Home` | int64 | 0 to 14 | Numeric feature |
| `Charging_Stations_Near_Work` | int64 | 0 to 19 | Numeric feature |
| `Environmental_Concern_Level` | int64 (1–5) | 1 to 5 | Numeric (ordinal) feature |
| `Gender` | str | {Female, Male, Other} | Categorical feature |
| `City_Type` | str | {Rural, Suburban, Urban} | Categorical feature |
| `Current_Car_Type` | str | {Hatchback, Sedan, SUV, Truck} | Categorical feature |
| `Home_Charging_Possible` | str | {No, Yes} | Categorical (binary) feature |
| `Subsidy_Available` | str | {No, Yes} | Categorical (binary) feature — **strong signal** |
| `Range_Anxiety_Level` | str | {High, Medium, Low} | Categorical (ordinal) — **strongest signal** |
| `Will_Buy_EV` | str | {No, Yes} | **Target** — 17.5% Yes |

## Class distribution
| Class | Count | % of train |
|---|---|---|
| No (0) | 551,886 | 82.54% |
| Yes (1) | 116,779 | 17.46% |

Class imbalance is moderate. ROC-AUC is invariant to threshold choice so we don't need to rebalance, but if we ever switch to a threshold-based metric we should consider class weights or oversampling.

## Missing values
- **Train:** 0 missing values across all 15 columns.
- **Test:** 0 missing values across all 14 columns.

No imputation logic is needed in v1. If the dataset is regenerated or a later round introduces NaNs, the FE / loader should be revisited.

## Duplicates
- `id` is unique in both train and test.
- 0 full-row duplicates in train.

## Per-feature signal strength (EDA crosstabs)
Strong → weak:

| Feature | Type | Signal | Observation |
|---|---|---|---|
| `Range_Anxiety_Level` | ordinal cat | **Very strong** | High → 0.14% Yes, Medium → 4.17% Yes, Low → 18.90% Yes |
| `Subsidy_Available` | binary cat | **Very strong** | No → 0.58% Yes, Yes → 27.47% Yes |
| `Environmental_Concern_Level` | ordinal num | **Strong** | Yes mean = 4.38, No mean = 2.63 |
| `Annual_Income_USD` | continuous num | **Strong** | Yes mean = $98,827, No mean = $81,795 |
| `Home_Charging_Possible` | binary cat | Moderate | Yes → 19.6% Yes, No → 12.7% Yes |
| `City_Type` | cat | Moderate | Rural → 19.3%, Suburban → 18.1%, Urban → 16.1% |
| `Current_Car_Type` | cat | Weak | Truck lowest (15.6%), SUV highest (18.1%) |
| `Charging_Stations_Near_Home` | num | Weak | Small difference in means |
| `Charging_Stations_Near_Work` | num | Weak | Small difference in means |
| `Daily_Commute_km` | num | Weak | Slightly shorter commute in Yes class |
| `Age` | num | Weak | Almost identical means |
| `Number_of_Cars_Owned` | num | Very weak | Nearly identical distributions |
| `Gender` | cat | Very weak | F = 17.8%, M = 17.2%, O = 17.4% |

## Preprocessing applied (v1, in `src/data.py` + `src/features.py`)
1. Read CSVs with pandas.
2. Validate columns (raise on missing or unexpected).
3. Map `Will_Buy_EV: {No: 0, Yes: 1}` in train.
4. In `build_features`:
   - `Anxiety_ord = {High: 0, Medium: 1, Low: 2}`
   - `Stations_Total = Charging_Stations_Near_Home + Charging_Stations_Near_Work`
   - `Income_per_Age = Annual_Income_USD / Age`
   - `Env_x_Subsidy = Environmental_Concern_Level * (Subsidy_Available == 'Yes')`
   - `Subsidy_x_Income = (Subsidy_Available == 'Yes') * Annual_Income_USD`
   - `Anxiety_x_Stations_Total = Anxiety_ord * Stations_Total`
5. Drop `id` from the feature matrix; keep it separately for submission alignment.

## Known issues / watch-outs
- **`Environmental_Concern_Level` is stored as `float64`** in the raw CSV even though every value is an integer 1–5. The data loader should coerce to `int64` to avoid LightGBM treating it as continuous. **Action: handled in `src/data.py` during Phase 2.**
- **Synthetic data may have rounding artifacts** (e.g., 805 unique commute values, 13,214 unique income values). Not a problem for tree models, but worth noting if we ever use linear models.
- **No temporal column.** All rows are exchangeable. No need for time-based CV.

## What to watch out for
- **Category-set consistency between train and test.** All 6 categorical columns have the same set of unique values in both. Verified at EDA. The loader should re-assert this in case the dataset is regenerated.
- **Income and Age are not log-transformed.** Linear models would benefit. Tree models do not.
- **Engineered features are all linear combinations of raw features.** LightGBM could derive most of them itself via splits. We add them anyway because (a) they encode domain priors (e.g., a person with high income AND a subsidy is a special segment) and (b) the gain on this synthetic data has historically been positive.
