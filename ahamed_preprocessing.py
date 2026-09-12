# ============================================================
# PART 3 — DATA PREPROCESSING
# STEP 7 — FINAL DATASET PREPARATION
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 7.1 — LOAD PERSON 2 OUTPUT
# ============================================================

INPUT_FILE = "/content/step_4_feature_engineering_pre_target_encoding.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("STEP 7 — FINAL DATASET PREPARATION")
print("=" * 80)

print(f"Input file: {INPUT_FILE}")
print(f"Dataset shape: {df.shape}")
print()


# ============================================================
# STEP 7.1 — DEFINE FINAL MODEL-READY COLUMNS
# ============================================================

final_columns_proposed = [
    "date",
    "arrivals",

    "arrivals_lag_1",
    "arrivals_lag_7",
    "arrivals_lag_14",
    "arrivals_lag_30",

    "arrivals_rolling_mean_7",
    "arrivals_rolling_mean_14",
    "arrivals_rolling_mean_30",

    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "is_weekend",

    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",

    "gdp",
    "inflation",
    "exchange_rate",
    "exchange_rates_complete",

    "holiday_name_grouped",
    "holiday_type_grouped",
    "is_holiday",

    "event_name_grouped",
    "is_event",

    "temperature",
    "rainfall",

    "tourism_season",
]


# ============================================================
# VALIDATE COLUMNS
# ============================================================

missing_columns = [
    col for col in final_columns_proposed
    if col not in df.columns
]

excluded_columns = [
    col for col in df.columns
    if col not in final_columns_proposed
]

if missing_columns:
    print("ERROR — MISSING REQUIRED COLUMNS")

    for col in missing_columns:
        print(f"  - {col}")

    raise ValueError(
        "Required model-ready columns are missing."
    )

print("All required columns are available.")
print()

print("Excluded columns:")
if excluded_columns:
    for col in excluded_columns:
        print(f"  - {col}")
else:
    print("  None")

print()


# ============================================================
# STEP 7.2 — CREATE OPTIMIZED DATASET
# ============================================================

print("=" * 80)
print("STEP 7.2 — CREATE OPTIMIZED DATASET")
print("=" * 80)

df_optimized = df[final_columns_proposed].copy()


# ============================================================
# COLUMN ORDER
# ============================================================

predictor_columns = [
    col
    for col in df_optimized.columns
    if col not in ["date", "arrivals"]
]

predictor_columns = sorted(predictor_columns)

final_column_order = (
    ["date"]
    + predictor_columns
    + ["arrivals"]
)

df_optimized = df_optimized[final_column_order]


# ============================================================
# VERIFY OPTIMIZED DATASET
# ============================================================

print(f"Optimized dataset shape: {df_optimized.shape}")
print()

print("First 10 columns:")
print(df_optimized.columns[:10].tolist())

print()
print("Last 10 columns:")
print(df_optimized.columns[-10:].tolist())

print()

missing_after_optimization = (
    df_optimized.isnull().sum()
)

missing_after_optimization = (
    missing_after_optimization[
        missing_after_optimization > 0
    ]
)

print("Missing values after optimization:")
if len(missing_after_optimization) > 0:
    print(missing_after_optimization)
else:
    print("None")

print()


# ============================================================
# MODEL-READY DATASET PREPARATION
# ============================================================

print("=" * 80)
print("MODEL-READY DATASET PREPARATION")
print("=" * 80)


model_df = df_optimized.copy()


# ============================================================
# SORT BY DATE
# ============================================================

model_df["date"] = pd.to_datetime(
    model_df["date"],
    errors="coerce"
)

model_df = model_df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# KEEP COMPLETE EXCHANGE-RATE RECORDS
# ============================================================

if "exchange_rates_complete" in model_df.columns:

    before_filter = len(model_df)

    model_df = model_df[
        model_df["exchange_rates_complete"] == 1
    ].copy()

    model_df = model_df.reset_index(drop=True)

    after_filter = len(model_df)

    print(
        f"Rows before exchange-rate completeness filter: "
        f"{before_filter}"
    )

    print(
        f"Rows after exchange-rate completeness filter: "
        f"{after_filter}"
    )

    print(
        f"Rows removed: {before_filter - after_filter}"
    )

print()


# ============================================================
# CALENDAR FEATURES
# ============================================================

model_df["year"] = model_df["date"].dt.year
model_df["month"] = model_df["date"].dt.month
model_df["day_of_month"] = model_df["date"].dt.day
model_df["day_of_week"] = model_df["date"].dt.dayofweek

model_df["is_weekend"] = (
    model_df["day_of_week"] >= 5
).astype(int)


# ============================================================
# CYCLICAL CALENDAR FEATURES
# ============================================================

model_df["month_sin"] = np.sin(
    2 * np.pi * model_df["month"] / 12
)

model_df["month_cos"] = np.cos(
    2 * np.pi * model_df["month"] / 12
)

model_df["day_of_week_sin"] = np.sin(
    2 * np.pi * model_df["day_of_week"] / 7
)

model_df["day_of_week_cos"] = np.cos(
    2 * np.pi * model_df["day_of_week"] / 7
)


# ============================================================
# TARGET AND CATEGORICAL COLUMNS
# ============================================================

target_column = "arrivals"

categorical_columns = [
    col
    for col in [
        "event_name_grouped",
        "holiday_name_grouped"
    ]
    if col in model_df.columns
]


# ============================================================
# IDENTIFY PREDICTORS
# ============================================================

predictor_columns = [
    col
    for col in model_df.columns
    if col not in [
        target_column,
        "date"
    ]
]

numeric_predictors = model_df[
    predictor_columns
].select_dtypes(
    include=[np.number]
).columns.tolist()


# ============================================================
# FINAL MODEL-READY VALIDATION
# ============================================================

print("Target column:")
print(f"  {target_column}")

print()
print("Categorical columns:")
for col in categorical_columns:
    print(f"  - {col}")

print()
print(f"Total predictors: {len(predictor_columns)}")
print(f"Numeric predictors: {len(numeric_predictors)}")

print()

missing_model_values = (
    model_df.isnull().sum()
)

missing_model_values = (
    missing_model_values[
        missing_model_values > 0
    ]
)

print("Missing values in model-ready dataset:")

if len(missing_model_values) > 0:
    print(missing_model_values)
else:
    print("None")

print()
print("STEP 7.2 completed successfully.")