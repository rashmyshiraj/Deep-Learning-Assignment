# ============================================================
# PART 3 — DATA PREPROCESSING
# FINAL DATASET PREPARATION
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
print("PART 3 — FINAL DATASET PREPROCESSING")
print("=" * 80)

print(f"Input file: {INPUT_FILE}")
print(f"Input dataset shape: {df.shape}")
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
    col
    for col in final_columns_proposed
    if col not in df.columns
]

excluded_columns = [
    col
    for col in df.columns
    if col not in final_columns_proposed
]

print("=" * 80)
print("STEP 7.1 — COLUMN VALIDATION")
print("=" * 80)

print(f"Required columns: {len(final_columns_proposed)}")
print(f"Available columns: {len(df.columns)}")
print()

if missing_columns:

    print("MISSING REQUIRED COLUMNS")
    print("-" * 80)

    for col in missing_columns:
        print(f"  - {col}")

    raise ValueError(
        "Required model-ready columns are missing."
    )

print("All required model-ready columns are available.")
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

df_optimized = df[
    final_columns_proposed
].copy()


# ============================================================
# ORGANIZE COLUMN ORDER
# ============================================================

predictor_columns = [
    col
    for col in df_optimized.columns
    if col not in ["date", "arrivals"]
]

predictor_columns = sorted(
    predictor_columns
)

final_column_order = (
    ["date"]
    + predictor_columns
    + ["arrivals"]
)

df_optimized = df_optimized[
    final_column_order
]


# ============================================================
# VERIFY OPTIMIZED DATASET
# ============================================================

print(f"Optimized dataset shape: {df_optimized.shape}")
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
# FILTER COMPLETE EXCHANGE-RATE RECORDS
# ============================================================

if "exchange_rates_complete" in model_df.columns:

    before_filter = len(model_df)

    model_df = model_df[
        model_df["exchange_rates_complete"] == 1
    ].copy()

    model_df = model_df.reset_index(
        drop=True
    )

    after_filter = len(model_df)

    print(
        f"Rows before completeness filter: "
        f"{before_filter}"
    )

    print(
        f"Rows after completeness filter: "
        f"{after_filter}"
    )

    print(
        f"Rows removed: "
        f"{before_filter - after_filter}"
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
# CYCLICAL FEATURES
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
# TARGET / PREDICTORS
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
# MODEL-READY VALIDATION
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

missing_model_values = (
    model_df.isnull().sum()
)

missing_model_values = (
    missing_model_values[
        missing_model_values > 0
    ]
)

print()
print("Missing values in model-ready dataset:")

if len(missing_model_values) > 0:
    print(missing_model_values)
else:
    print("None")

print()


# ============================================================
# STEP 7.3 — FINAL OUTPUT DIRECTORY
# ============================================================

print("=" * 80)
print("STEP 7.3 — SAVE FINAL PREPROCESSED DATASETS")
print("=" * 80)

OUTPUT_DIR = "/content/final_preprocessed_dataset"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE OPTIMIZED PREPROCESSED DATASET
# ============================================================

preprocessed_file = os.path.join(
    OUTPUT_DIR,
    "Sri_Lanka_Tourism_Arrivals_Preprocessed.csv"
)

df_optimized.to_csv(
    preprocessed_file,
    index=False
)

print(
    f"Saved preprocessed dataset: "
    f"{preprocessed_file}"
)


# ============================================================
# DATA DICTIONARY
# ============================================================

feature_descriptions = {

    "date":
        "Date of the observation.",

    "arrivals":
        "Daily Sri Lanka tourist arrivals target variable.",

    "arrivals_lag_1":
        "Tourist arrivals from the previous day.",

    "arrivals_lag_7":
        "Tourist arrivals from seven days earlier.",

    "arrivals_lag_14":
        "Tourist arrivals from fourteen days earlier.",

    "arrivals_lag_30":
        "Tourist arrivals from thirty days earlier.",

    "arrivals_rolling_mean_7":
        "Seven-day rolling mean of tourist arrivals.",

    "arrivals_rolling_mean_14":
        "Fourteen-day rolling mean of tourist arrivals.",

    "arrivals_rolling_mean_30":
        "Thirty-day rolling mean of tourist arrivals.",

    "year":
        "Calendar year extracted from the observation date.",

    "month":
        "Calendar month extracted from the observation date.",

    "day_of_month":
        "Day of month extracted from the observation date.",

    "day_of_week":
        "Day of week represented numerically.",

    "is_weekend":
        "Binary indicator showing whether the date falls on a weekend.",

    "month_sin":
        "Sine transformation representing the cyclical month position.",

    "month_cos":
        "Cosine transformation representing the cyclical month position.",

    "day_of_week_sin":
        "Sine transformation representing cyclical day-of-week position.",

    "day_of_week_cos":
        "Cosine transformation representing cyclical day-of-week position.",

    "gdp":
        "Gross domestic product feature used as an economic predictor.",

    "inflation":
        "Inflation feature used as an economic predictor.",

    "exchange_rate":
        "Exchange rate feature used as an economic predictor.",

    "exchange_rates_complete":
        "Indicator showing whether the exchange-rate information is complete.",

    "holiday_name_grouped":
        "Grouped holiday name feature.",

    "holiday_type_grouped":
        "Grouped holiday type feature.",

    "is_holiday":
        "Binary indicator identifying holidays.",

    "event_name_grouped":
        "Grouped event name feature.",

    "is_event":
        "Binary indicator identifying event periods.",

    "temperature":
        "Temperature-related weather feature.",

    "rainfall":
        "Rainfall-related weather feature.",

    "tourism_season":
        "Tourism season feature representing seasonal tourism patterns.",
}


# ============================================================
# CREATE DATA DICTIONARY
# ============================================================

data_dictionary = pd.DataFrame({
    "feature": df_optimized.columns,
    "description": [
        feature_descriptions.get(
            col,
            "Feature used in the final preprocessed dataset."
        )
        for col in df_optimized.columns
    ],
    "data_type": [
        str(df_optimized[col].dtype)
        for col in df_optimized.columns
    ],
})


# ============================================================
# SAVE DATA DICTIONARY
# ============================================================

dictionary_file = os.path.join(
    OUTPUT_DIR,
    "Sri_Lanka_Tourism_Arrivals_Data_Dictionary.csv"
)

data_dictionary.to_csv(
    dictionary_file,
    index=False
)

print(
    f"Saved data dictionary: "
    f"{dictionary_file}"
)


# ============================================================
# SAVE MODEL-READY CLEAN DATASET
# ============================================================

clean_file = os.path.join(
    OUTPUT_DIR,
    "Sri_Lanka_Tourism_Arrivals_Clean_Complete.csv"
)

model_df.to_csv(
    clean_file,
    index=False
)

print(
    f"Saved clean model-ready dataset: "
    f"{clean_file}"
)


# ============================================================
# VERIFY SAVED FILES
# ============================================================

print()
print("=" * 80)
print("VERIFY SAVED OUTPUTS")
print("=" * 80)

saved_files = [
    preprocessed_file,
    dictionary_file,
    clean_file,
]

for file_path in saved_files:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Expected output file was not created: {file_path}"
        )

    file_size = os.path.getsize(file_path)

    print(
        f"✓ {os.path.basename(file_path)} "
        f"({file_size:,} bytes)"
    )


# ============================================================
# RELOAD AND VERIFY
# ============================================================

verified_preprocessed = pd.read_csv(
    preprocessed_file
)

verified_dictionary = pd.read_csv(
    dictionary_file
)

verified_clean = pd.read_csv(
    clean_file
)

print()
print("Verified dataset shapes:")

print(
    f"Preprocessed dataset: "
    f"{verified_preprocessed.shape}"
)

print(
    f"Data dictionary: "
    f"{verified_dictionary.shape}"
)

print(
    f"Clean model-ready dataset: "
    f"{verified_clean.shape}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("PART 3 PREPROCESSING COMPLETED")
print("=" * 80)

print(
    f"Final optimized dataset rows: "
    f"{len(df_optimized):,}"
)

print(
    f"Final optimized dataset columns: "
    f"{len(df_optimized.columns):,}"
)

print(
    f"Final model-ready rows: "
    f"{len(model_df):,}"
)

print(
    f"Final model-ready columns: "
    f"{len(model_df.columns):,}"
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Files created:")
for file_path in saved_files:
    print(f"  - {os.path.basename(file_path)}")

print()
print("Preprocessing completed successfully.")