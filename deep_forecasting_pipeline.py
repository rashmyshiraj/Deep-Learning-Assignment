# -*- coding: utf-8 -*-
"""Deep Learning Tourism Forecasting Pipeline

Original file: Untitled5.ipynb
"""

import pandas as pd
import numpy as np

# Change this only if your uploaded filename is different
file_path = "Sri_Lanka_Tourism_Arrivals_Encoded.csv"

# Load the dataset without modifying it
df = pd.read_csv(file_path)

print("=" * 80)
print("1. DATASET SHAPE")
print("=" * 80)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")

print("\n" + "=" * 80)
print("2. COLUMN NAMES")
print("=" * 80)
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("4. FIRST FIVE ROWS")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("5. LAST FIVE ROWS")
print("=" * 80)
print(df.tail())

print("\n" + "=" * 80)
print("6. MISSING VALUES")
print("=" * 80)
missing_values = df.isna().sum()
missing_summary = pd.DataFrame({
    "missing_count": missing_values,
    "missing_percentage": (missing_values / len(df) * 100).round(4)
})
print(missing_summary)

print("\n" + "=" * 80)
print("7. DUPLICATE ROWS")
print("=" * 80)
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\n" + "=" * 80)
print("8. NUMERIC COLUMN SUMMARY")
print("=" * 80)
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if numeric_columns:
    print(df[numeric_columns].describe().T)
else:
    print("No numeric columns found.")

print("\n" + "=" * 80)
print("9. INFINITE VALUES")
print("=" * 80)
if numeric_columns:
    infinite_counts = np.isinf(df[numeric_columns]).sum()
    print(infinite_counts)
    print(f"Total infinite values: {infinite_counts.sum():,}")
else:
    print("No numeric columns available for infinite-value checking.")

print("\n" + "=" * 80)
print("10. POSSIBLE DATE COLUMNS")
print("=" * 80)
possible_date_columns = []

for column in df.columns:
    column_name = str(column).lower()

    if (
        "date" in column_name
        or "time" in column_name
        or "day" in column_name
        or "month" in column_name
        or "year" in column_name
    ):
        possible_date_columns.append(column)

print(possible_date_columns)

print("\n" + "=" * 80)
print("11. POSSIBLE TARGET / ARRIVALS COLUMNS")
print("=" * 80)
possible_target_columns = []

for column in df.columns:
    column_name = str(column).lower()

    if (
        "arrival" in column_name
        or "tourist" in column_name
        or "visitor" in column_name
        or "target" in column_name
        or "passenger" in column_name
    ):
        possible_target_columns.append(column)

print(possible_target_columns)

print("\n" + "=" * 80)
print("12. DETAILED DATE CHECKS")
print("=" * 80)

for date_column in possible_date_columns:
    print(f"\nDate column candidate: {date_column}")

    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

    print(f"Valid dates: {parsed_dates.notna().sum():,}")
    print(f"Invalid or unparsed dates: {parsed_dates.isna().sum():,}")

    if parsed_dates.notna().any():
        valid_dates = parsed_dates.dropna().sort_values()

        print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
        print(f"Duplicate dates: {parsed_dates.duplicated().sum():,}")

        date_differences = valid_dates.diff().dropna()

        print("Most common date intervals:")
        print(date_differences.value_counts().head(10))

        expected_daily_dates = pd.date_range(
            start=valid_dates.min(),
            end=valid_dates.max(),
            freq="D"
        )

        actual_dates = pd.DatetimeIndex(valid_dates.unique())
        missing_dates = expected_daily_dates.difference(actual_dates)

        print(f"Expected dates for continuous daily frequency: {len(expected_daily_dates):,}")
        print(f"Unique dates found: {len(actual_dates):,}")
        print(f"Missing dates: {len(missing_dates):,}")

        if len(missing_dates) > 0:
            print("First missing dates:")
            print(missing_dates[:20].tolist())

print("\n" + "=" * 80)
print("13. POSSIBLE TARGET VALUE CHECKS")
print("=" * 80)

for target_column in possible_target_columns:
    print(f"\nTarget column candidate: {target_column}")

    target_numeric = pd.to_numeric(df[target_column], errors="coerce")

    print(f"Numeric values: {target_numeric.notna().sum():,}")
    print(f"Non-numeric or unparsed values: {target_numeric.isna().sum():,}")
    print(f"Negative values: {(target_numeric < 0).sum():,}")

    if target_numeric.notna().any():
        print("Basic statistics:")
        print(target_numeric.describe())

print("\n" + "=" * 80)
print("14. UNIQUE VALUE COUNTS")
print("=" * 80)

unique_summary = pd.DataFrame({
    "column": df.columns,
    "unique_values": [df[column].nunique(dropna=False) for column in df.columns],
    "data_type": [str(df[column].dtype) for column in df.columns]
})

print(unique_summary.to_string(index=False))

print("\n" + "=" * 80)
print("15. SUSPICIOUS OR POTENTIALLY PROBLEMATIC COLUMN NAMES")
print("=" * 80)

suspicious_keywords = [
    "target",
    "label",
    "future",
    "forecast",
    "prediction",
    "predicted",
    "actual",
    "index",
    "id",
    "unnamed",
    "rolling",
    "moving",
    "lag",
    "lead",
    "next",
    "future",
    "encoded"
]

suspicious_columns = []

for column in df.columns:
    column_name = str(column).lower()

    matched_keywords = [
        keyword for keyword in suspicious_keywords
        if keyword in column_name
    ]

    if matched_keywords:
        suspicious_columns.append({
            "column": column,
            "matched_keywords": matched_keywords,
            "data_type": str(df[column].dtype),
            "unique_values": df[column].nunique(dropna=False)
        })

if suspicious_columns:
    print(pd.DataFrame(suspicious_columns).to_string(index=False))
else:
    print("No column names matched the initial suspicious-keyword check.")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("No data was modified.")