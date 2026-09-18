
# -*- coding: utf-8 -*-
"""Data Categorical Encoding

Original file: Untitled4.ipynb
"""

import pandas as pd

# Load the CSV already uploaded to the environment
file_path = "/content/Sri_Lanka_Tourism_Arrivals_Clean_Complete.csv"

df = pd.read_csv(file_path)

categorical_columns = [
    "holiday_name_grouped",
    "event_name_grouped"
]

# Check that both columns exist
print("Columns found:")
print([column for column in categorical_columns if column in df.columns])

# Display basic information
print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df[categorical_columns].dtypes)

print("\nMissing values:")
print(df[categorical_columns].isna().sum())

print("\nNumber of unique values:")
print(df[categorical_columns].nunique(dropna=False))

# Display categories and their frequencies
for column in categorical_columns:
    print("\n" + "=" * 60)
    print(f"Column: {column}")
    print("=" * 60)
    print(df[column].value_counts(dropna=False))

# Make a safety copy so the original df remains available if needed
df_encoded = df.copy()

# Convert text categories into 0/1 numeric columns
df_encoded = pd.get_dummies(
    df_encoded,
    columns=categorical_columns,
    prefix=categorical_columns,
    dtype=int
)

# Show the new dataset size
print("Original dataset shape:", df.shape)
print("Encoded dataset shape:", df_encoded.shape)

# Count the newly created numerical columns
holiday_encoded_columns = [
    col for col in df_encoded.columns
    if col.startswith("holiday_name_grouped_")
]

event_encoded_columns = [
    col for col in df_encoded.columns
    if col.startswith("event_name_grouped_")
]

print("\nHoliday numerical columns created:", len(holiday_encoded_columns))
print("Event numerical columns created:", len(event_encoded_columns))

print("\nFirst 10 holiday encoded columns:")
print(holiday_encoded_columns[:10])

print("\nFirst 10 event encoded columns:")
print(event_encoded_columns[:10])

import numpy as np

# Original text columns should no longer be present
original_text_columns = [
    "holiday_name_grouped",
    "event_name_grouped"
]

print("1. Original text columns removed:")
for col in original_text_columns:
    print(f"{col}: {'REMOVED ✅' if col not in df_encoded.columns else 'STILL PRESENT ❌'}")

# Verify all generated encoding columns are numerical
encoded_columns = holiday_encoded_columns + event_encoded_columns

print("\n2. Data types of encoded columns:")
print(df_encoded[encoded_columns].dtypes.value_counts())

# Verify values are only 0 or 1
non_binary_columns = []

for col in encoded_columns:
    unique_values = set(df_encoded[col].dropna().unique())

    if not unique_values.issubset({0, 1}):
        non_binary_columns.append({
            "column": col,
            "unique_values": sorted(unique_values)
        })

print("\n3. Binary-value check:")
if len(non_binary_columns) == 0:
    print("PASS ✅ All holiday and event encoded columns contain only 0 and 1.")
else:
    print("FAIL ❌ Some columns contain values other than 0 and 1:")
    print(non_binary_columns)

# Check that one category is active per row for each original column
holiday_row_sums = df_encoded[holiday_encoded_columns].sum(axis=1)
event_row_sums = df_encoded[event_encoded_columns].sum(axis=1)

print("\n4. One active category per row:")
print("Rows where exactly one holiday category is active:",
      (holiday_row_sums == 1).sum(), "out of", len(df_encoded))

print("Rows where exactly one event category is active:",
      (event_row_sums == 1).sum(), "out of", len(df_encoded))

# Check missing values in the new numerical features
print("\n5. Missing values in encoded columns:")
print(df_encoded[encoded_columns].isna().sum().sum())

# Final summary
print("\n6. Final encoded dataset shape:")
print(df_encoded.shape)

# Define the output file name
output_file = "Sri_Lanka_Tourism_Arrivals_Encoded.csv"

# Save the encoded dataset as a CSV file
df_encoded.to_csv(output_file, index=False)

# Verify the saved file
print("File saved successfully:", output_file)
print("Saved dataset shape:", df_encoded.shape)

# Optional: preview the first 5 rows
display(df_encoded.head())