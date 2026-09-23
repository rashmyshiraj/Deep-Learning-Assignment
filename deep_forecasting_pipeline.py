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


# ============================================================
# FEATURE CLASSIFICATION AND LEAKAGE CHECKS
# ============================================================
df["date"] = pd.to_datetime(df["date"])

target_column = "arrivals"

historical_features = [
    "brent_crude_price",
    "cny_lkr",
    "eur_lkr",
    "gbp_lkr",
    "inr_lkr",
    "rub_lkr",
    "usd_lkr",
    "gdp_per_capita",
    "inflation_rate",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "is_rainy_day",
    "image_search",
    "web_search",
    "youtube_search",
    "exchange_rates_complete",
    "cny_lkr_was_missing",
    "eur_lkr_was_missing",
    "gbp_lkr_was_missing",
    "inr_lkr_was_missing",
    "rub_lkr_was_missing",
    "usd_lkr_was_missing",
    "gdp_per_capita_was_missing",
    "inflation_rate_was_missing"
]

known_future_features = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "days_since_last_holiday",
    "days_since_last_holiday_was_missing",
    "days_to_next_holiday_was_missing",
    "in_holiday_window",
    "is_holiday",
    "is_tourist_event"
]

holiday_features = [
    column for column in df.columns
    if column.startswith("holiday_name_grouped_")
]

event_features = [
    column for column in df.columns
    if column.startswith("event_name_grouped_")
]

all_classified_features = (
    historical_features
    + known_future_features
    + holiday_features
    + event_features
)

unclassified_features = [
    column for column in df.columns
    if column not in ["date", target_column]
    and column not in all_classified_features
]

print("=" * 90)
print("1. TARGET")
print("=" * 90)
print(f"Target column: {target_column}")
print(f"Target data type: {df[target_column].dtype}")
print(f"Target minimum: {df[target_column].min()}")
print(f"Target maximum: {df[target_column].max()}")

print("\n" + "=" * 90)
print("2. HISTORICAL / OBSERVED FEATURES")
print("=" * 90)
for column in historical_features:
    print(f"{column} | dtype={df[column].dtype} | unique_values={df[column].nunique()}")

print("\n" + "=" * 90)
print("3. KNOWN-FUTURE CALENDAR / HOLIDAY FEATURES")
print("=" * 90)

known_future_display = [
    column for column in known_future_features
    if column in df.columns
]

for column in known_future_display:
    print(f"{column} | dtype={df[column].dtype} | unique_values={df[column].nunique()}")

print("\n" + "=" * 90)
print("4. HOLIDAY ONE-HOT FEATURES")
print("=" * 90)
print(f"Number of holiday features: {len(holiday_features)}")
for column in holiday_features:
    print(f"{column} | unique_values={df[column].nunique()}")

print("\n" + "=" * 90)
print("5. EVENT ONE-HOT FEATURES")
print("=" * 90)
print(f"Number of event features: {len(event_features)}")
for column in event_features:
    print(f"{column} | unique_values={df[column].nunique()}")

print("\n" + "=" * 90)
print("6. UNCLASSIFIED FEATURES")
print("=" * 90)

if unclassified_features:
    for column in unclassified_features:
        print(f"{column} | dtype={df[column].dtype} | unique_values={df[column].nunique()}")
else:
    print("No unclassified features found.")

print("\n" + "=" * 90)
print("7. POTENTIAL LEAKAGE KEYWORD CHECK")
print("=" * 90)

leakage_keywords = [
    "target",
    "arrival",
    "actual",
    "future",
    "forecast",
    "predicted",
    "prediction",
    "next",
    "lead",
    "rolling",
    "moving",
    "lag",
    "label"
]

leakage_candidates = []

for column in df.columns:
    column_lower = column.lower()

    matched_keywords = [
        keyword for keyword in leakage_keywords
        if keyword in column_lower
    ]

    if matched_keywords:
        leakage_candidates.append({
            "column": column,
            "matched_keywords": matched_keywords,
            "dtype": str(df[column].dtype),
            "unique_values": df[column].nunique()
        })

if leakage_candidates:
    leakage_table = pd.DataFrame(leakage_candidates)
    print(leakage_table.to_string(index=False))
else:
    print("No columns matched the leakage keyword check.")

print("\n" + "=" * 90)
print("8. CONSTANT FEATURES")
print("=" * 90)

constant_features = [
    column for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

if constant_features:
    for column in constant_features:
        print(
            f"{column} | value={df[column].iloc[0]} | "
            f"dtype={df[column].dtype}"
        )
else:
    print("No constant features found.")

print("\n" + "=" * 90)
print("9. FEATURE VALUE AVAILABILITY SUMMARY")
print("=" * 90)

feature_summary = []

for column in df.columns:
    if column == "date":
        category = "Date"
    elif column == target_column:
        category = "Target"
    elif column in historical_features:
        category = "Historical / observed"
    elif column in known_future_features:
        category = "Known future candidate"
    elif column in holiday_features:
        category = "Known future holiday candidate"
    elif column in event_features:
        category = "Known future event candidate"
    else:
        category = "Unclassified"

    feature_summary.append({
        "column": column,
        "category": category,
        "dtype": str(df[column].dtype),
        "unique_values": df[column].nunique(),
        "missing_values": df[column].isna().sum()
    })

feature_summary_df = pd.DataFrame(feature_summary)

print(
    feature_summary_df["category"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 90)
print("10. FULL FEATURE CLASSIFICATION TABLE")
print("=" * 90)
print(feature_summary_df.to_string(index=False))

print("\n" + "=" * 90)
print("CLASSIFICATION COMPLETE")
print("=" * 90)
print("No columns were removed or modified.")

# ============================================================
# CONSTRUCT FINAL MODELING FEATURES DATAFRAME
# ============================================================
columns_to_drop = [
    "covid_impact_factor",
    "crisis_impact_factor",
    "exchange_rates_complete"
]

columns_to_drop = [c for c in columns_to_drop if c in df.columns]
df_model = df.drop(columns=columns_to_drop)

target_column = "arrivals"

historical_features = [
    "brent_crude_price",
    "cny_lkr",
    "eur_lkr",
    "gbp_lkr",
    "inr_lkr",
    "rub_lkr",
    "usd_lkr",
    "gdp_per_capita",
    "inflation_rate",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "is_rainy_day",
    "image_search",
    "web_search",
    "youtube_search",
    "cny_lkr_was_missing",
    "eur_lkr_was_missing",
    "gbp_lkr_was_missing",
    "inr_lkr_was_missing",
    "rub_lkr_was_missing",
    "usd_lkr_was_missing",
    "gdp_per_capita_was_missing",
    "inflation_rate_was_missing"
]

known_future_base_features = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "days_since_last_holiday",
    "days_since_last_holiday_was_missing",
    "days_to_next_holiday_was_missing",
    "in_holiday_window",
    "is_holiday",
    "is_tourist_event"
]

holiday_features = [
    c for c in df_model.columns
    if c.startswith("holiday_name_grouped_")
]

event_features = [
    c for c in df_model.columns
    if c.startswith("event_name_grouped_")
]

known_future_features = (
    known_future_base_features
    + holiday_features
    + event_features
)

all_features = historical_features + known_future_features

missing_columns = [c for c in all_features + [target_column] if c not in df_model.columns]

if missing_columns:
    print("ERROR: The following expected columns are missing:")
    print(missing_columns)
else:
    print("All expected feature columns are present.")

modeling_columns = ["date"] + all_features + [target_column]
df_final = df_model[modeling_columns].copy()

print("\nFinal modeling dataframe shape:")
print(df_final.shape)

output_filename = "Sri_Lanka_Tourism_Modeling_Features.csv"
df_final.to_csv(output_filename, index=False)
print(f"\nSaved modeling dataframe to: {output_filename}")


# ============================================================
# CHRONOLOGICAL SPLITTING AND SCALING
# ============================================================
from sklearn.preprocessing import StandardScaler

df_final["date"] = pd.to_datetime(df_final["date"])
df_final = df_final.sort_values("date").reset_index(drop=True)

train_end_date = pd.Timestamp("2023-12-31")
val_end_date = pd.Timestamp("2024-12-31")
test_start_date = pd.Timestamp("2025-01-01")

train_mask = df_final["date"] <= train_end_date
val_mask = (df_final["date"] > train_end_date) & (df_final["date"] <= val_end_date)
test_mask = df_final["date"] >= test_start_date

train_df = df_final[train_mask].reset_index(drop=True)
val_df = df_final[val_mask].reset_index(drop=True)
test_df = df_final[test_mask].reset_index(drop=True)

features_to_scale = [
    "brent_crude_price",
    "cny_lkr",
    "eur_lkr",
    "gbp_lkr",
    "inr_lkr",
    "rub_lkr",
    "usd_lkr",
    "gdp_per_capita",
    "inflation_rate",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "image_search",
    "web_search",
    "youtube_search",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "days_since_last_holiday"
]

feature_columns = [
    c for c in df_final.columns
    if c not in ["date", target_column]
]

scaler = StandardScaler()
scaler.fit(train_df[features_to_scale])

train_scaled_values = scaler.transform(train_df[features_to_scale])
val_scaled_values = scaler.transform(val_df[features_to_scale])
test_scaled_values = scaler.transform(test_df[features_to_scale])

train_scaled_features = pd.DataFrame(
    train_scaled_values,
    columns=features_to_scale,
    index=train_df.index
)

val_scaled_features = pd.DataFrame(
    val_scaled_values,
    columns=features_to_scale,
    index=val_df.index
)

test_scaled_features = pd.DataFrame(
    test_scaled_values,
    columns=features_to_scale,
    index=test_df.index
)

unscaled_features = [
    c for c in feature_columns
    if c not in features_to_scale
]

train_full = pd.concat(
    [
        train_df[["date"]],
        train_scaled_features,
        train_df[unscaled_features],
        train_df[[target_column]]
    ],
    axis=1
)

val_full = pd.concat(
    [
        val_df[["date"]],
        val_scaled_features,
        val_df[unscaled_features],
        val_df[[target_column]]
    ],
    axis=1
)

test_full = pd.concat(
    [
        test_df[["date"]],
        test_scaled_features,
        test_df[unscaled_features],
        test_df[[target_column]]
    ],
    axis=1
)

train_full.to_csv("train_scaled.csv", index=False)
val_full.to_csv("val_scaled.csv", index=False)
test_full.to_csv("test_scaled.csv", index=False)

print("Chronological split and scaling complete.")

# ============================================================
# SLIDING WINDOW CREATION (90-DAY INPUT, 30-DAY HORIZON)
# ============================================================
input_window = 90
forecast_horizon = 30

def create_windows(dataframe, feature_columns, target_column,
                   input_window=90, forecast_horizon=30):

    feature_values = dataframe[feature_columns].to_numpy(dtype=np.float32)
    target_values = dataframe[target_column].to_numpy(dtype=np.float32)
    dates = dataframe["date"].to_numpy()

    X_windows = []
    y_windows = []
    input_date_windows = []
    target_date_windows = []

    max_start = len(dataframe) - input_window - forecast_horizon + 1

    for start_index in range(max_start):
        input_start = start_index
        input_end = start_index + input_window

        target_start = input_end
        target_end = target_start + forecast_horizon

        X_windows.append(feature_values[input_start:input_end])
        y_windows.append(target_values[target_start:target_end])

        input_date_windows.append(dates[input_start:input_end])
        target_date_windows.append(dates[target_start:target_end])

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_windows, dtype=np.float32),
        np.array(input_date_windows),
        np.array(target_date_windows)
    )

X_train, y_train, train_input_dates, train_target_dates = create_windows(
    train_full,
    feature_columns,
    target_column,
    input_window,
    forecast_horizon
)

X_val, y_val, val_input_dates, val_target_dates = create_windows(
    val_full,
    feature_columns,
    target_column,
    input_window,
    forecast_horizon
)

X_test, y_test, test_input_dates, test_target_dates = create_windows(
    test_full,
    feature_columns,
    target_column,
    input_window,
    forecast_horizon
)

assert X_train.ndim == 3 and X_val.ndim == 3 and X_test.ndim == 3
print("3D time series tensor windows generated successfully.")


# ============================================================
# SEASONAL NAÏVE BASELINE MODEL (7-DAY PERIODICITY)
# ============================================================
seasonal_period = 7

def seasonal_naive_forecast(history_values, horizon, seasonal_period=7):
    history_values = list(history_values)
    predictions = []
    for _ in range(horizon):
        next_prediction = history_values[-seasonal_period]
        predictions.append(next_prediction)
        history_values.append(next_prediction)
    return np.array(predictions, dtype=np.float32)

seasonal_naive_predictions = []
for sample_index in range(len(y_test)):
    target_start_date = pd.Timestamp(test_target_dates[sample_index][0])
    history_end_date = target_start_date - pd.Timedelta(days=1)
    history_start_date = history_end_date - pd.Timedelta(days=seasonal_period - 1)

    history_values = test_full.loc[
        (test_full["date"] >= history_start_date) &
        (test_full["date"] <= history_end_date),
        target_column
    ].to_numpy(dtype=np.float32)

    prediction = seasonal_naive_forecast(
        history_values=history_values,
        horizon=forecast_horizon,
        seasonal_period=seasonal_period
    )
    seasonal_naive_predictions.append(prediction)

seasonal_naive_predictions = np.array(seasonal_naive_predictions)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

train_target_values = train_full[target_column].to_numpy(dtype=np.float32)
mase_denominator = np.mean(
    np.abs(train_target_values[seasonal_period:] - train_target_values[:-seasonal_period])
)

def mase(y_true, y_pred, denominator):
    return np.mean(np.abs(y_true - y_pred)) / denominator

baseline_mae = mae(y_test, seasonal_naive_predictions)
baseline_rmse = rmse(y_test, seasonal_naive_predictions)
baseline_smape = smape(y_test, seasonal_naive_predictions)
baseline_mase = mase(y_test, seasonal_naive_predictions, mase_denominator)

baseline_results = {
    "model": "Seasonal Naive (7-day)",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "sMAPE": baseline_smape,
    "MASE": baseline_mase,
    "Day+1_MAE": mae(y_test[:, 0], seasonal_naive_predictions[:, 0]),
    "Day+7_MAE": mae(y_test[:, 6], seasonal_naive_predictions[:, 6]),
    "Day+14_MAE": mae(y_test[:, 13], seasonal_naive_predictions[:, 13]),
    "Day+21_MAE": mae(y_test[:, 20], seasonal_naive_predictions[:, 20]),
    "Day+30_MAE": mae(y_test[:, 29], seasonal_naive_predictions[:, 29])
}
print("Baseline Seasonal Naïve evaluation completed.")


# ============================================================
# LSTM MODEL ARCHITECTURE AND TRAINING
# ============================================================
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

target_scaler = StandardScaler()
train_arrivals = train_full[["arrivals"]].to_numpy(dtype=np.float32)
val_arrivals = val_full[["arrivals"]].to_numpy(dtype=np.float32)
test_arrivals = test_full[["arrivals"]].to_numpy(dtype=np.float32)

target_scaler.fit(train_arrivals)

train_arrivals_scaled = target_scaler.transform(train_arrivals).flatten()
val_arrivals_scaled = target_scaler.transform(val_arrivals).flatten()
test_arrivals_scaled = target_scaler.transform(test_arrivals).flatten()

def get_scaled_targets_for_windows(dataframe, scaled_arrivals, input_date_windows, target_date_windows):
    scaled_arrival_map = dict(zip(dataframe["date"], scaled_arrivals))
    scaled_input_arrivals = []
    scaled_target_arrivals = []
    for input_dates, target_dates in zip(input_date_windows, target_date_windows):
        scaled_input_arrivals.append([scaled_arrival_map[d] for d in pd.to_datetime(input_dates)])
        scaled_target_arrivals.append([scaled_arrival_map[d] for d in pd.to_datetime(target_dates)])
    return np.array(scaled_input_arrivals, dtype=np.float32), np.array(scaled_target_arrivals, dtype=np.float32)

train_arrival_history_scaled, y_train_scaled = get_scaled_targets_for_windows(train_full, train_arrivals_scaled, train_input_dates, train_target_dates)
val_arrival_history_scaled, y_val_scaled = get_scaled_targets_for_windows(val_full, val_arrivals_scaled, val_input_dates, val_target_dates)
test_arrival_history_scaled, y_test_scaled = get_scaled_targets_for_windows(test_full, test_arrivals_scaled, test_input_dates, test_target_dates)

X_train_lstm = np.concatenate([X_train, train_arrival_history_scaled[..., np.newaxis]], axis=2)
X_val_lstm = np.concatenate([X_val, val_arrival_history_scaled[..., np.newaxis]], axis=2)
X_test_lstm = np.concatenate([X_test, test_arrival_history_scaled[..., np.newaxis]], axis=2)

lstm_checkpoint_path = "best_lstm_model.keras"

lstm_model = Sequential([
    Input(shape=(input_window, X_train_lstm.shape[2])),
    LSTM(64, dropout=0.20, recurrent_dropout=0.0),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(forecast_horizon)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

early_stopping = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0001, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=lstm_checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)

history_lstm = lstm_model.fit(
    X_train_lstm,
    y_train_scaled,
    validation_data=(X_val_lstm, y_val_scaled),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# ============================================================
# LSTM TEST SET EVALUATION
# ============================================================
best_lstm_model = tf.keras.models.load_model(lstm_checkpoint_path)
lstm_test_predictions_scaled = best_lstm_model.predict(X_test_lstm, verbose=1)

lstm_test_predictions = target_scaler.inverse_transform(
    lstm_test_predictions_scaled.reshape(-1, 1)
).reshape(lstm_test_predictions_scaled.shape)

lstm_test_actuals = target_scaler.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).reshape(y_test_scaled.shape)

lstm_mae = mae(lstm_test_actuals, lstm_test_predictions)
lstm_rmse = rmse(lstm_test_actuals, lstm_test_predictions)
lstm_smape = smape(lstm_test_actuals, lstm_test_predictions)
lstm_mase = mase(lstm_test_actuals, lstm_test_predictions, mase_denominator)

lstm_results = {
    "model": "LSTM",
    "MAE": lstm_mae,
    "RMSE": lstm_rmse,
    "sMAPE": lstm_smape,
    "MASE": lstm_mase,
    "Day+1_MAE": mae(lstm_test_actuals[:, 0], lstm_test_predictions[:, 0]),
    "Day+7_MAE": mae(lstm_test_actuals[:, 6], lstm_test_predictions[:, 6]),
    "Day+14_MAE": mae(lstm_test_actuals[:, 13], lstm_test_predictions[:, 13]),
    "Day+21_MAE": mae(lstm_test_actuals[:, 20], lstm_test_predictions[:, 20]),
    "Day+30_MAE": mae(lstm_test_actuals[:, 29], lstm_test_predictions[:, 29]),
    "parameter_count": best_lstm_model.count_params()
}

lstm_results_df = pd.DataFrame([lstm_results])
lstm_results_df.to_csv("lstm_test_results.csv", index=False)


# ============================================================
# GRU + TEMPORAL ATTENTION MODEL
# ============================================================
from tensorflow.keras.layers import GRU, Attention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model

gru_inputs = Input(shape=(input_window, X_train_lstm.shape[2]), name="historical_inputs")
gru_sequence = GRU(units=64, dropout=0.20, return_sequences=True, name="gru_encoder")(gru_inputs)
attention_sequence = Attention(name="temporal_attention")([gru_sequence, gru_sequence])
attention_context = GlobalAveragePooling1D(name="attention_context")(attention_sequence)
gru_context = GlobalAveragePooling1D(name="gru_context")(gru_sequence)
combined_context = Concatenate(name="combined_context")([gru_context, attention_context])

x = Dense(64, activation="relu", name="dense_hidden")(combined_context)
x = Dropout(0.20, name="dropout_hidden")(x)
gru_outputs = Dense(forecast_horizon, name="forecast_output")(x)

gru_attention_model = Model(inputs=gru_inputs, outputs=gru_outputs, name="GRU_Temporal_Attention")
gru_attention_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

gru_attention_checkpoint_path = "best_gru_attention_model.keras"
gru_early_stopping = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0001, restore_best_weights=True, verbose=1)
gru_model_checkpoint = ModelCheckpoint(filepath=gru_attention_checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)

history_gru_attention = gru_attention_model.fit(
    X_train_lstm,
    y_train_scaled,
    validation_data=(X_val_lstm, y_val_scaled),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[gru_early_stopping, gru_model_checkpoint],
    verbose=1
)

best_gru_attention_model = tf.keras.models.load_model(gru_attention_checkpoint_path)
gru_attention_predictions_scaled = best_gru_attention_model.predict(X_test_lstm, verbose=1)
gru_attention_predictions = target_scaler.inverse_transform(gru_attention_predictions_scaled.reshape(-1, 1)).reshape(gru_attention_predictions_scaled.shape)
gru_attention_actuals = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)

gru_attention_results = {
    "model": "GRU + Temporal Attention",
    "MAE": mae(gru_attention_actuals, gru_attention_predictions),
    "RMSE": rmse(gru_attention_actuals, gru_attention_predictions),
    "sMAPE": smape(gru_attention_actuals, gru_attention_predictions),
    "MASE": mase(gru_attention_actuals, gru_attention_predictions, mase_denominator),
    "Day+1_MAE": mae(gru_attention_actuals[:, 0], gru_attention_predictions[:, 0]),
    "Day+7_MAE": mae(gru_attention_actuals[:, 6], gru_attention_predictions[:, 6]),
    "Day+14_MAE": mae(gru_attention_actuals[:, 13], gru_attention_predictions[:, 13]),
    "Day+21_MAE": mae(gru_attention_actuals[:, 20], gru_attention_predictions[:, 20]),
    "Day+30_MAE": mae(gru_attention_actuals[:, 29], gru_attention_predictions[:, 29]),
    "parameter_count": best_gru_attention_model.count_params()
}
pd.DataFrame([gru_attention_results]).to_csv("gru_attention_test_results.csv", index=False)


# ============================================================
# DILATED RESIDUAL TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================
from tensorflow.keras.layers import Conv1D, Add, Activation, SpatialDropout1D

def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    residual = x
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=dilation_rate, activation=None)(x)
    x = Activation("relu")(x)
    x = SpatialDropout1D(dropout_rate)(x)

    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=dilation_rate, activation=None)(x)
    x = Activation("relu")(x)
    x = SpatialDropout1D(dropout_rate)(x)

    if residual.shape[-1] != filters:
        residual = Conv1D(filters=filters, kernel_size=1, padding="same")(residual)

    x = Add()([x, residual])
    return Activation("relu")(x)

tcn_inputs = Input(shape=(input_window, X_train_lstm.shape[2]), name="historical_inputs")
x = tcn_inputs
for dilation_rate in [1, 2, 4, 8, 16]:
    x = residual_tcn_block(x=x, filters=32, kernel_size=3, dilation_rate=dilation_rate, dropout_rate=0.10)

x = GlobalAveragePooling1D(name="temporal_pooling")(x)
x = Dense(64, activation="relu", name="dense_hidden")(x)
x = Dropout(0.20, name="dropout_hidden")(x)
tcn_outputs = Dense(forecast_horizon, name="forecast_output")(x)

tcn_model = Model(inputs=tcn_inputs, outputs=tcn_outputs, name="Dilated_Residual_TCN")
tcn_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

tcn_checkpoint_path = "best_tcn_model.keras"
tcn_early_stopping = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0001, restore_best_weights=True, verbose=1)
tcn_model_checkpoint = ModelCheckpoint(filepath=tcn_checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)

history_tcn = tcn_model.fit(
    X_train_lstm,
    y_train_scaled,
    validation_data=(X_val_lstm, y_val_scaled),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[tcn_early_stopping, tcn_model_checkpoint],
    verbose=1
)

best_tcn_model = tf.keras.models.load_model(tcn_checkpoint_path)
tcn_predictions_scaled = best_tcn_model.predict(X_test_lstm, verbose=1)
tcn_predictions = target_scaler.inverse_transform(tcn_predictions_scaled.reshape(-1, 1)).reshape(tcn_predictions_scaled.shape)
tcn_actuals = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)

tcn_results = {
    "model": "Dilated Residual TCN",
    "MAE": mae(tcn_actuals, tcn_predictions),
    "RMSE": rmse(tcn_actuals, tcn_predictions),
    "sMAPE": smape(tcn_actuals, tcn_predictions),
    "MASE": mase(tcn_actuals, tcn_predictions, mase_denominator),
    "Day+1_MAE": mae(tcn_actuals[:, 0], tcn_predictions[:, 0]),
    "Day+7_MAE": mae(tcn_actuals[:, 6], tcn_predictions[:, 6]),
    "Day+14_MAE": mae(tcn_actuals[:, 13], tcn_predictions[:, 13]),
    "Day+21_MAE": mae(tcn_actuals[:, 20], tcn_predictions[:, 20]),
    "Day+30_MAE": mae(tcn_actuals[:, 29], tcn_predictions[:, 29]),
    "parameter_count": best_tcn_model.count_params()
}
pd.DataFrame([tcn_results]).to_csv("tcn_test_results.csv", index=False)

# ============================================================
# TEMPORAL FUSION TRANSFORMER (TFT) STYLE FORECASTER
# ============================================================
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, TimeDistributed, Reshape

historical_observed_features = [
    "brent_crude_price", "cny_lkr", "eur_lkr", "gbp_lkr", "inr_lkr", "rub_lkr", "usd_lkr",
    "gdp_per_capita", "inflation_rate", "temperature", "humidity", "precipitation",
    "wind_speed", "is_rainy_day", "image_search", "web_search", "youtube_search",
    "cny_lkr_was_missing", "eur_lkr_was_missing", "gbp_lkr_was_missing",
    "inr_lkr_was_missing", "rub_lkr_was_missing", "usd_lkr_was_missing",
    "gdp_per_capita_was_missing", "inflation_rate_was_missing"
]

historical_feature_indices = [feature_columns.index(f) for f in historical_observed_features]
known_future_feature_indices = [feature_columns.index(f) for f in known_future_features]

X_train_tft_encoder = np.concatenate([
    X_train[:, :, historical_feature_indices],
    X_train[:, :, known_future_feature_indices],
    train_arrival_history_scaled[..., np.newaxis]
], axis=2)

X_val_tft_encoder = np.concatenate([
    X_val[:, :, historical_feature_indices],
    X_val[:, :, known_future_feature_indices],
    val_arrival_history_scaled[..., np.newaxis]
], axis=2)

X_test_tft_encoder = np.concatenate([
    X_test[:, :, historical_feature_indices],
    X_test[:, :, known_future_feature_indices],
    test_arrival_history_scaled[..., np.newaxis]
], axis=2)

def build_future_known_tensor(dataframe, known_future_features, target_date_windows):
    feature_lookup = dataframe.set_index("date")[known_future_features]
    future_known_windows = []
    for target_dates in target_date_windows:
        future_known_windows.append(feature_lookup.loc[pd.to_datetime(target_dates), known_future_features].to_numpy(dtype=np.float32))
    return np.array(future_known_windows, dtype=np.float32)

X_train_tft_decoder = build_future_known_tensor(train_full, known_future_features, train_target_dates)
X_val_tft_decoder = build_future_known_tensor(val_full, known_future_features, val_target_dates)
X_test_tft_decoder = build_future_known_tensor(test_full, known_future_features, test_target_dates)

encoder_inputs = Input(shape=(90, X_train_tft_encoder.shape[2]), name="encoder_historical_inputs")
decoder_inputs = Input(shape=(30, X_train_tft_decoder.shape[2]), name="decoder_known_future_inputs")

encoder_sequence, encoder_state_h, encoder_state_c = LSTM(units=64, return_sequences=True, return_state=True, dropout=0.20, name="encoder_lstm")(encoder_inputs)
decoder_sequence = LSTM(units=64, return_sequences=True, dropout=0.20, name="decoder_lstm")(decoder_inputs, initial_state=[encoder_state_h, encoder_state_c])

attention_output = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.20, name="temporal_cross_attention")(query=decoder_sequence, value=encoder_sequence, key=encoder_sequence)
x = Add(name="attention_residual")([decoder_sequence, attention_output])
x = LayerNormalization(name="attention_layer_norm")(x)

feed_forward = TimeDistributed(Dense(64, activation="relu"), name="time_distributed_dense")(x)
feed_forward = TimeDistributed(Dropout(0.20), name="time_distributed_dropout")(feed_forward)
x = Add(name="feed_forward_residual")([x, feed_forward])
x = LayerNormalization(name="feed_forward_layer_norm")(x)

forecast_sequence = TimeDistributed(Dense(1), name="daily_forecast")(x)
tft_outputs = Reshape((30,), name="thirty_day_forecast")(forecast_sequence)

tft_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=tft_outputs, name="TFT_Style_Forecaster")
tft_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

tft_checkpoint_path = "best_tft_style_model.keras"
tft_early_stopping = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0001, restore_best_weights=True, verbose=1)
tft_model_checkpoint = ModelCheckpoint(filepath=tft_checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)

history_tft = tft_model.fit(
    x=[X_train_tft_encoder, X_train_tft_decoder],
    y=y_train_scaled,
    validation_data=([X_val_tft_encoder, X_val_tft_decoder], y_val_scaled),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[tft_early_stopping, tft_model_checkpoint],
    verbose=1
)

best_tft_model = tf.keras.models.load_model(tft_checkpoint_path)
tft_predictions_scaled = best_tft_model.predict([X_test_tft_encoder, X_test_tft_decoder], verbose=1)
tft_predictions = target_scaler.inverse_transform(tft_predictions_scaled.reshape(-1, 1)).reshape(tft_predictions_scaled.shape)
tft_actuals = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)

tft_results = {
    "model": "TFT-Style Forecaster",
    "MAE": mae(tft_actuals, tft_predictions),
    "RMSE": rmse(tft_actuals, tft_predictions),
    "sMAPE": smape(tft_actuals, tft_predictions),
    "MASE": mase(tft_actuals, tft_predictions, mase_denominator),
    "Day+1_MAE": mae(tft_actuals[:, 0], tft_predictions[:, 0]),
    "Day+7_MAE": mae(tft_actuals[:, 6], tft_predictions[:, 6]),
    "Day+14_MAE": mae(tft_actuals[:, 13], tft_predictions[:, 13]),
    "Day+21_MAE": mae(tft_actuals[:, 20], tft_predictions[:, 20]),
    "Day+30_MAE": mae(tft_actuals[:, 29], tft_predictions[:, 29]),
    "parameter_count": best_tft_model.count_params()
}
pd.DataFrame([tft_results]).to_csv("tft_test_results.csv", index=False)

# ============================================================
# INFERENCE BENCHMARKING, VISUALIZATIONS, AND FINAL EVALUATION
# ============================================================
import time
import matplotlib.pyplot as plt

def measure_inference_time(model, model_inputs, repeats=3):
    model.predict(model_inputs, verbose=0)
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        model.predict(model_inputs, verbose=0)
        times.append(time.perf_counter() - start_time)
    return float(np.median(times))

lstm_inference_seconds = measure_inference_time(best_lstm_model, X_test_lstm)
gru_inference_seconds = measure_inference_time(best_gru_attention_model, X_test_lstm)
tcn_inference_seconds = measure_inference_time(best_tcn_model, X_test_lstm)
tft_inference_seconds = measure_inference_time(best_tft_model, [X_test_tft_encoder, X_test_tft_decoder])

final_comparison_df = pd.DataFrame([
    {**baseline_results, "parameter_count": 0, "approx_training_time_minutes": 0.0, "inference_time_seconds": 0.0},
    {**lstm_results, "approx_training_time_minutes": 2.0, "inference_time_seconds": lstm_inference_seconds},
    {**gru_attention_results, "approx_training_time_minutes": 5.0, "inference_time_seconds": gru_inference_seconds},
    {**tcn_results, "approx_training_time_minutes": 13.0, "inference_time_seconds": tcn_inference_seconds},
    {**tft_results, "approx_training_time_minutes": 6.0, "inference_time_seconds": tft_inference_seconds}
]).sort_values(by="MAE", ascending=True).reset_index(drop=True)

final_comparison_df.to_csv("final_model_comparison.csv", index=False)
print("Pipeline evaluation, visual analysis, and benchmark suite completed.")