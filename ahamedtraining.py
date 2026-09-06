# %%
from google.colab import files
import pandas as pd
import numpy as np
import os

uploaded = files.upload()

print("\nUploaded files:")
for filename in uploaded.keys():
    print(filename)

# Automatically find the uploaded CSV
csv_files = [f for f in uploaded.keys() if f.lower().endswith(".csv")]

if len(csv_files) != 1:
    raise ValueError(f"Expected exactly 1 CSV file, found: {csv_files}")

file_path = csv_files[0]

df = pd.read_csv(file_path)

print("\n========================================")
print("DATASET LOADED")
print("========================================")
print("File:", file_path)
print("Shape:", df.shape)
print("Rows:", len(df))
print("Columns:", len(df.columns))

# %%
import pandas as pd
import numpy as np

print("\n" + "="*80)
print("1. BASIC INFORMATION")
print("="*80)

print("Shape:", df.shape)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"{i:3d}: {col}")


print("\n" + "="*80)
print("2. DATA TYPES")
print("="*80)

print(df.dtypes.value_counts())
print("\nNon-numeric columns:")
print(df.select_dtypes(exclude=["number"]).columns.tolist())


print("\n" + "="*80)
print("3. DATE INFORMATION")
print("="*80)

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

if "date" in df.columns:
    temp_dates = pd.to_datetime(df["date"], errors="coerce")

    print("\nDate column:")
    print("First date:", temp_dates.min())
    print("Last date:", temp_dates.max())
    print("Invalid dates:", temp_dates.isna().sum())
    print("Duplicate dates:", temp_dates.duplicated().sum())

    date_diffs = temp_dates.sort_values().diff().dt.days

    print("\nDate gap distribution:")
    print(date_diffs.value_counts().sort_index().head(20))

    print("\nGaps greater than 1 day:")
    gaps = date_diffs[date_diffs > 1]
    print("Number of gaps:", len(gaps))

    if len(gaps) > 0:
        print(gaps.head(30))


print("\n" + "="*80)
print("4. MISSING VALUES")
print("="*80)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Total missing values:", df.isna().sum().sum())

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


print("\n" + "="*80)
print("5. DUPLICATES")
print("="*80)

print("Duplicate rows:", df.duplicated().sum())

if "date" in df.columns:
    print("Duplicate dates:", pd.to_datetime(df["date"], errors="coerce").duplicated().sum())


print("\n" + "="*80)
print("6. TARGET: ARRIVALS")
print("="*80)

if "arrivals" in df.columns:

    print(df["arrivals"].describe())

    print("\nTarget quantiles:")
    print(
        df["arrivals"].quantile(
            [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0]
        )
    )

    print("\nLowest arrivals:")
    print(df[["date", "arrivals"]].sort_values("arrivals").head(20))

    print("\nHighest arrivals:")
    print(df[["date", "arrivals"]].sort_values("arrivals", ascending=False).head(20))


print("\n" + "="*80)
print("7. UNIQUE VALUES / CONSTANT COLUMNS")
print("="*80)

unique_counts = df.nunique().sort_values()

print("Columns with <= 1 unique value:")
print(unique_counts[unique_counts <= 1])

print("\nColumns with <= 5 unique values:")
print(unique_counts[unique_counts <= 5])


print("\n" + "="*80)
print("8. POSSIBLE COVID / CRISIS COLUMNS")
print("="*80)

keywords = [
    "covid",
    "crisis",
    "pandemic",
    "lockdown",
    "recovery",
    "economic",
    "crisis",
    "event",
    "holiday"
]

possible_cols = []

for col in df.columns:
    col_lower = col.lower()

    if any(keyword in col_lower for keyword in keywords):
        possible_cols.append(col)

print("Possible relevant columns:")
for col in possible_cols:
    print("-", col)


print("\n" + "="*80)
print("9. ALL COLUMN SUMMARY")
print("="*80)

summary = pd.DataFrame({
    "column": df.columns,
    "dtype": [df[c].dtype for c in df.columns],
    "unique_values": [df[c].nunique() for c in df.columns],
    "missing": [df[c].isna().sum() for c in df.columns],
})

print(summary.to_string(index=False))


print("\n" + "="*80)
print("10. CORRELATION WITH ARRIVALS")
print("="*80)

numeric_df = df.select_dtypes(include=["number"])

if "arrivals" in numeric_df.columns:

    correlations = (
        numeric_df.corr()["arrivals"]
        .drop("arrivals")
        .sort_values(key=abs, ascending=False)
    )

    print(correlations.to_string())


print("\n" + "="*80)
print("11. DATA BY YEAR")
print("="*80)

if "date" in df.columns and "arrivals" in df.columns:

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp["year"] = temp["date"].dt.year

    yearly = temp.groupby("year")["arrivals"].agg(
        ["count", "mean", "std", "min", "max", "sum"]
    )

    print(yearly.to_string())


print("\n" + "="*80)
print("12. COVID / CRISIS PERIOD CHECK")
print("="*80)

if "date" in df.columns and "arrivals" in df.columns:

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")

    periods = {
        "2018-2019": ("2018-01-01", "2020-01-01"),
        "2020-2021": ("2020-01-01", "2022-01-01"),
        "2022": ("2022-01-01", "2023-01-01"),
        "2023": ("2023-01-01", "2024-01-01"),
        "2024": ("2024-01-01", "2025-01-01"),
        "2025+": ("2025-01-01", "2030-01-01")
    }

    for name, (start, end) in periods.items():

        subset = temp[
            (temp["date"] >= start) &
            (temp["date"] < end)
        ]

        if len(subset) > 0:
            print(
                f"{name:10s} | "
                f"Rows: {len(subset):4d} | "
                f"Mean: {subset['arrivals'].mean():10.2f} | "
                f"Min: {subset['arrivals'].min():10.2f} | "
                f"Max: {subset['arrivals'].max():10.2f}"
            )


print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

# %%
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Keep original dataframe untouched
# ---------------------------------------------------------
df_original = df.copy()

# Convert date only in the copy used for this experiment
df_adjusted = df.copy()

df_adjusted["date"] = pd.to_datetime(df_adjusted["date"])
df_adjusted = df_adjusted.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# 2. Remove COVID/crisis period
# ---------------------------------------------------------
crisis_start = pd.Timestamp("2020-01-01")
crisis_end   = pd.Timestamp("2022-12-31")

df_adjusted = df_adjusted[
    (df_adjusted["date"] < crisis_start) |
    (df_adjusted["date"] > crisis_end)
].copy()

df_adjusted = df_adjusted.reset_index(drop=True)

# ---------------------------------------------------------
# 3. Remove crisis-specific / useless columns
# ---------------------------------------------------------
columns_to_remove = [
    "covid_impact_factor",
    "crisis_impact_factor",
    "event_name_grouped_Esala Perahera (Limited due to COVID)",
    "exchange_rates_complete"
]

columns_to_remove = [
    col for col in columns_to_remove
    if col in df_adjusted.columns
]

df_adjusted = df_adjusted.drop(columns=columns_to_remove)

# ---------------------------------------------------------
# 4. Create block IDs
# ---------------------------------------------------------
# A new block starts whenever two consecutive retained
# dates are not exactly one day apart.

df_adjusted["block_id"] = (
    df_adjusted["date"].diff().dt.days.ne(1).cumsum()
)

# ---------------------------------------------------------
# 5. Convert original date temporarily for reporting
# ---------------------------------------------------------
original_dates = pd.to_datetime(df_original["date"])

# ---------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------
print("Original dataset:")
print(f"  Shape: {df_original.shape}")
print(
    f"  Date range: "
    f"{original_dates.min().date()} → {original_dates.max().date()}"
)

print("\nCrisis-adjusted dataset:")
print(f"  Shape: {df_adjusted.shape}")
print(
    f"  Date range: "
    f"{df_adjusted['date'].min().date()} → "
    f"{df_adjusted['date'].max().date()}"
)

print("\nRemoved columns:")
for col in columns_to_remove:
    print(f"  - {col}")

print(f"\nRemaining columns: {len(df_adjusted.columns)}")

print("\nDate ranges by block:")
for block_id, group in df_adjusted.groupby("block_id"):
    print(
        f"  Block {block_id}: "
        f"{group['date'].min().date()} → "
        f"{group['date'].max().date()} "
        f"({len(group):,} days)"
    )

print("\nYear counts:")
print(
    df_adjusted["date"]
    .dt.year
    .value_counts()
    .sort_index()
)

# %%
# =========================================================
# EXPERIMENT A — SEQUENCE CREATION
# Crisis-adjusted dataset
# =========================================================

LOOKBACK = 90
HORIZON = 30
TARGET = "arrivals"

# ---------------------------------------------------------
# 1. Features
# ---------------------------------------------------------

# Don't use date or block_id as model features
feature_columns = [
    col for col in df_adjusted.columns
    if col not in ["date", "block_id"]
]

print("Number of features:", len(feature_columns))
print("Target:", TARGET)

# ---------------------------------------------------------
# 2. Create sequences separately inside each block
# ---------------------------------------------------------

X_list = []
y_list = []
target_dates_list = []

for block_id, block in df_adjusted.groupby("block_id"):

    block = block.sort_values("date").reset_index(drop=True)

    X_values = block[feature_columns].values
    y_values = block[TARGET].values
    dates = block["date"].values

    # Need 90 days history + 30 days future
    max_start = len(block) - LOOKBACK - HORIZON + 1

    for i in range(max_start):

        # Previous 90 days
        X_window = X_values[i:i + LOOKBACK]

        # Following 30 days
        y_window = y_values[
            i + LOOKBACK:
            i + LOOKBACK + HORIZON
        ]

        # Date of the FIRST prediction
        target_start_date = dates[i + LOOKBACK]

        X_list.append(X_window)
        y_list.append(y_window)
        target_dates_list.append(target_start_date)

# Convert to numpy
X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=np.float32)
target_dates = pd.to_datetime(target_dates_list)

print("\nSequence shapes:")
print("X_all:", X_all.shape)
print("y_all:", y_all.shape)
print("target_dates:", target_dates.shape)

# ---------------------------------------------------------
# 3. Chronological split based on TARGET dates
# ---------------------------------------------------------

train_mask = (
    (target_dates < pd.Timestamp("2020-01-01")) |
    (
        (target_dates >= pd.Timestamp("2023-01-01")) &
        (target_dates < pd.Timestamp("2024-01-01"))
    )
)

val_mask = (
    (target_dates >= pd.Timestamp("2024-01-01")) &
    (target_dates < pd.Timestamp("2025-01-01"))
)

test_mask = (
    (target_dates >= pd.Timestamp("2025-01-01")) &
    (target_dates <= pd.Timestamp("2026-07-31"))
)

X_train = X_all[train_mask]
y_train = y_all[train_mask]

X_val = X_all[val_mask]
y_val = y_all[val_mask]

X_test = X_all[test_mask]
y_test = y_all[test_mask]

dates_train = target_dates[train_mask]
dates_val = target_dates[val_mask]
dates_test = target_dates[test_mask]

# ---------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------

print("\nFINAL SPLIT")
print("=" * 50)

print("\nTRAIN")
print("X:", X_train.shape)
print("y:", y_train.shape)
print(
    "Target dates:",
    dates_train.min().date(),
    "→",
    dates_train.max().date()
)

print("\nVALIDATION")
print("X:", X_val.shape)
print("y:", y_val.shape)
print(
    "Target dates:",
    dates_val.min().date(),
    "→",
    dates_val.max().date()
)

print("\nTEST")
print("X:", X_test.shape)
print("y:", y_test.shape)
print(
    "Target dates:",
    dates_test.min().date(),
    "→",
    dates_test.max().date()
)

# ---------------------------------------------------------
# 5. Verify that no sequence crosses the 2019 → 2023 gap
# ---------------------------------------------------------

print("\nSequence integrity check:")

invalid_sequences = 0

for block_id, block in df_adjusted.groupby("block_id"):

    block = block.sort_values("date").reset_index(drop=True)

    # Check whether the block itself is continuous
    date_diffs = block["date"].diff().dropna().dt.days

    if not (date_diffs == 1).all():
        invalid_sequences += 1

print(
    "Blocks containing internal date gaps:",
    invalid_sequences
)

print("\nDone.")

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

# =========================================================
# TRAINING-ONLY SCALING
# =========================================================

# ---------------------------------------------------------
# 1. Create feature scaler
# ---------------------------------------------------------

feature_scaler = StandardScaler()

# X_train shape:
# (samples, 90 days, 105 features)
#
# StandardScaler expects 2D data, so flatten the first
# two dimensions temporarily.

X_train_2d = X_train.reshape(-1, X_train.shape[-1])

# Fit ONLY on training data
feature_scaler.fit(X_train_2d)

# ---------------------------------------------------------
# 2. Transform all X splits
# ---------------------------------------------------------

X_train_scaled = feature_scaler.transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_val_scaled = feature_scaler.transform(
    X_val.reshape(-1, X_val.shape[-1])
).reshape(X_val.shape)

X_test_scaled = feature_scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

# ---------------------------------------------------------
# 3. Target scaler
# ---------------------------------------------------------

target_scaler = StandardScaler()

# y_train contains 30 future values per sample.
# Fit using ONLY training target values.

target_scaler.fit(
    y_train.reshape(-1, 1)
)

# Transform targets
y_train_scaled = target_scaler.transform(
    y_train.reshape(-1, 1)
).reshape(y_train.shape)

y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
).reshape(y_val.shape)

y_test_scaled = target_scaler.transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

# ---------------------------------------------------------
# 4. Print shapes
# ---------------------------------------------------------

print("Scaled shapes")
print("=" * 50)

print("X_train_scaled:", X_train_scaled.shape)
print("X_val_scaled:  ", X_val_scaled.shape)
print("X_test_scaled: ", X_test_scaled.shape)

print("\ny_train_scaled:", y_train_scaled.shape)
print("y_val_scaled:  ", y_val_scaled.shape)
print("y_test_scaled: ", y_test_scaled.shape)

# ---------------------------------------------------------
# 5. Scaling sanity check
# ---------------------------------------------------------

print("\nFeature scaler:")
print("Training feature means (first 5):")
print(feature_scaler.mean_[:5])

print("\nTraining feature scales (first 5):")
print(feature_scaler.scale_[:5])

print("\nTarget scaler:")
print("Mean:", target_scaler.mean_[0])
print("Scale:", target_scaler.scale_[0])

# ---------------------------------------------------------
# 6. Verify inverse transformation
# ---------------------------------------------------------

sample_scaled = y_test_scaled[0]

sample_original = target_scaler.inverse_transform(
    sample_scaled.reshape(-1, 1)
).flatten()

print("\nTarget inverse-transform check:")
print("Original first 5:", y_test[0][:5])
print("Recovered first 5:", sample_original[:5])

print("\nDone.")

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# EXPERIMENT A — LSTM
# Crisis-adjusted dataset
# =========================================================

# Reproducibility
tf.keras.utils.set_random_seed(42)

# ---------------------------------------------------------
# 1. Build model
# ---------------------------------------------------------

lstm_model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(LOOKBACK, X_train_scaled.shape[-1])
    ),

    Dropout(0.20),

    LSTM(
        32,
        return_sequences=False
    ),

    Dense(64, activation="relu"),

    Dropout(0.20),

    Dense(HORIZON)
])

# ---------------------------------------------------------
# 2. Compile
# ---------------------------------------------------------

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

# ---------------------------------------------------------
# 3. Show architecture
# ---------------------------------------------------------

lstm_model.summary()

# ---------------------------------------------------------
# 4. Early stopping
# ---------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------

history_lstm_adjusted = lstm_model.fit(
    X_train_scaled,
    y_train_scaled,

    validation_data=(
        X_val_scaled,
        y_val_scaled
    ),

    epochs=100,
    batch_size=32,

    callbacks=[early_stopping],

    verbose=1
)

# ---------------------------------------------------------
# 6. Training summary
# ---------------------------------------------------------

best_epoch = np.argmin(history_lstm_adjusted.history["val_loss"]) + 1
best_val_loss = min(history_lstm_adjusted.history["val_loss"])

print("\nTraining complete.")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)

print("\nFinal model parameters:",
      lstm_model.count_params())

# %%
# =========================================================
# EXPERIMENT A — LSTM TEST EVALUATION
# =========================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------------------------------------------------------
# 1. Predict
# ---------------------------------------------------------

y_pred_scaled = lstm_model.predict(
    X_test_scaled,
    verbose=1
)

# ---------------------------------------------------------
# 2. Convert predictions back to original arrivals scale
# ---------------------------------------------------------

y_pred_lstm_adjusted = target_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(y_pred_scaled.shape)

# Actual values are already in original scale
y_actual_lstm_adjusted = y_test.copy()

# ---------------------------------------------------------
# 3. MAE
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_actual_lstm_adjusted.flatten(),
    y_pred_lstm_adjusted.flatten()
)

# ---------------------------------------------------------
# 4. RMSE
# ---------------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_actual_lstm_adjusted.flatten(),
        y_pred_lstm_adjusted.flatten()
    )
)

# ---------------------------------------------------------
# 5. sMAPE
# ---------------------------------------------------------

smape = (
    100
    * np.mean(
        2 * np.abs(
            y_pred_lstm_adjusted - y_actual_lstm_adjusted
        )
        /
        (
            np.abs(y_actual_lstm_adjusted)
            + np.abs(y_pred_lstm_adjusted)
            + 1e-8
        )
    )
)

# ---------------------------------------------------------
# 6. 7-day MASE
# ---------------------------------------------------------

# Seasonal naive benchmark:
# prediction = value from 7 days earlier

train_arrivals = df_adjusted.loc[
    df_adjusted["date"] < pd.Timestamp("2020-01-01"),
    TARGET
].values

# Add the 2023 training period
train_2023 = df_adjusted.loc[
    (df_adjusted["date"] >= pd.Timestamp("2023-01-01")) &
    (df_adjusted["date"] < pd.Timestamp("2024-01-01")),
    TARGET
].values

mase_training_values = np.concatenate([
    train_arrivals,
    train_2023
])

mase_denominator = np.mean(
    np.abs(
        mase_training_values[7:]
        - mase_training_values[:-7]
    )
)

mase = np.mean(
    np.abs(
        y_actual_lstm_adjusted - y_pred_lstm_adjusted
    )
) / mase_denominator

# ---------------------------------------------------------
# 7. Horizon-specific MAE
# ---------------------------------------------------------

horizons = [0, 6, 13, 20, 29]

horizon_mae = {}

for h in horizons:
    horizon_mae[h + 1] = mean_absolute_error(
        y_actual_lstm_adjusted[:, h],
        y_pred_lstm_adjusted[:, h]
    )

# ---------------------------------------------------------
# 8. Forecast compression diagnostics
# ---------------------------------------------------------

actual_mean = y_actual_lstm_adjusted.mean()
pred_mean = y_pred_lstm_adjusted.mean()

actual_std = y_actual_lstm_adjusted.std()
pred_std = y_pred_lstm_adjusted.std()

actual_min = y_actual_lstm_adjusted.min()
actual_max = y_actual_lstm_adjusted.max()

pred_min = y_pred_lstm_adjusted.min()
pred_max = y_pred_lstm_adjusted.max()

# ---------------------------------------------------------
# 9. Print results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CRISIS-ADJUSTED LSTM — TEST RESULTS")
print("=" * 60)

print(f"\nMAE:    {mae:.2f}")
print(f"RMSE:   {rmse:.2f}")
print(f"sMAPE:  {smape:.2f}%")
print(f"MASE:   {mase:.4f}")

print("\nHorizon MAE:")
for h, value in horizon_mae.items():
    print(f"  Day +{h:2d}: {value:.2f}")

print("\nForecast distribution:")
print(f"  Actual mean: {actual_mean:.2f}")
print(f"  Pred mean:   {pred_mean:.2f}")

print(f"\n  Actual std:  {actual_std:.2f}")
print(f"  Pred std:    {pred_std:.2f}")

print(f"\n  Actual range: {actual_min:.0f} → {actual_max:.0f}")
print(f"  Pred range:   {pred_min:.0f} → {pred_max:.0f}")

print("\nFirst 30-day forecast:")
print("Actual:")
print(np.round(y_actual_lstm_adjusted[0]).astype(int))

print("\nPredicted:")
print(np.round(y_pred_lstm_adjusted[0]).astype(int))

print("\n" + "=" * 60)

# %%
# =========================================================
# EXPERIMENT B — ADD EXPLICIT WEEKLY / ROLLING FEATURES
# =========================================================

df_weekly = df_adjusted.copy()

# ---------------------------------------------------------
# 1. Create historical arrival features within each block
# ---------------------------------------------------------

for lag in [7, 14, 21, 28]:
    df_weekly[f"lag_{lag}"] = (
        df_weekly
        .groupby("block_id")[TARGET]
        .shift(lag)
    )

# Rolling features must use ONLY previous values.
# shift(1) prevents today's target from entering
# today's rolling statistics.

df_weekly["rolling_mean_7"] = (
    df_weekly
    .groupby("block_id")[TARGET]
    .shift(1)
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)

df_weekly["rolling_mean_14"] = (
    df_weekly
    .groupby("block_id")[TARGET]
    .shift(1)
    .rolling(14)
    .mean()
    .reset_index(level=0, drop=True)
)

df_weekly["rolling_mean_30"] = (
    df_weekly
    .groupby("block_id")[TARGET]
    .shift(1)
    .rolling(30)
    .mean()
    .reset_index(level=0, drop=True)
)

df_weekly["rolling_std_7"] = (
    df_weekly
    .groupby("block_id")[TARGET]
    .shift(1)
    .rolling(7)
    .std()
    .reset_index(level=0, drop=True)
)

df_weekly["rolling_std_30"] = (
    df_weekly
    .groupby("block_id")[TARGET]
    .shift(1)
    .rolling(30)
    .std()
    .reset_index(level=0, drop=True)
)

# ---------------------------------------------------------
# 2. Remove rows where the new features aren't available
# ---------------------------------------------------------

new_features = [
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_30"
]

print("Missing values in new features BEFORE removal:")
print(df_weekly[new_features].isna().sum())

# We don't want to fill these artificially.
df_weekly = df_weekly.dropna(
    subset=new_features
).reset_index(drop=True)

# ---------------------------------------------------------
# 3. Recreate block IDs
# ---------------------------------------------------------

df_weekly["block_id"] = (
    df_weekly["date"].diff().dt.days.ne(1).cumsum()
)

# ---------------------------------------------------------
# 4. Check
# ---------------------------------------------------------

print("\nExperiment B dataset:")
print("Shape:", df_weekly.shape)

print("\nNew features:")
for feature in new_features:
    print(" ", feature)

print("\nDate ranges by block:")

for block_id, group in df_weekly.groupby("block_id"):
    print(
        f"  Block {block_id}: "
        f"{group['date'].min().date()} → "
        f"{group['date'].max().date()} "
        f"({len(group):,} days)"
    )

print("\nRemaining missing values:")
print(df_weekly[new_features].isna().sum().sum())

print("\nDone.")

# %%
# =========================================================
# EXPERIMENT B — SEQUENCE CREATION
# =========================================================

LOOKBACK = 90
HORIZON = 30
TARGET = "arrivals"

# ---------------------------------------------------------
# 1. Features
# ---------------------------------------------------------

feature_columns_B = [
    col for col in df_weekly.columns
    if col not in ["date", "block_id"]
]

print("Number of features:", len(feature_columns_B))
print("Target:", TARGET)

# ---------------------------------------------------------
# 2. Create sequences within each continuous block
# ---------------------------------------------------------

X_list_B = []
y_list_B = []
target_dates_list_B = []

for block_id, block in df_weekly.groupby("block_id"):

    block = block.sort_values("date").reset_index(drop=True)

    X_values = block[feature_columns_B].values
    y_values = block[TARGET].values
    dates = block["date"].values

    max_start = len(block) - LOOKBACK - HORIZON + 1

    for i in range(max_start):

        X_window = X_values[i:i + LOOKBACK]

        y_window = y_values[
            i + LOOKBACK:
            i + LOOKBACK + HORIZON
        ]

        target_start_date = dates[i + LOOKBACK]

        X_list_B.append(X_window)
        y_list_B.append(y_window)
        target_dates_list_B.append(target_start_date)

# Convert to arrays
X_all_B = np.array(X_list_B, dtype=np.float32)
y_all_B = np.array(y_list_B, dtype=np.float32)
target_dates_B = pd.to_datetime(target_dates_list_B)

print("\nSequence shapes:")
print("X_all_B:", X_all_B.shape)
print("y_all_B:", y_all_B.shape)
print("target_dates_B:", target_dates_B.shape)

# ---------------------------------------------------------
# 3. Chronological split
# ---------------------------------------------------------

train_mask_B = (
    (target_dates_B < pd.Timestamp("2020-01-01")) |
    (
        (target_dates_B >= pd.Timestamp("2023-01-01")) &
        (target_dates_B < pd.Timestamp("2024-01-01"))
    )
)

val_mask_B = (
    (target_dates_B >= pd.Timestamp("2024-01-01")) &
    (target_dates_B < pd.Timestamp("2025-01-01"))
)

test_mask_B = (
    (target_dates_B >= pd.Timestamp("2025-01-01")) &
    (target_dates_B <= pd.Timestamp("2026-07-31"))
)

X_train_B = X_all_B[train_mask_B]
y_train_B = y_all_B[train_mask_B]

X_val_B = X_all_B[val_mask_B]
y_val_B = y_all_B[val_mask_B]

X_test_B = X_all_B[test_mask_B]
y_test_B = y_all_B[test_mask_B]

dates_train_B = target_dates_B[train_mask_B]
dates_val_B = target_dates_B[val_mask_B]
dates_test_B = target_dates_B[test_mask_B]

# ---------------------------------------------------------
# 4. Print split results
# ---------------------------------------------------------

print("\nFINAL SPLIT — EXPERIMENT B")
print("=" * 50)

print("\nTRAIN")
print("X:", X_train_B.shape)
print("y:", y_train_B.shape)
print(
    "Target dates:",
    dates_train_B.min().date(),
    "→",
    dates_train_B.max().date()
)

print("\nVALIDATION")
print("X:", X_val_B.shape)
print("y:", y_val_B.shape)
print(
    "Target dates:",
    dates_val_B.min().date(),
    "→",
    dates_val_B.max().date()
)

print("\nTEST")
print("X:", X_test_B.shape)
print("y:", y_test_B.shape)
print(
    "Target dates:",
    dates_test_B.min().date(),
    "→",
    dates_test_B.max().date()
)

# ---------------------------------------------------------
# 5. Check sequence integrity
# ---------------------------------------------------------

invalid_blocks_B = 0

for block_id, block in df_weekly.groupby("block_id"):

    block = block.sort_values("date").reset_index(drop=True)

    date_diffs = block["date"].diff().dropna().dt.days

    if not (date_diffs == 1).all():
        invalid_blocks_B += 1

print("\nSequence integrity check:")
print(
    "Blocks containing internal date gaps:",
    invalid_blocks_B
)

print("\nDone.")

# %%
from sklearn.preprocessing import StandardScaler

# =========================================================
# EXPERIMENT B — TRAINING-ONLY SCALING
# =========================================================

# ---------------------------------------------------------
# 1. Feature scaler
# ---------------------------------------------------------

feature_scaler_B = StandardScaler()

X_train_B_2d = X_train_B.reshape(
    -1,
    X_train_B.shape[-1]
)

# FIT ONLY ON TRAINING DATA
feature_scaler_B.fit(X_train_B_2d)

# ---------------------------------------------------------
# 2. Transform X
# ---------------------------------------------------------

X_train_scaled_B = feature_scaler_B.transform(
    X_train_B.reshape(-1, X_train_B.shape[-1])
).reshape(X_train_B.shape)

X_val_scaled_B = feature_scaler_B.transform(
    X_val_B.reshape(-1, X_val_B.shape[-1])
).reshape(X_val_B.shape)

X_test_scaled_B = feature_scaler_B.transform(
    X_test_B.reshape(-1, X_test_B.shape[-1])
).reshape(X_test_B.shape)

# ---------------------------------------------------------
# 3. Target scaler
# ---------------------------------------------------------

target_scaler_B = StandardScaler()

# FIT ONLY ON TRAINING TARGETS
target_scaler_B.fit(
    y_train_B.reshape(-1, 1)
)

# Transform targets
y_train_scaled_B = target_scaler_B.transform(
    y_train_B.reshape(-1, 1)
).reshape(y_train_B.shape)

y_val_scaled_B = target_scaler_B.transform(
    y_val_B.reshape(-1, 1)
).reshape(y_val_B.shape)

y_test_scaled_B = target_scaler_B.transform(
    y_test_B.reshape(-1, 1)
).reshape(y_test_B.shape)

# ---------------------------------------------------------
# 4. Print shapes
# ---------------------------------------------------------

print("Experiment B — scaled shapes")
print("=" * 50)

print("X_train_scaled_B:", X_train_scaled_B.shape)
print("X_val_scaled_B:  ", X_val_scaled_B.shape)
print("X_test_scaled_B: ", X_test_scaled_B.shape)

print("\ny_train_scaled_B:", y_train_scaled_B.shape)
print("y_val_scaled_B:  ", y_val_scaled_B.shape)
print("y_test_scaled_B: ", y_test_scaled_B.shape)

# ---------------------------------------------------------
# 5. Target scaler information
# ---------------------------------------------------------

print("\nTarget scaler B:")
print("Mean:", target_scaler_B.mean_[0])
print("Scale:", target_scaler_B.scale_[0])

# ---------------------------------------------------------
# 6. Inverse transformation check
# ---------------------------------------------------------

sample_scaled_B = y_test_scaled_B[0]

sample_original_B = target_scaler_B.inverse_transform(
    sample_scaled_B.reshape(-1, 1)
).flatten()

print("\nTarget inverse-transform check:")
print("Original first 5:", y_test_B[0][:5])
print("Recovered first 5:", sample_original_B[:5])

print("\nDone.")

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# EXPERIMENT B — LSTM
# Crisis-adjusted + weekly/rolling features
# =========================================================

tf.keras.utils.set_random_seed(42)

# ---------------------------------------------------------
# 1. Build model
# ---------------------------------------------------------

lstm_model_B = Sequential([
    Input(shape=(LOOKBACK, X_train_scaled_B.shape[-1])),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.20),

    LSTM(
        32,
        return_sequences=False
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(HORIZON)
])

# ---------------------------------------------------------
# 2. Compile
# ---------------------------------------------------------

lstm_model_B.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)

# ---------------------------------------------------------
# 3. Show architecture
# ---------------------------------------------------------

lstm_model_B.summary()

# ---------------------------------------------------------
# 4. Early stopping
# ---------------------------------------------------------

early_stopping_B = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------

history_lstm_B = lstm_model_B.fit(
    X_train_scaled_B,
    y_train_scaled_B,

    validation_data=(
        X_val_scaled_B,
        y_val_scaled_B
    ),

    epochs=100,
    batch_size=32,

    callbacks=[early_stopping_B],

    verbose=1
)

# ---------------------------------------------------------
# 6. Training summary
# ---------------------------------------------------------

best_epoch_B = (
    np.argmin(history_lstm_B.history["val_loss"]) + 1
)

best_val_loss_B = min(
    history_lstm_B.history["val_loss"]
)

print("\nTraining complete.")
print("Best epoch:", best_epoch_B)
print("Best validation loss:", best_val_loss_B)
print(
    "Final model parameters:",
    lstm_model_B.count_params()
)

# %%
# =========================================================
# EXPERIMENT B — LSTM TEST EVALUATION
# =========================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------------------------------------------------------
# 1. Predict
# ---------------------------------------------------------

y_pred_scaled_B = lstm_model_B.predict(
    X_test_scaled_B,
    verbose=1
)

# ---------------------------------------------------------
# 2. Inverse transform
# ---------------------------------------------------------

y_pred_lstm_B = target_scaler_B.inverse_transform(
    y_pred_scaled_B.reshape(-1, 1)
).reshape(y_pred_scaled_B.shape)

y_actual_lstm_B = y_test_B.copy()

# ---------------------------------------------------------
# 3. MAE
# ---------------------------------------------------------

mae_B = mean_absolute_error(
    y_actual_lstm_B.flatten(),
    y_pred_lstm_B.flatten()
)

# ---------------------------------------------------------
# 4. RMSE
# ---------------------------------------------------------

rmse_B = np.sqrt(
    mean_squared_error(
        y_actual_lstm_B.flatten(),
        y_pred_lstm_B.flatten()
    )
)

# ---------------------------------------------------------
# 5. sMAPE
# ---------------------------------------------------------

smape_B = (
    100
    * np.mean(
        2 * np.abs(
            y_pred_lstm_B - y_actual_lstm_B
        )
        /
        (
            np.abs(y_actual_lstm_B)
            + np.abs(y_pred_lstm_B)
            + 1e-8
        )
    )
)

# ---------------------------------------------------------
# 6. 7-day MASE
# ---------------------------------------------------------

train_arrivals_B = df_weekly.loc[
    df_weekly["date"] < pd.Timestamp("2020-01-01"),
    TARGET
].values

train_2023_B = df_weekly.loc[
    (df_weekly["date"] >= pd.Timestamp("2023-01-01")) &
    (df_weekly["date"] < pd.Timestamp("2024-01-01")),
    TARGET
].values

mase_training_values_B = np.concatenate([
    train_arrivals_B,
    train_2023_B
])

mase_denominator_B = np.mean(
    np.abs(
        mase_training_values_B[7:]
        - mase_training_values_B[:-7]
    )
)

mase_B = (
    np.mean(
        np.abs(
            y_actual_lstm_B - y_pred_lstm_B
        )
    )
    / mase_denominator_B
)

# ---------------------------------------------------------
# 7. Horizon MAE
# ---------------------------------------------------------

horizons = [0, 6, 13, 20, 29]

horizon_mae_B = {}

for h in horizons:
    horizon_mae_B[h + 1] = mean_absolute_error(
        y_actual_lstm_B[:, h],
        y_pred_lstm_B[:, h]
    )

# ---------------------------------------------------------
# 8. Forecast distribution
# ---------------------------------------------------------

actual_mean_B = y_actual_lstm_B.mean()
pred_mean_B = y_pred_lstm_B.mean()

actual_std_B = y_actual_lstm_B.std()
pred_std_B = y_pred_lstm_B.std()

actual_min_B = y_actual_lstm_B.min()
actual_max_B = y_actual_lstm_B.max()

pred_min_B = y_pred_lstm_B.min()
pred_max_B = y_pred_lstm_B.max()

# ---------------------------------------------------------
# 9. Results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EXPERIMENT B — CRISIS-ADJUSTED + WEEKLY FEATURES")
print("LSTM TEST RESULTS")
print("=" * 60)

print(f"\nMAE:    {mae_B:.2f}")
print(f"RMSE:   {rmse_B:.2f}")
print(f"sMAPE:  {smape_B:.2f}%")
print(f"MASE:   {mase_B:.4f}")

print("\nHorizon MAE:")
for h, value in horizon_mae_B.items():
    print(f"  Day +{h:2d}: {value:.2f}")

print("\nForecast distribution:")
print(f"  Actual mean: {actual_mean_B:.2f}")
print(f"  Pred mean:   {pred_mean_B:.2f}")

print(f"\n  Actual std:  {actual_std_B:.2f}")
print(f"  Pred std:    {pred_std_B:.2f}")

print(f"\n  Actual range: {actual_min_B:.0f} → {actual_max_B:.0f}")
print(f"  Pred range:   {pred_min_B:.0f} → {pred_max_B:.0f}")

print("\nFirst 30-day forecast:")

print("Actual:")
print(
    np.round(y_actual_lstm_B[0]).astype(int)
)

print("\nPredicted:")
print(
    np.round(y_pred_lstm_B[0]).astype(int)
)

print("\n" + "=" * 60)

# %%
# ============================================================
# EXPERIMENT B — CORRECTED SEQUENCE SPLIT
# ============================================================

LOOKBACK = 90
HORIZON = 30
TARGET = "arrivals"

def create_sequences_corrected(df, feature_cols):
    X, y, target_dates = [], [], []

    for block_id, block in df.groupby("block_id", sort=True):
        block = block.sort_values("date").reset_index(drop=True)

        features = block[feature_cols].values
        target = block[TARGET].values
        dates = block["date"].values

        for i in range(LOOKBACK, len(block) - HORIZON + 1):
            X.append(features[i-LOOKBACK:i])
            y.append(target[i:i+HORIZON])
            target_dates.append(dates[i])

    return (
        np.array(X),
        np.array(y),
        pd.to_datetime(target_dates)
    )


# ------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------

feature_cols_B = [
    c for c in df_weekly.columns
    if c not in ["date", "block_id"]
]

print("Number of features:", len(feature_cols_B))


# ------------------------------------------------------------
# Create all sequences
# ------------------------------------------------------------

X_all_B, y_all_B, target_dates_B = create_sequences_corrected(
    df_weekly,
    feature_cols_B
)

print("All sequences:")
print("X:", X_all_B.shape)
print("y:", y_all_B.shape)


# ------------------------------------------------------------
# Split based on COMPLETE 30-day target horizon
# ------------------------------------------------------------

train_start = pd.Timestamp("2010-01-01")
train_end   = pd.Timestamp("2023-12-31")

val_start   = pd.Timestamp("2024-01-01")
val_end     = pd.Timestamp("2024-12-31")

test_start  = pd.Timestamp("2025-01-01")
test_end    = pd.Timestamp("2026-07-31")


# Since target_dates_B is the FIRST day of the forecast horizon,
# require the final target day to remain inside the split.

target_end_dates_B = target_dates_B + pd.Timedelta(days=HORIZON - 1)


train_mask_B = (
    (target_dates_B >= train_start) &
    (target_end_dates_B <= train_end)
)

val_mask_B = (
    (target_dates_B >= val_start) &
    (target_end_dates_B <= val_end)
)

test_mask_B = (
    (target_dates_B >= test_start) &
    (target_end_dates_B <= test_end)
)


X_train_B = X_all_B[train_mask_B]
y_train_B = y_all_B[train_mask_B]

X_val_B = X_all_B[val_mask_B]
y_val_B = y_all_B[val_mask_B]

X_test_B = X_all_B[test_mask_B]
y_test_B = y_all_B[test_mask_B]

dates_train_B = target_dates_B[train_mask_B]
dates_val_B = target_dates_B[val_mask_B]
dates_test_B = target_dates_B[test_mask_B]


print("\n============================================================")
print("CORRECTED SPLIT")
print("============================================================")

print("\nTrain:")
print("X:", X_train_B.shape)
print("y:", y_train_B.shape)
print("Target start:", dates_train_B.min())
print("Target start:", dates_train_B.max())
print("Target end:", dates_train_B.max() + pd.Timedelta(days=HORIZON-1))

print("\nValidation:")
print("X:", X_val_B.shape)
print("y:", y_val_B.shape)
print("Target start:", dates_val_B.min())
print("Target start:", dates_val_B.max())
print("Target end:", dates_val_B.max() + pd.Timedelta(days=HORIZON-1))

print("\nTest:")
print("X:", X_test_B.shape)
print("y:", y_test_B.shape)
print("Target start:", dates_test_B.min())
print("Target start:", dates_test_B.max())
print("Target end:", dates_test_B.max() + pd.Timedelta(days=HORIZON-1))

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================
# SCALE CORRECTED EXPERIMENT B
# ============================================================

# ------------------------------------------------------------
# Feature scaler
# Fit ONLY on training data
# ------------------------------------------------------------

feature_scaler_B_corrected = StandardScaler()

feature_scaler_B_corrected.fit(
    X_train_B.reshape(-1, X_train_B.shape[-1])
)


X_train_scaled_B = feature_scaler_B_corrected.transform(
    X_train_B.reshape(-1, X_train_B.shape[-1])
).reshape(X_train_B.shape)

X_val_scaled_B = feature_scaler_B_corrected.transform(
    X_val_B.reshape(-1, X_val_B.shape[-1])
).reshape(X_val_B.shape)

X_test_scaled_B = feature_scaler_B_corrected.transform(
    X_test_B.reshape(-1, X_test_B.shape[-1])
).reshape(X_test_B.shape)


# ------------------------------------------------------------
# Target scaler
# Fit ONLY on training targets
# ------------------------------------------------------------

target_scaler_B_corrected = StandardScaler()

target_scaler_B_corrected.fit(
    y_train_B.reshape(-1, 1)
)


y_train_scaled_B = target_scaler_B_corrected.transform(
    y_train_B.reshape(-1, 1)
).reshape(y_train_B.shape)

y_val_scaled_B = target_scaler_B_corrected.transform(
    y_val_B.reshape(-1, 1)
).reshape(y_val_B.shape)

y_test_scaled_B = target_scaler_B_corrected.transform(
    y_test_B.reshape(-1, 1)
).reshape(y_test_B.shape)


# ============================================================
# CHECK
# ============================================================

print("============================================================")
print("CORRECTED EXPERIMENT B — SCALING")
print("============================================================")

print("\nFeature arrays:")
print("X_train:", X_train_scaled_B.shape)
print("X_val:  ", X_val_scaled_B.shape)
print("X_test: ", X_test_scaled_B.shape)

print("\nTarget arrays:")
print("y_train:", y_train_scaled_B.shape)
print("y_val:  ", y_val_scaled_B.shape)
print("y_test: ", y_test_scaled_B.shape)

print("\nTarget scaler:")
print("Mean :", target_scaler_B_corrected.mean_[0])
print("Scale:", target_scaler_B_corrected.scale_[0])

# Verify inverse transformation
check = target_scaler_B_corrected.inverse_transform(
    y_test_scaled_B[0].reshape(-1, 1)
).flatten()

print("\nFirst 5 original test targets:")
print(y_test_B[0][:5])

print("\nFirst 5 after inverse scaling:")
print(np.round(check[:5], 3))

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
tf.keras.utils.set_random_seed(42)

# ============================================================
# CORRECTED EXPERIMENT B — LSTM
# ============================================================

model_lstm_B = Sequential([
    Input(shape=(LOOKBACK, X_train_scaled_B.shape[-1])),

    LSTM(64, return_sequences=True),
    Dropout(0.20),

    LSTM(32, return_sequences=False),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(HORIZON)
])

model_lstm_B.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model_lstm_B.summary()


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


history_lstm_B = model_lstm_B.fit(
    X_train_scaled_B,
    y_train_scaled_B,
    validation_data=(X_val_scaled_B, y_val_scaled_B),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ============================================================
# CORRECTED EXPERIMENT B — LSTM EVALUATION
# ============================================================

# Predict
y_pred_scaled_B_corrected = model_lstm_B.predict(
    X_test_scaled_B,
    verbose=1
)

# Inverse transform
y_pred_B_corrected = target_scaler_B_corrected.inverse_transform(
    y_pred_scaled_B_corrected.reshape(-1, 1)
).reshape(y_pred_scaled_B_corrected.shape)


# ------------------------------------------------------------
# Basic metrics
# ------------------------------------------------------------

y_true = y_test_B.astype(float)
y_pred = y_pred_B_corrected.astype(float)

mae = mean_absolute_error(
    y_true.flatten(),
    y_pred.flatten()
)

rmse = np.sqrt(
    mean_squared_error(
        y_true.flatten(),
        y_pred.flatten()
    )
)

smape = np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100


# ------------------------------------------------------------
# Correct 7-day MASE denominator
#
# Calculate seasonal-naive differences INSIDE each continuous
# training block only. This prevents the 2019 -> 2023 gap from
# entering the denominator.
# ------------------------------------------------------------

seasonal_errors = []

for block_id, block in df_weekly.groupby("block_id", sort=True):

    block = block.sort_values("date")

    # Only training period
    block = block[
        block["date"] <= pd.Timestamp("2023-12-31")
    ]

    arrivals = block[TARGET].values.astype(float)

    if len(arrivals) > 7:
        seasonal_errors.extend(
            np.abs(arrivals[7:] - arrivals[:-7])
        )

mase_denominator = np.mean(seasonal_errors)

mase = np.mean(
    np.abs(y_true - y_pred)
) / mase_denominator


# ------------------------------------------------------------
# Horizon-specific MAE
# ------------------------------------------------------------

horizons = [0, 6, 13, 20, 29]

horizon_mae = {}

for h in horizons:
    horizon_mae[h + 1] = mean_absolute_error(
        y_true[:, h],
        y_pred[:, h]
    )


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

actual_mean = np.mean(y_true)
pred_mean = np.mean(y_pred)

actual_std = np.std(y_true)
pred_std = np.std(y_pred)

actual_min = np.min(y_true)
actual_max = np.max(y_true)

pred_min = np.min(y_pred)
pred_max = np.max(y_pred)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CORRECTED EXPERIMENT B — LSTM TEST RESULTS")
print("=" * 60)

print(f"\nMAE:    {mae:.2f}")
print(f"RMSE:   {rmse:.2f}")
print(f"sMAPE:  {smape:.2f}%")
print(f"MASE:   {mase:.4f}")

print(f"\nMASE denominator: {mase_denominator:.4f}")

print("\nHorizon MAE:")

for h, value in horizon_mae.items():
    print(f"  Day +{h:2d}: {value:.2f}")

print("\nForecast distribution:")

print(f"  Actual mean: {actual_mean:.2f}")
print(f"  Pred mean:   {pred_mean:.2f}")

print(f"\n  Actual std: {actual_std:.2f}")
print(f"  Pred std:   {pred_std:.2f}")

print(f"\n  Actual range: {actual_min:.0f} → {actual_max:.0f}")
print(f"  Pred range:   {pred_min:.0f} → {pred_max:.0f}")

print("\nFirst 30-day forecast:")

print("Actual:")
print(y_true[0].astype(int))

print("\nPredicted:")
print(np.round(y_pred[0]).astype(int))

print("\n" + "=" * 60)

# %%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    Dropout,
    Add,
    GlobalAveragePooling1D,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

# ============================================================
# EXPERIMENT B — TCN
# ============================================================

def residual_tcn_block(
    x,
    filters,
    kernel_size,
    dilation_rate,
    dropout_rate=0.20
):

    residual = x

    # First causal convolution
    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal"
    )(x)

    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)

    # Second causal convolution
    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal"
    )(y)

    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)

    # Match residual channels if necessary
    if residual.shape[-1] != filters:
        residual = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same"
        )(residual)

    y = Add()([residual, y])
    y = Activation("relu")(y)

    return y


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

inputs = Input(
    shape=(LOOKBACK, X_train_scaled_B.shape[-1])
)

x = inputs

filters = 64
kernel_size = 3
dilations = [1, 2, 4, 8, 16]

for dilation in dilations:
    x = residual_tcn_block(
        x,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        dropout_rate=0.20
    )

x = GlobalAveragePooling1D()(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.20)(x)

outputs = Dense(HORIZON)(x)

model_tcn_B = Model(
    inputs=inputs,
    outputs=outputs
)

model_tcn_B.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)

model_tcn_B.summary()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

early_stopping_tcn = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history_tcn_B = model_tcn_B.fit(
    X_train_scaled_B,
    y_train_scaled_B,
    validation_data=(
        X_val_scaled_B,
        y_val_scaled_B
    ),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping_tcn],
    verbose=1
)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ============================================================
# EXPERIMENT B — TCN TEST EVALUATION
# ============================================================

y_pred_scaled_tcn_B = model_tcn_B.predict(
    X_test_scaled_B,
    verbose=1
)

# Inverse transform
y_pred_tcn_B = target_scaler_B_corrected.inverse_transform(
    y_pred_scaled_tcn_B.reshape(-1, 1)
).reshape(y_pred_scaled_tcn_B.shape)


y_true = y_test_B.astype(float)
y_pred = y_pred_tcn_B.astype(float)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_true.flatten(),
    y_pred.flatten()
)

rmse = np.sqrt(
    mean_squared_error(
        y_true.flatten(),
        y_pred.flatten()
    )
)

smape = np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100


# ------------------------------------------------------------
# Correct 7-day MASE denominator
# ------------------------------------------------------------

seasonal_errors = []

for block_id, block in df_weekly.groupby("block_id", sort=True):

    block = block.sort_values("date")

    block = block[
        block["date"] <= pd.Timestamp("2023-12-31")
    ]

    arrivals = block[TARGET].values.astype(float)

    if len(arrivals) > 7:
        seasonal_errors.extend(
            np.abs(arrivals[7:] - arrivals[:-7])
        )

mase_denominator = np.mean(seasonal_errors)

mase = np.mean(
    np.abs(y_true - y_pred)
) / mase_denominator


# ------------------------------------------------------------
# Horizon MAE
# ------------------------------------------------------------

horizons = [0, 6, 13, 20, 29]

horizon_mae = {}

for h in horizons:
    horizon_mae[h + 1] = mean_absolute_error(
        y_true[:, h],
        y_pred[:, h]
    )


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

actual_mean = np.mean(y_true)
pred_mean = np.mean(y_pred)

actual_std = np.std(y_true)
pred_std = np.std(y_pred)

actual_min = np.min(y_true)
actual_max = np.max(y_true)

pred_min = np.min(y_pred)
pred_max = np.max(y_pred)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT B — TCN TEST RESULTS")
print("=" * 60)

print(f"\nMAE:    {mae:.2f}")
print(f"RMSE:   {rmse:.2f}")
print(f"sMAPE:  {smape:.2f}%")
print(f"MASE:   {mase:.4f}")

print(f"\nMASE denominator: {mase_denominator:.4f}")

print("\nHorizon MAE:")

for h, value in horizon_mae.items():
    print(f"  Day +{h:2d}: {value:.2f}")

print("\nForecast distribution:")

print(f"  Actual mean: {actual_mean:.2f}")
print(f"  Pred mean:   {pred_mean:.2f}")

print(f"\n  Actual std: {actual_std:.2f}")
print(f"  Pred std:   {pred_std:.2f}")

print(f"\n  Actual range: {actual_min:.0f} → {actual_max:.0f}")
print(f"  Pred range:   {pred_min:.0f} → {pred_max:.0f}")

print("\nFirst 30-day forecast:")

print("Actual:")
print(y_true[0].astype(int))

print("\nPredicted:")
print(np.round(y_pred[0]).astype(int))

print("\n" + "=" * 60)


# %%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Embedding
)
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

# ============================================================
# EXPERIMENT B — TRANSFORMER ENCODER
# ============================================================

class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, d_model):
        super().__init__()

        self.position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=d_model
        )

    def call(self, inputs):

        length = tf.shape(inputs)[1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        position_embeddings = self.position_embedding(
            positions
        )

        return inputs + position_embeddings


def transformer_encoder(
    inputs,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.20
):

    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate
    )(
        inputs,
        inputs
    )

    attention_output = Dropout(
        dropout_rate
    )(attention_output)

    # Residual + normalization
    x = LayerNormalization(
        epsilon=1e-6
    )(inputs + attention_output)

    # Feed-forward network
    ff_output = Dense(
        ff_dim,
        activation="relu"
    )(x)

    ff_output = Dropout(
        dropout_rate
    )(ff_output)

    ff_output = Dense(
        d_model
    )(ff_output)

    ff_output = Dropout(
        dropout_rate
    )(ff_output)

    # Residual + normalization
    return LayerNormalization(
        epsilon=1e-6
    )(x + ff_output)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

inputs = Input(
    shape=(LOOKBACK, X_train_scaled_B.shape[-1])
)

# Project 114 input features → 64-dimensional representation
x = Dense(64)(inputs)

# Add learned positional information
x = PositionalEmbedding(
    sequence_length=LOOKBACK,
    d_model=64
)(x)


# Two Transformer encoder blocks
x = transformer_encoder(
    x,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.20
)

x = transformer_encoder(
    x,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.20
)


# Aggregate the 90 time steps
x = GlobalAveragePooling1D()(x)

# Forecast head
x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(0.20)(x)

outputs = Dense(
    HORIZON
)(x)


model_transformer_B = Model(
    inputs=inputs,
    outputs=outputs
)


model_transformer_B.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


model_transformer_B.summary()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

early_stopping_transformer = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


history_transformer_B = model_transformer_B.fit(
    X_train_scaled_B,
    y_train_scaled_B,
    validation_data=(
        X_val_scaled_B,
        y_val_scaled_B
    ),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping_transformer],
    verbose=1
)

# %%
import numpy as np

# ============================================================
# EXPERIMENT B — TRANSFORMER TEST EVALUATION
# ============================================================

# Predict on the test set
y_pred_transformer_scaled = model_transformer_B.predict(
    X_test_scaled_B,
    verbose=1
)

# Convert predictions and actual values back to original scale
y_pred_transformer = target_scaler_B.inverse_transform(
    y_pred_transformer_scaled.reshape(-1, 1)
).reshape(y_pred_transformer_scaled.shape)

y_test_actual = target_scaler_B.inverse_transform(
    y_test_scaled_B.reshape(-1, 1)
).reshape(y_test_scaled_B.shape)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

# MAE
mae_transformer = np.mean(
    np.abs(y_test_actual - y_pred_transformer)
)

# RMSE
rmse_transformer = np.sqrt(
    np.mean(
        (y_test_actual - y_pred_transformer) ** 2
    )
)

# sMAPE
smape_transformer = np.mean(
    2 * np.abs(y_test_actual - y_pred_transformer) /
    (
        np.abs(y_test_actual) +
        np.abs(y_pred_transformer) +
        1e-8
    )
) * 100


# 7-day seasonal naive MASE
mase_transformer = (
    mae_transformer /
    mase_denominator
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRANSFORMER — TEST RESULTS")
print("=" * 60)

print(f"MAE:       {mae_transformer:.2f}")
print(f"RMSE:      {rmse_transformer:.2f}")
print(f"sMAPE:     {smape_transformer:.2f}%")
print(f"7-day MASE: {mase_transformer:.4f}")

print(f"\nMASE denominator: {mase_denominator:.4f}")


# ------------------------------------------------------------
# Horizon-specific MAE
# ------------------------------------------------------------

horizons = [1, 7, 14, 21, 30]

print("\nHorizon-specific MAE:")

for day in horizons:
    horizon_mae = np.mean(
        np.abs(
            y_test_actual[:, day - 1] -
            y_pred_transformer[:, day - 1]
        )
    )

    print(
        f"Day +{day:02d}: "
        f"{horizon_mae:.2f}"
    )


# ------------------------------------------------------------
# Prediction distribution
# ------------------------------------------------------------

print("\nPrediction distribution:")

print(
    f"Actual mean: "
    f"{np.mean(y_test_actual):.2f}"
)

print(
    f"Predicted mean: "
    f"{np.mean(y_pred_transformer):.2f}"
)

print(
    f"Actual std: "
    f"{np.std(y_test_actual):.2f}"
)

print(
    f"Predicted std: "
    f"{np.std(y_pred_transformer):.2f}"
)

print(
    f"Actual range: "
    f"{np.min(y_test_actual):.0f} → "
    f"{np.max(y_test_actual):.0f}"
)

print(
    f"Predicted range: "
    f"{np.min(y_pred_transformer):.0f} → "
    f"{np.max(y_pred_transformer):.0f}"
)


# ------------------------------------------------------------
# First 30-day forecast example
# ------------------------------------------------------------

print("\nFirst test sample — actual:")
print(
    np.round(
        y_test_actual[0]
    ).astype(int)
)

print("\nFirst test sample — Transformer prediction:")
print(
    np.round(
        y_pred_transformer[0]
    ).astype(int)
)

# %%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Add
)
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

# ============================================================
# EXPERIMENT B — N-BEATS-STYLE MODEL
# ============================================================

def nbeats_block(
    x,
    hidden_units=128,
    forecast_size=30,
    dropout_rate=0.20
):

    # Fully connected block
    h = Dense(
        hidden_units,
        activation="relu"
    )(x)

    h = Dropout(
        dropout_rate
    )(h)

    h = Dense(
        hidden_units,
        activation="relu"
    )(h)

    h = Dropout(
        dropout_rate
    )(h)

    h = Dense(
        hidden_units,
        activation="relu"
    )(h)

    h = Dropout(
        dropout_rate
    )(h)

    # Backcast branch
    backcast = Dense(
        90,
        activation="linear"
    )(h)

    # Forecast branch
    forecast = Dense(
        forecast_size,
        activation="linear"
    )(h)

    return backcast, forecast


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

inputs = Input(
    shape=(LOOKBACK, X_train_scaled_B.shape[-1])
)

# Flatten the complete 90-day × 114-feature history
x = tf.keras.layers.Flatten()(inputs)


# ------------------------------------------------------------
# Initial representation
# ------------------------------------------------------------

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(
    0.20
)(x)


# ------------------------------------------------------------
# N-BEATS-style blocks
# ------------------------------------------------------------

backcast1, forecast1 = nbeats_block(
    x,
    hidden_units=128,
    forecast_size=HORIZON,
    dropout_rate=0.20
)

# First residual connection
x2 = Add()([
    x,
    Dense(128)(backcast1)
])


backcast2, forecast2 = nbeats_block(
    x2,
    hidden_units=128,
    forecast_size=HORIZON,
    dropout_rate=0.20
)

x3 = Add()([
    x2,
    Dense(128)(backcast2)
])


backcast3, forecast3 = nbeats_block(
    x3,
    hidden_units=128,
    forecast_size=HORIZON,
    dropout_rate=0.20
)


# ------------------------------------------------------------
# Combine forecasts from all blocks
# ------------------------------------------------------------

outputs = Add()([
    forecast1,
    forecast2,
    forecast3
])


# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

model_nbeats_B = Model(
    inputs=inputs,
    outputs=outputs
)


model_nbeats_B.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


model_nbeats_B.summary()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

early_stopping_nbeats = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


history_nbeats_B = model_nbeats_B.fit(
    X_train_scaled_B,
    y_train_scaled_B,
    validation_data=(
        X_val_scaled_B,
        y_val_scaled_B
    ),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping_nbeats],
    verbose=1
)

# %%
import numpy as np

# ============================================================
# EXPERIMENT B — N-BEATS-STYLE TEST EVALUATION
# ============================================================

# Predict on the test set
y_pred_nbeats_scaled = model_nbeats_B.predict(
    X_test_scaled_B,
    verbose=1
)

# Convert predictions back to original arrivals scale
y_pred_nbeats = target_scaler_B.inverse_transform(
    y_pred_nbeats_scaled.reshape(-1, 1)
).reshape(y_pred_nbeats_scaled.shape)

# Actual values in original scale
y_test_actual = target_scaler_B.inverse_transform(
    y_test_scaled_B.reshape(-1, 1)
).reshape(y_test_scaled_B.shape)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

# MAE
mae_nbeats = np.mean(
    np.abs(
        y_test_actual - y_pred_nbeats
    )
)

# RMSE
rmse_nbeats = np.sqrt(
    np.mean(
        (y_test_actual - y_pred_nbeats) ** 2
    )
)

# sMAPE
smape_nbeats = np.mean(
    2 * np.abs(
        y_test_actual - y_pred_nbeats
    ) /
    (
        np.abs(y_test_actual) +
        np.abs(y_pred_nbeats) +
        1e-8
    )
) * 100

# 7-day seasonal-naive MASE
mase_nbeats = (
    mae_nbeats /
    mase_denominator
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("N-BEATS-STYLE — TEST RESULTS")
print("=" * 60)

print(f"MAE:       {mae_nbeats:.2f}")
print(f"RMSE:      {rmse_nbeats:.2f}")
print(f"sMAPE:     {smape_nbeats:.2f}%")
print(f"7-day MASE: {mase_nbeats:.4f}")

print(f"\nMASE denominator: {mase_denominator:.4f}")


# ------------------------------------------------------------
# Horizon-specific MAE
# ------------------------------------------------------------

horizons = [1, 7, 14, 21, 30]

print("\nHorizon-specific MAE:")

for day in horizons:

    horizon_mae = np.mean(
        np.abs(
            y_test_actual[:, day - 1] -
            y_pred_nbeats[:, day - 1]
        )
    )

    print(
        f"Day +{day:02d}: "
        f"{horizon_mae:.2f}"
    )


# ------------------------------------------------------------
# Prediction distribution
# ------------------------------------------------------------

print("\nPrediction distribution:")

print(
    f"Actual mean: "
    f"{np.mean(y_test_actual):.2f}"
)

print(
    f"Predicted mean: "
    f"{np.mean(y_pred_nbeats):.2f}"
)

print(
    f"Actual std: "
    f"{np.std(y_test_actual):.2f}"
)

print(
    f"Predicted std: "
    f"{np.std(y_pred_nbeats):.2f}"
)

print(
    f"Actual range: "
    f"{np.min(y_test_actual):.0f} → "
    f"{np.max(y_test_actual):.0f}"
)

print(
    f"Predicted range: "
    f"{np.min(y_pred_nbeats):.0f} → "
    f"{np.max(y_pred_nbeats):.0f}"
)


# ------------------------------------------------------------
# First 30-day forecast example
# ------------------------------------------------------------

print("\nFirst test sample — actual:")
print(
    np.round(
        y_test_actual[0]
    ).astype(int)
)

print("\nFirst test sample — N-BEATS-style prediction:")
print(
    np.round(
        y_pred_nbeats[0]
    ).astype(int)
)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# TEST DATES
# =========================================================

test_target_dates = target_dates_B[test_mask_B]

print("Number of test forecast windows:", len(test_target_dates))
print("First forecast:", test_target_dates.min())
print("Last forecast:", test_target_dates.max())


# =========================================================
# FUNCTION:
# Convert overlapping 30-day forecasts into
# one prediction for each calendar day
# =========================================================

def reconstruct_daily_predictions(predictions, start_dates, horizon=30):

    daily_predictions = {}

    for i, start_date in enumerate(start_dates):

        for h in range(horizon):

            date = pd.Timestamp(start_date) + pd.Timedelta(days=h)

            if date not in daily_predictions:
                daily_predictions[date] = []

            daily_predictions[date].append(predictions[i, h])

    # Average overlapping forecasts for the same day
    daily_predictions = {
        date: np.mean(values)
        for date, values in daily_predictions.items()
    }

    return pd.Series(daily_predictions).sort_index()


# =========================================================
# CREATE DAILY PREDICTIONS
# =========================================================

lstm_daily = reconstruct_daily_predictions(
    y_pred_B_corrected,
    test_target_dates
)

tcn_daily = reconstruct_daily_predictions(
    y_pred_tcn_B,
    test_target_dates
)

transformer_daily = reconstruct_daily_predictions(
    y_pred_transformer,
    test_target_dates
)

nbeats_daily = reconstruct_daily_predictions(
    y_pred_nbeats,
    test_target_dates
)


# =========================================================
# ACTUAL ARRIVALS
# =========================================================

actual_daily = df_weekly.set_index("date")["arrivals"]


# =========================================================
# COMMON DATES
# =========================================================

common_dates = (
    actual_daily.index
    .intersection(lstm_daily.index)
    .intersection(tcn_daily.index)
    .intersection(transformer_daily.index)
    .intersection(nbeats_daily.index)
)


# =========================================================
# FINAL FULL-PERIOD RESULTS TABLE
# =========================================================

results = pd.DataFrame({
    "Actual": actual_daily.loc[common_dates],
    "LSTM": lstm_daily.loc[common_dates],
    "TCN": tcn_daily.loc[common_dates],
    "Transformer": transformer_daily.loc[common_dates],
    "N-BEATS": nbeats_daily.loc[common_dates]
})

print()
print("==============================================")
print("FULL TEST PERIOD")
print("==============================================")
print("Start:", results.index.min())
print("End:  ", results.index.max())
print("Number of daily observations:", len(results))

print()
print(results.head())
print()
print(results.tail())

# %%
plt.figure(figsize=(20, 8))

plt.plot(
    results.index,
    results["Actual"],
    label="Actual",
    linewidth=2
)

plt.plot(
    results.index,
    results["LSTM"],
    label="LSTM",
    linewidth=1.5
)

plt.plot(
    results.index,
    results["TCN"],
    label="TCN",
    linewidth=1.5
)

plt.plot(
    results.index,
    results["Transformer"],
    label="Transformer",
    linewidth=1.5
)

plt.plot(
    results.index,
    results["N-BEATS"],
    label="N-BEATS",
    linewidth=1.5
)

plt.title(
    "Actual vs Predicted Sri Lankan Tourist Arrivals\n"
    "Test Period: January 2025 – July 2026",
    fontsize=16
)

plt.xlabel("Date", fontsize=12)
plt.ylabel("Tourist Arrivals", fontsize=12)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
models = [
    "LSTM",
    "TCN",
    "Transformer",
    "N-BEATS"
]

for model in models:

    plt.figure(figsize=(20, 7))

    # Actual values
    plt.plot(
        results.index,
        results["Actual"],
        label="Actual",
        linewidth=2
    )

    # Model predictions
    plt.plot(
        results.index,
        results[model],
        label=model,
        linewidth=1.5
    )

    plt.title(
        f"Actual vs {model} Predicted Tourist Arrivals\n"
        "Test Period: January 2025 – July 2026",
        fontsize=16
    )

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Tourist Arrivals", fontsize=12)

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
models = [
    "LSTM",
    "TCN",
    "Transformer",
    "N-BEATS"
]

plt.figure(figsize=(20, 8))

for model in models:

    # Absolute prediction error
    absolute_error = np.abs(
        results[model] - results["Actual"]
    )

    # 30-day rolling MAE
    rolling_mae = absolute_error.rolling(
        window=30,
        min_periods=30
    ).mean()

    plt.plot(
        results.index,
        rolling_mae,
        label=model,
        linewidth=1.8
    )

plt.title(
    "30-Day Rolling MAE Across the Test Period\n"
    "January 2025 – July 2026",
    fontsize=16
)

plt.xlabel("Date", fontsize=12)
plt.ylabel("30-Day Rolling MAE", fontsize=12)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
models = [
    "LSTM",
    "TCN",
    "Transformer",
    "N-BEATS"
]

plt.figure(figsize=(20, 8))

for model in models:

    error = results[model] - results["Actual"]

    plt.plot(
        results.index,
        error,
        label=model,
        linewidth=1.5
    )

# Zero-error reference line
plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Prediction Error Across the Test Period\n"
    "January 2025 – July 2026",
    fontsize=16
)

plt.xlabel("Date", fontsize=12)
plt.ylabel("Prediction Error (Predicted − Actual)", fontsize=12)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# =========================================================
# FINAL MODEL METRICS
# =========================================================

metrics = pd.DataFrame({
    "Model": [
        "LSTM",
        "TCN",
        "Transformer",
        "N-BEATS"
    ],
    "MAE": [
        1300.38,
        1330.41,
        1654.65,
        1652.87
    ],
    "RMSE": [
        1787.63,
        1868.03,
        2058.20,
        2237.67
    ],
    "sMAPE": [
        20.50,
        20.62,
        26.02,
        26.11
    ],
    "MASE": [
        7.2941,
        7.4625,
        9.2813,
        9.2713
    ]
})


# =========================================================
# CREATE FOUR SEPARATE GRAPHS
# =========================================================

metric_names = ["MAE", "RMSE", "sMAPE", "MASE"]

for metric in metric_names:

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        metrics["Model"],
        metrics[metric]
    )

    # Add exact values above each bar
    for bar, value in zip(bars, metrics[metric]):

        if metric in ["sMAPE"]:
            label = f"{value:.2f}%"
        elif metric == "MASE":
            label = f"{value:.4f}"
        else:
            label = f"{value:.2f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.title(
        f"{metric} Comparison Across Deep Learning Models",
        fontsize=15
    )

    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

# %%
# =========================================================
# MONTHLY AVERAGES
# =========================================================

monthly_results = results.resample("MS").mean()

print("Monthly observations:", len(monthly_results))
print("Period:", monthly_results.index.min(), "to", monthly_results.index.max())

print()
print(monthly_results)

# %%
plt.figure(figsize=(18, 8))

plt.plot(
    monthly_results.index,
    monthly_results["Actual"],
    marker="o",
    linewidth=2.5,
    label="Actual"
)

plt.plot(
    monthly_results.index,
    monthly_results["LSTM"],
    marker="o",
    linewidth=1.8,
    label="LSTM"
)

plt.plot(
    monthly_results.index,
    monthly_results["TCN"],
    marker="o",
    linewidth=1.8,
    label="TCN"
)

plt.plot(
    monthly_results.index,
    monthly_results["Transformer"],
    marker="o",
    linewidth=1.8,
    label="Transformer"
)

plt.plot(
    monthly_results.index,
    monthly_results["N-BEATS"],
    marker="o",
    linewidth=1.8,
    label="N-BEATS"
)

plt.title(
    "Monthly Average Actual vs Predicted Tourist Arrivals\n"
    "January 2025 – July 2026",
    fontsize=16
)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Daily Tourist Arrivals", fontsize=12)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.xticks(
    monthly_results.index,
    monthly_results.index.strftime("%b %Y"),
    rotation=45
)

plt.tight_layout()
plt.show()

# %%



