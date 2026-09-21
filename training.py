# %%
import pandas as pd

df = pd.read_csv("/content/Sri_Lanka_Tourism_Arrivals_Encoded.csv")

print("Shape:", df.shape)
print("Date range:", df["date"].min(), "→", df["date"].max())
print("Missing values:", df.isna().sum().sum())
print("Columns:", len(df.columns))

# %%
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("First date:", df["date"].min())
print("Last date:", df["date"].max())
print("Rows:", len(df))
print("Duplicate dates:", df["date"].duplicated().sum())

# %%
train_df = df[df["date"] < "2024-01-01"].copy()

val_df = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] < "2025-01-01")
].copy()

test_df = df[df["date"] >= "2025-01-01"].copy()

print("TRAIN:", len(train_df), train_df["date"].min(), "→", train_df["date"].max())
print("VALIDATION:", len(val_df), val_df["date"].min(), "→", val_df["date"].max())
print("TEST:", len(test_df), test_df["date"].min(), "→", test_df["date"].max())

# %%
# Target
target = "arrivals"

# Future-known features
# These are available when making a 30-day forecast.
future_known_features = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_holiday",
    "is_tourist_event",
    "in_holiday_window",
    "days_since_last_holiday"
]

# Variables that are only observed historically
observed_features = [
    col for col in df.columns
    if col not in future_known_features
    and col not in ["date", target]
]

print("Target:", target)
print("\nFuture-known features:", len(future_known_features))
print(future_known_features)

print("\nObserved historical features:", len(observed_features))
print(observed_features)

# %%
# Check data types
print(df.dtypes.value_counts())

# Find any non-numeric columns
non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNon-numeric columns:", non_numeric)
print("Count:", len(non_numeric))

# %%
import numpy as np

# Remove only date from model inputs
model_features = [col for col in df.columns if col != "date"]

# Target column
target_col = "arrivals"
target_index = model_features.index(target_col)

LOOKBACK = 90
HORIZON = 30


def create_sequences(data, target_index, lookback=90, horizon=30):
    X = []
    y = []

    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i-lookback:i])
        y.append(data[i:i+horizon, target_index])

    return np.array(X), np.array(y)


# Convert to NumPy
data = df[model_features].values.astype(np.float32)

X, y = create_sequences(
    data,
    target_index,
    LOOKBACK,
    HORIZON
)

print("X shape:", X.shape)
print("y shape:", y.shape)

# %%
# Find the date associated with each prediction window
target_dates = df["date"].iloc[
    LOOKBACK : LOOKBACK + len(y)
].reset_index(drop=True)

# Train: target dates in 2010–2023
train_mask = target_dates < "2024-01-01"

# Validation: target dates in 2024
val_mask = (
    (target_dates >= "2024-01-01") &
    (target_dates < "2025-01-01")
)

# Test: target dates from 2025 onward
test_mask = target_dates >= "2025-01-01"

X_train = X[train_mask.values]
y_train = y[train_mask.values]

X_val = X[val_mask.values]
y_val = y[val_mask.values]

X_test = X[test_mask.values]
y_test = y[test_mask.values]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nX_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nX_test:", X_test.shape)
print("y_test:", y_test.shape)

# %%
from sklearn.preprocessing import StandardScaler

# Reshape training data from:
# (samples, 90 days, 109 features)
# to:
# (samples * 90, 109)
# so StandardScaler can scale each feature.

n_features = X_train.shape[2]

scaler = StandardScaler()

X_train_2d = X_train.reshape(-1, n_features)

# FIT ONLY on training data
scaler.fit(X_train_2d)

# Transform all inputs using the training-fitted scaler
X_train = scaler.transform(
    X_train_2d
).reshape(X_train.shape)

X_val = scaler.transform(
    X_val.reshape(-1, n_features)
).reshape(X_val.shape)

X_test = scaler.transform(
    X_test.reshape(-1, n_features)
).reshape(X_test.shape)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nTraining mean:", X_train.mean())
print("Training std:", X_train.std())

# %%
from sklearn.preprocessing import StandardScaler

# Create target scaler
target_scaler = StandardScaler()

# Fit ONLY on training target values
target_scaler.fit(y_train.reshape(-1, 1))

# Transform train, validation, and test targets
y_train_scaled = target_scaler.transform(
    y_train.reshape(-1, 1)
).reshape(y_train.shape)

y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
).reshape(y_val.shape)

y_test_scaled = target_scaler.transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

print("y_train_scaled:", y_train_scaled.shape)
print("y_val_scaled:", y_val_scaled.shape)
print("y_test_scaled:", y_test_scaled.shape)

print("\nScaled train mean:", y_train_scaled.mean())
print("Scaled train std:", y_train_scaled.std())
