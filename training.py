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

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
tf.random.set_seed(42)

# Build Stacked LSTM
lstm_model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(90, 109)
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dense(64, activation="relu"),

    Dropout(0.2),

    Dense(30)
])

# Compile
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# Display architecture
lstm_model.summary()

# %%
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history_lstm = lstm_model.fit(
    X_train,
    y_train_scaled,

    validation_data=(
        X_val,
        y_val_scaled
    ),

    epochs=100,
    batch_size=32,

    callbacks=[early_stopping],

    verbose=1
)

# %%
# Predict the 30-day forecasts for the test set
y_pred_lstm_scaled = lstm_model.predict(
    X_test,
    verbose=1
)

print("Prediction shape:", y_pred_lstm_scaled.shape)

# %%
# Convert predictions back to actual tourist-arrival values
y_pred_lstm = target_scaler.inverse_transform(
    y_pred_lstm_scaled.reshape(-1, 1)
).reshape(y_pred_lstm_scaled.shape)

# Convert actual test values back as well
y_test_actual = target_scaler.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).reshape(y_test_scaled.shape)

print("Predictions shape:", y_pred_lstm.shape)
print("Actual shape:", y_test_actual.shape)

print("\nFirst prediction:")
print(y_pred_lstm[0])

print("\nFirst actual 30 days:")
print(y_test_actual[0])

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Flatten all 548 × 30 predictions into one array
y_true = y_test_actual.flatten()
y_pred = y_pred_lstm.flatten()

# MAE
mae = mean_absolute_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# sMAPE
smape = np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100

# MASE
# Naive one-step seasonal benchmark using lag-1 arrivals
train_arrivals = train_df["arrivals"].values

naive_errors = np.abs(
    train_arrivals[1:] - train_arrivals[:-1]
)

mase_scale = np.mean(naive_errors)

mase = np.mean(np.abs(y_true - y_pred)) / mase_scale

print("LSTM TEST RESULTS")
print("-----------------")
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")
print(f"sMAPE: {smape:.2f}%")
print(f"MASE:  {mase:.4f}")

# %%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization,
    Activation, Add, Dropout,
    GlobalAveragePooling1D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)


def tcn_residual_block(x, filters, kernel_size, dilation_rate):
    # First causal convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate
    )(x)

    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Dropout(0.2)(conv1)

    # Second causal convolution
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate
    )(conv1)

    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Dropout(0.2)(conv2)

    # Residual connection
    if x.shape[-1] != filters:
        x = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same"
        )(x)

    return Add()([x, conv2])


# Input
inputs = Input(shape=(90, 109))

x = Conv1D(
    filters=64,
    kernel_size=3,
    padding="causal"
)(inputs)

# Dilated TCN blocks
for dilation in [1, 2, 4, 8, 16]:
    x = tcn_residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=dilation
    )

# Convert sequence to vector
x = GlobalAveragePooling1D()(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

# 30-day forecast
outputs = Dense(30)(x)

tcn_model = Model(inputs, outputs)

tcn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

tcn_model.summary()

# %%
early_stopping_tcn = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history_tcn = tcn_model.fit(
    X_train,
    y_train_scaled,

    validation_data=(
        X_val,
        y_val_scaled
    ),

    epochs=100,
    batch_size=32,

    callbacks=[early_stopping_tcn],

    verbose=1
)

# %%
y_pred_tcn_scaled = tcn_model.predict(
    X_test,
    verbose=1
)

# Convert back to actual arrival counts
y_pred_tcn = target_scaler.inverse_transform(
    y_pred_tcn_scaled.reshape(-1, 1)
).reshape(y_pred_tcn_scaled.shape)

print("TCN prediction shape:", y_pred_tcn.shape)

# %%
y_true = y_test_actual.flatten()
y_pred = y_pred_tcn.flatten()

mae_tcn = mean_absolute_error(y_true, y_pred)

rmse_tcn = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

smape_tcn = np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100

mase_tcn = np.mean(
    np.abs(y_true - y_pred)
) / mase_scale

print("TCN TEST RESULTS")
print("----------------")
print(f"MAE:   {mae_tcn:.2f}")
print(f"RMSE:  {rmse_tcn:.2f}")
print(f"sMAPE: {smape_tcn:.2f}%")
print(f"MASE:  {mase_tcn:.4f}")

# %%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    Embedding, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=d_model
        )

    def call(self, inputs):
        positions = tf.range(
            start=0,
            limit=tf.shape(inputs)[1],
            delta=1
        )
        return inputs + self.position_embedding(positions)


def transformer_encoder(x, d_model=64, num_heads=4, ff_dim=128):
    # Self-attention
    attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)

    attention = Dropout(0.1)(attention)

    # Residual + normalization
    x = Add()([x, attention])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(0.1)(ff)
    ff = Dense(d_model)(ff)

    # Residual + normalization
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# Input
inputs = Input(shape=(90, 109))

# Project 109 features → 64-dimensional representation
x = Dense(64)(inputs)

# Add positional information
x = PositionalEmbedding(
    sequence_length=90,
    d_model=64
)(x)

# Transformer encoder blocks
x = transformer_encoder(x)
x = transformer_encoder(x)

# Convert sequence to forecast
x = GlobalAveragePooling1D()(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

# 30-day output
outputs = Dense(30)(x)

transformer_model = Model(inputs, outputs)

transformer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

transformer_model.summary()

# %%
early_stopping_transformer = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history_transformer = transformer_model.fit(
    X_train,
    y_train_scaled,

    validation_data=(
        X_val,
        y_val_scaled
    ),

    epochs=100,
    batch_size=32,

    callbacks=[early_stopping_transformer],

    verbose=1
)

# %%
y_pred_transformer_scaled = transformer_model.predict(
    X_test,
    verbose=1
)

y_pred_transformer = target_scaler.inverse_transform(
    y_pred_transformer_scaled.reshape(-1, 1)
).reshape(y_pred_transformer_scaled.shape)

print("Prediction shape:", y_pred_transformer.shape)

# %%
y_true = y_test_actual.flatten()
y_pred = y_pred_transformer.flatten()

mae_transformer = mean_absolute_error(y_true, y_pred)

rmse_transformer = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

smape_transformer = np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100

mase_transformer = np.mean(
    np.abs(y_true - y_pred)
) / mase_scale

print("TRANSFORMER TEST RESULTS")
print("------------------------")
print(f"MAE:   {mae_transformer:.2f}")
print(f"RMSE:  {rmse_transformer:.2f}")
print(f"sMAPE: {smape_transformer:.2f}%")
print(f"MASE:  {mase_transformer:.4f}")

# %%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Subtract, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)


def nbeats_block(x, hidden_units=256, input_size=256, forecast_size=30):

    # Shared fully connected layers
    h = Dense(hidden_units, activation="relu")(x)
    h = Dense(hidden_units, activation="relu")(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.2)(h)

    # Backcast must have the SAME size as x
    backcast = Dense(input_size)(h)

    # 30-day forecast
    forecast = Dense(forecast_size)(h)

    return backcast, forecast


# Input: 90 days × 109 features
inputs = Input(shape=(90, 109))

# Flatten to 9810 values
x = Flatten()(inputs)

# Project to fixed N-BEATS representation
x = Dense(256, activation="relu")(x)

# -------------------------
# Block 1
# -------------------------
backcast1, forecast1 = nbeats_block(x)

residual1 = Subtract()([
    x,
    backcast1
])

# -------------------------
# Block 2
# -------------------------
backcast2, forecast2 = nbeats_block(residual1)

residual2 = Subtract()([
    residual1,
    backcast2
])

# -------------------------
# Block 3
# -------------------------
backcast3, forecast3 = nbeats_block(residual2)

# Combine forecasts
outputs = Add()([
    forecast1,
    forecast2,
    forecast3
])

# Create model
nbeats_model = Model(
    inputs=inputs,
    outputs=outputs
)

# Compile
nbeats_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

nbeats_model.summary()

# %%
early_stopping_nbeats = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history_nbeats = nbeats_model.fit(
    X_train,
    y_train_scaled,

    validation_data=(
        X_val,
        y_val_scaled
    ),

    epochs=100,
    batch_size=32,

    callbacks=[early_stopping_nbeats],

    verbose=1
)

# %%
y_pred_nbeats_scaled = nbeats_model.predict(
    X_test,
    verbose=1
)

y_pred_nbeats = target_scaler.inverse_transform(
    y_pred_nbeats_scaled.reshape(-1, 1)
).reshape(y_pred_nbeats_scaled.shape)

print("Prediction shape:", y_pred_nbeats.shape)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_true = y_test_actual.flatten()
y_pred = y_pred_nbeats.flatten()

# MAE
mae = mean_absolute_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# sMAPE
smape = np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
) * 100

# MASE — same calculation used for the other models
train_arrivals = train_df["arrivals"].values

naive_errors = np.abs(
    train_arrivals[1:] - train_arrivals[:-1]
)

mase_scale = np.mean(naive_errors)

mase = np.mean(
    np.abs(y_true - y_pred)
) / mase_scale

print("N-BEATS-Style Test Results")
print("--------------------------")
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")
print(f"sMAPE: {smape:.2f}%")
print(f"MASE:  {mase:.4f}")

# %%
import matplotlib.pyplot as plt

models = {
    "LSTM": y_pred_lstm,
    "TCN": y_pred_tcn,
    "Transformer": y_pred_transformer,
    "N-BEATS-style": y_pred_nbeats
}

# Dates for the first 30-day test forecast
forecast_dates = target_dates[test_mask.values][:30]

for name, predictions in models.items():

    plt.figure(figsize=(12, 5))

    plt.plot(
        forecast_dates,
        y_test_actual[0],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        forecast_dates,
        predictions[0],
        label="Predicted",
        linewidth=2
    )

    plt.title(
        f"{name}: Actual vs Predicted — First 30-Day Test Forecast"
    )

    plt.xlabel("Date")
    plt.ylabel("Tourist Arrivals")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

# %%
import matplotlib.pyplot as plt

histories = {
    "LSTM": history_lstm,
    "TCN": history_tcn,
    "Transformer": history_transformer,
    "N-BEATS-style": history_nbeats
}

for name, history in histories.items():

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title(f"{name}: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

# %%
import matplotlib.pyplot as plt

models = ["LSTM", "TCN", "Transformer", "N-BEATS-style"]

mae = [1552.42, 1911.47, 2532.51, 2659.23]
rmse = [2212.47, 2945.04, 3590.27, 3704.99]
smape = [26.40, 33.28, 55.56, 58.54]
mase = [9.0193, 11.1052, 14.7133, 15.4496]

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "sMAPE (%)": smape,
    "MASE": mase
}

for metric, values in metrics.items():

    plt.figure(figsize=(9, 5))

    bars = plt.bar(models, values)

    plt.title(f"Model Comparison — {metric}")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=20)

    # Display values above bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
