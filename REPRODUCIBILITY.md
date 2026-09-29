# Reproducibility Information

## Environment

The project was primarily developed and executed using Google Colab with Python and TensorFlow/Keras.

Install dependencies using:

```bash
pip install -r requirements.txt
```

Some files use Google Colab `/content/...` paths. When running from a cloned repository, update these paths or copy the required input files into `/content`.

## Common Forecasting Configuration

- Input lookback: 90 days
- Forecast horizon: 30 days
- General training period: up to 31 December 2023
- Validation period: calendar year 2024
- Test period: from 01 January 2025 onward
- Random seed: 42 where applicable
- Main loss: Mean Squared Error
- Main optimizer: Adam
- Evaluation metrics: MAE, RMSE, sMAPE and MASE

All final evaluation metrics are calculated after converting predictions back to the original tourist-arrival scale.

## Random Seeds

The project uses seed 42 where applicable.

Depending on the experiment, the source code includes settings such as:

```python
tf.random.set_seed(42)
```

and/or:

```python
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
```

The primary controlled experiment also attempts to enable deterministic TensorFlow operations where supported.

## Experiment 1 — Preliminary Full-Feature Daily Experiment

Script: `training.py`

### Configuration

- Input window: 90 days
- Forecast horizon: 30 days
- Input feature dimension: 109
- Training: target start dates before 2024
- Validation: target start dates during 2024
- Test: target start dates from 2025 onward
- Training-only feature scaling
- Training-only target scaling
- Random seed: 42

### Core Models

- Stacked LSTM
- TCN
- Transformer
- N-BEATS-style model

### Core LSTM Training Configuration

- LSTM 64
- Dropout 0.20
- LSTM 32
- Dense 64 with ReLU
- Dropout 0.20
- Dense 30 output
- Adam learning rate: 0.001
- Loss: MSE
- Batch size: 32
- Maximum epochs: 100
- Early stopping patience: 10

### Methodological Note

This was the preliminary/development experiment.

Its sequence assignment uses the first target date to assign a sequence to train/validation/test. Because each sequence predicts 30 days, some target windows near a partition boundary can extend into the next period.

The code is retained unchanged for reproducibility and the final report treats this experiment as preliminary rather than the principal controlled comparison.

## Primary Controlled Full-History Experiment

Script: `deep_forecasting_pipeline.py`

### Data

- Final encoded dataset: 6,053 rows × 110 columns
- `covid_impact_factor`, `crisis_impact_factor` and `exchange_rates_complete` are removed from the final modelling feature set.
- 105 exogenous predictors remain.
- 25 are historically observed variables.
- 80 are known-future variables.
- Historical scaled arrivals are appended to the neural-network inputs.
- Final historical neural input shape: 90 × 106.
- Output shape: 30.

### Date Split

- Training: through 31 December 2023
- Validation: 01 January 2024 to 31 December 2024
- Test: 01 January 2025 to 31 July 2026

### Windowing

Windows are created separately inside each partition.

- Training windows: 4,991
- Validation windows: 247
- Test windows: 458

This prevents train/validation/test forecast windows from crossing partition boundaries.

### Scaling

- Continuous predictors: StandardScaler fitted on training data only
- Target arrivals: StandardScaler fitted on training arrivals only
- Binary and one-hot variables remain in their original numerical representation

### Models

- 7-day Seasonal Naive baseline
- LSTM
- GRU with Temporal Attention
- Dilated Residual TCN
- TFT-style forecaster

### Shared Neural Training Settings

Where applicable:

- Random seed: 42
- Adam learning rate: 0.001
- Loss: MSE
- Batch size: 32
- Maximum epochs: 100
- `shuffle=False`
- Early stopping
- Best-model checkpointing
- Test set excluded from training and model selection

Model-specific architectures and dropout values are preserved directly in `deep_forecasting_pipeline.py`.

## Experiment 3 — Crisis-Adjusted Robustness Experiment

Script: `ahamedtraining.py`

### Historical Treatment

Rows from 01 January 2020 through 31 December 2022 are excluded.

This creates two continuous modelling blocks:

- 2010–2019
- 2023–2026

Sequences are created so that no artificial window bridges the removed period.

### Additional Past-Only Arrival Features

Lags:

- 7 days
- 14 days
- 21 days
- 28 days

Rolling means:

- 7 days
- 14 days
- 30 days

Rolling standard deviations:

- 7 days
- 30 days

Rolling calculations use past arrivals only with a one-day shift before the rolling calculation.

### Feature / Sequence Configuration

- Input window: 90 days
- Forecast horizon: 30 days
- Input features: 114
- Training windows: 3,716
- Validation windows: 337
- Test windows: 548

The complete 30-day target horizon must remain within its assigned partition.

### Models

- LSTM
- TCN
- Transformer
- N-BEATS-style model

Training-only scaling is used. The MASE denominator is calculated using valid training observations while respecting the discontinuity between the two historical blocks.

## Dataset Reproducibility

Two key CSV files are committed:

- `0.  Research Data.csv` — base dataset obtained from a senior student
- `Sri_Lanka_Tourism_Arrivals_Encoded.csv` — final common encoded modelling dataset

The final encoded dataset should be used when the objective is to reproduce model results.

Some Google Trends extension values were added manually. Therefore, the committed dataset is the authoritative version used by the experiments.

## Source-of-Truth Rule

For exact architecture settings, use the corresponding Python script as the source of truth.

This document records the shared experimental configuration, while model-specific layer sizes, callbacks and tuning steps remain in the source files.
