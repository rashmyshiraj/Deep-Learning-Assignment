# Sri Lanka Tourist Arrivals Forecasting Using Deep Learning

## SE4050 – Deep Learning

### GitHub Repository
https://github.com/rashmyshiraj/Deep-Learning-Assignment

## Group Members

| Student | Registration Number |
|---|---|
| I.R. Shiraj | IT22192028 |
| M.S.S. Ahamed | IT23357358 |
| A.G.T.S Ranasinghe | IT23345546 |

## Project Overview

This project investigates deep-learning methods for forecasting daily international tourist arrivals to Sri Lanka.

The forecasting problem is formulated as supervised multi-step time-series forecasting. Historical tourist arrivals and related external variables are used to predict tourist arrivals for the following 30 days.

The dataset includes daily tourist arrivals, weather variables, GDP per capita, inflation, Brent crude oil price, exchange rates, Google Trends search-interest variables, Sri Lankan public holidays, tourism-related events, and calendar/time-series features.

The main forecasting setup uses the previous 90 days of information to predict the following 30 days.

## Repository Contents

### Datasets

- `0.  Research Data.csv` — original/base dataset obtained from a senior student.
- `Sri_Lanka_Tourism_Arrivals_Encoded.csv` — final cleaned, feature-engineered and one-hot encoded dataset used as the common modelling dataset.

See `DATASET.md` for dataset provenance and access information.

### Data Collection / Extension

- `data_collection_GDP_inflation.py` — GDP per capita and inflation data collection/extension.
- `exchange_rates.py` — exchange-rate data collection and integration.
- `weather_and_arrivals.py` — weather and daily tourist-arrival extension and validation.

Some Google Trends values were added manually during dataset extension. The exact values used by the project are preserved in the committed datasets.

### Preprocessing

The preprocessing work was divided into three stages:

1. `preprocess_tharusha1.py` — preprocessing stage 1, completed by A.G.T.S Ranasinghe.
2. `rashmy_preprocessing.py` — preprocessing stage 2, completed by I.R. Shiraj.
3. `ahamed_preprocessing.py` — preprocessing stage 3, completed by M.S.S. Ahamed.
4. `categorical_encoding_tharusha.py` — final categorical one-hot encoding.

### Deep-Learning Experiments

- `training.py` — preliminary full-feature daily experiment using LSTM, TCN, Transformer and N-BEATS-style models.
- `deep_forecasting_pipeline.py` — primary controlled full-history experiment using a 7-day Seasonal Naive baseline, LSTM, GRU with Temporal Attention, Dilated Residual TCN and TFT-style forecaster.
- `ahamedtraining.py` — crisis-adjusted robustness experiment using LSTM, TCN, Transformer and N-BEATS-style models.

## Dataset Summary

The initial dataset was obtained from a senior student and originally covered 01 January 2010 to 18 September 2025.

The project group extended the dataset from 19 September 2025 to 31 July 2026.

The complete raw dataset contained:

- 6,056 daily observations
- 23 original columns
- Date range: 01 January 2010 to 31 July 2026

After preprocessing, removal of the initial rows with incomplete exchange-rate history, feature engineering and categorical encoding, the final encoded modelling dataset contained:

- 6,053 daily observations
- 110 columns
- Date range: 04 January 2010 to 31 July 2026

## Main Preprocessing Steps

The project preprocessing includes:

1. Date parsing and chronological sorting
2. Duplicate-date and missing-date checks
3. Target validation
4. Removal of the constant `location` field
5. Past-only forward filling of exchange-rate missing values
6. Missingness-indicator creation
7. GDP and inflation missing-value handling
8. Weather validation
9. Holiday and tourism-event cleaning
10. `NoHoliday` and `NoEvent` preparation
11. Rainy-day feature creation
12. Holiday-distance features
13. ±3-day holiday-window feature
14. COVID-19 date-based disruption factor
15. Sri Lankan economic-crisis date-based disruption factor
16. Rare holiday/event grouping
17. Calendar and cyclical feature generation
18. One-hot encoding using `pandas.get_dummies()`

## Forecasting Design

### Common Setup

- Forecasting type: supervised multi-step forecasting
- Lookback window: 90 days
- Forecast horizon: 30 days
- General training period: up to 31 December 2023
- Validation period: 01 January 2024 to 31 December 2024
- Test period: from 01 January 2025 onward
- Random seed: 42 where applicable
- Main optimizer: Adam
- Main loss: Mean Squared Error
- Evaluation metrics: MAE, RMSE, sMAPE and MASE

Feature and target scalers are fitted using training data only before being applied to validation and test data.

### Experiment 1 — Preliminary Full-Feature Experiment

Script: `training.py`

Core models:

- LSTM
- TCN
- Transformer
- N-BEATS-style model

This experiment is retained as a preliminary/development experiment. Its original sequence-assignment implementation is preserved for reproducibility.

### Primary Controlled Experiment

Script: `deep_forecasting_pipeline.py`

This is the principal controlled experiment used for the main architecture comparison.

It:

- retains the full historical period;
- removes retrospectively defined disruption factors from the final modelling feature set;
- separates historically observed and known-future variables;
- creates train, validation and test windows separately;
- fits scalers on training data only;
- uses a 90-day input window and 30-day direct forecast;
- compares four deep-learning architectures against a 7-day seasonal-naive baseline.

Main models:

- LSTM
- GRU with Temporal Attention
- Dilated Residual TCN
- TFT-style forecaster

### Experiment 3 — Crisis-Adjusted Robustness Experiment

Script: `ahamedtraining.py`

This experiment excludes the 2020–2022 disruption period and adds past-only tourist-arrival lag and rolling-statistic features.

Core models:

- LSTM
- TCN
- Transformer
- N-BEATS-style model

The experiment is treated as a robustness/sensitivity analysis rather than being pooled directly with the primary experiment.

## Setup

### Recommended Environment

The project was developed primarily in Google Colab using Python and TensorFlow/Keras.

### Install Dependencies

```bash
git clone https://github.com/rashmyshiraj/Deep-Learning-Assignment.git
cd Deep-Learning-Assignment
pip install -r requirements.txt
```

### Google Colab Note

Some scripts contain `/content/...` paths because they were developed in Google Colab.

When running them in Colab, either upload/copy the required CSV files to `/content`, or update the script paths to point to the cloned repository.

For example:

```python
file_path = "/content/Sri_Lanka_Tourism_Arrivals_Encoded.csv"
```

can be changed to:

```python
file_path = "Sri_Lanka_Tourism_Arrivals_Encoded.csv"
```

when running from the repository directory.

## Execution Instructions

### Option A — Reproduce the Model Experiments

This is the simplest route because the final encoded dataset is included in the repository.

Use:

`Sri_Lanka_Tourism_Arrivals_Encoded.csv`

Then run the modelling scripts separately:

```bash
python training.py
python deep_forecasting_pipeline.py
python ahamedtraining.py
```

Each experiment is independent and intentionally uses a different experimental configuration.

### Option B — Review/Reproduce the Full Data Workflow

Start with:

`0.  Research Data.csv`

Then run/review the data-collection and preprocessing scripts in workflow order:

```bash
python data_collection_GDP_inflation.py
python exchange_rates.py
python weather_and_arrivals.py
python preprocess_tharusha1.py
python rashmy_preprocessing.py
python ahamed_preprocessing.py
python categorical_encoding_tharusha.py
```

The scripts were developed as stages of a Colab workflow and may create/use intermediate CSV files. If one stage expects a different input filename from the previous stage's output filename, update the path before execution.

Google Trends extension values included a manual data-entry stage, so the committed datasets are the authoritative copies used in the project.

## Reproducibility

See:

- `REPRODUCIBILITY.md`
- `experiment_config.json`

Random seed 42 is used where applicable. The primary experiment sets Python, NumPy and TensorFlow seeds and attempts to enable deterministic TensorFlow operations where supported.

Model-specific architecture settings, dropout values and callbacks are preserved directly in the corresponding Python scripts.

## Dependencies

See `requirements.txt`.

Main libraries include:

- pandas
- NumPy
- TensorFlow / Keras
- scikit-learn
- Matplotlib
- statsmodels
- requests
- yfinance
- pytrends

## Contributors

### Preprocessing responsibility

- A.G.T.S Ranasinghe — preprocessing stage 1
- I.R. Shiraj — preprocessing stage 2
- M.S.S. Ahamed — preprocessing stage 3

The Git history is the authoritative record of individual commits and contributions across the project.

## Important Reproducibility Note

Both the original/base CSV and the final encoded CSV are included in the repository.

Therefore, the final model experiments can be reproduced directly from the committed encoded dataset even though part of the extension workflow, including some Google Trends values, involved manual entry.
