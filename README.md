# Deep Learning Assignment - Data Cleaning & Preprocessing Pipeline

This repository contains data preprocessing scripts for the tourist arrivals forecasting model.

## Pipeline Structure
- `preprocess_tharusha1.py`: Core cleaning, validation, and leakage-safe imputation module.

## Workflow Overview
1. **Initial Inspection & Validation**: Validates date formats, target column integrity, and checks constant features.
2. **Date Alignment & Sorting**: Ensures strict chronological order with daily frequency.
3. **Imputation**:
   - Past-only forward fill for exchange rates.
   - Forward fill for macroeconomic indicators across forecast horizons.
4. **Quality Checks**: Outlier and missingness checks on weather features.