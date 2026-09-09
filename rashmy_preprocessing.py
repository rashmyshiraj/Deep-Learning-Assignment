# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

# Load Tharusha output
step3_path = "/content/step_3_missing_values_preprocessed.csv"
if not os.path.exists(step3_path):
    raise FileNotFoundError(f"Tharusha dataset not found: {step3_path}")
df = pd.read_csv(step3_path, parse_dates=["date"])
print(f"Loaded Tharusha output: {step3_path}")

# STEP 4.1: PROPOSED COVID-19 IMPACT PERIODS
# Inspection only — this code does NOT modify df.
# ------------------------------------------------------------

covid_emerging_start = pd.Timestamp("2020-01-01")
covid_full_impact_start = pd.Timestamp("2020-04-01")
covid_full_impact_end = pd.Timestamp("2021-09-30")
covid_recovery_start = pd.Timestamp("2021-10-01")
covid_recovery_end = pd.Timestamp("2023-12-31")
covid_phase_summary = pd.DataFrame({
    "phase": [
        "Pre-COVID",
        "Emerging impact",
        "Full impact",
        "Recovery",
        "Post-recovery"
    ],
    "start_date": [
        df["date"].min(),
        covid_emerging_start,
        covid_full_impact_start,
        covid_recovery_start,
        covid_recovery_end + pd.Timedelta(days=1)
    ],
    "end_date": [
        covid_emerging_start - pd.Timedelta(days=1),
        covid_full_impact_start - pd.Timedelta(days=1),
        covid_full_impact_end,
        covid_recovery_end,
        df["date"].max()
    ],
    "expected_factor": [
        "0.0",
        "Linear increase: 0.0 to 1.0",
        "1.0",
        "Linear decrease: 1.0 to 0.0",
        "0.0"
    ]
})
print("PROPOSED COVID-19 IMPACT PHASES")
print("-" * 70)
display(covid_phase_summary)

# Inspect arrivals around critical boundaries only.
boundary_dates = [
    "2019-12-30",
    "2020-01-01",
    "2020-03-31",
    "2020-04-01",
    "2021-09-30",
    "2021-10-01",
    "2023-12-31",
    "2024-01-01"
]
print("\nDATASET RECORDS NEAR PROPOSED BOUNDARIES")
for date_value in boundary_dates:
    date_value = pd.Timestamp(date_value)
    boundary_window = df.loc[
        (df["date"] >= date_value - pd.Timedelta(days=1)) &
        (df["date"] <= date_value + pd.Timedelta(days=1)),
        ["date", "arrivals"]
    ]
    print(f"\nAround {date_value.date()}:")
    display(boundary_window)

# ------------------------------------------------------------
# STEP 4.1: CREATE COVID-19 IMPACT FACTOR
# ------------------------------------------------------------

def compute_covid_impact(date_value):
    """
    Returns a disruption factor from 0.0 to 1.0 using only calendar dates.
    0.0 = no modeled COVID-related disruption
    1.0 = maximum modeled disruption
    """
    # Phase 1: Before the emerging-impact period
    if date_value < covid_emerging_start:
        return 0.0
    # Phase 2: Linear escalation from 0.0 on 2020-01-01
    # to 1.0 on 2020-03-31.
    elif date_value < covid_full_impact_start:
        total_days = (
            covid_full_impact_start - covid_emerging_start
        ).days
        elapsed_days = (
            date_value - covid_emerging_start
        ).days
        return elapsed_days / total_days
    # Phase 3: Maximum disruption.
    elif date_value <= covid_full_impact_end:
        return 1.0
    # Phase 4: Linear recovery from 1.0 on 2021-10-01
    # to 0.0 on 2023-12-31.
    elif date_value <= covid_recovery_end:
        total_days = (
            covid_recovery_end - covid_recovery_start
        ).days
        elapsed_days = (
            date_value - covid_recovery_start
        ).days
        return 1.0 - (elapsed_days / total_days)
    # Phase 5: Post-recovery
    else:
        return 0.0

# Create the feature. This uses date only, not arrivals.
df["covid_impact_factor"] = (
    df["date"]
    .apply(compute_covid_impact)
    .round(6)
)

# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

print("COVID-19 IMPACT FACTOR VERIFICATION")
print("-" * 70)
print(f"Minimum factor value: {df['covid_impact_factor'].min():.6f}")
print(f"Maximum factor value: {df['covid_impact_factor'].max():.6f}")
print(f"Missing factor values: {df['covid_impact_factor'].isna().sum():,}")
# Verify representative boundary dates.
covid_verification_dates = [
    "2019-12-31",
    "2020-01-01",
    "2020-03-30",
    "2020-03-31",
    "2020-04-01",
    "2021-09-30",
    "2021-10-01",
    "2023-12-30",
    "2023-12-31",
    "2024-01-01"
]
print("\nBoundary-date verification:")
display(
    df.loc[
        df["date"].isin(pd.to_datetime(covid_verification_dates)),
        ["date", "covid_impact_factor"]
    ]
)
print("\nFactor value counts by broad category:")
covid_factor_category = pd.cut(
    df["covid_impact_factor"],
    bins=[-0.001, 0, 0.999999, 1.0],
    labels=["No impact (0)", "Partial impact (0 to 1)", "Full impact (1)"]
)
display(
    covid_factor_category
    .value_counts(dropna=False)
    .rename_axis("impact_category")
    .reset_index(name="row_count")
)

# ------------------------------------------------------------
# STEP 4.2: PROPOSED ECONOMIC-CRISIS IMPACT PERIODS
# Inspection only — this code does NOT modify df.
# ------------------------------------------------------------

crisis_emerging_start = pd.Timestamp("2021-04-01")
crisis_full_impact_start = pd.Timestamp("2022-04-01")
crisis_full_impact_end = pd.Timestamp("2022-12-31")
crisis_recovery_start = pd.Timestamp("2023-01-01")
crisis_recovery_end = pd.Timestamp("2024-12-31")
crisis_phase_summary = pd.DataFrame({
    "phase": [
        "Pre-crisis",
        "Emerging crisis",
        "Full crisis",
        "Recovery",
        "Post-crisis"
    ],
    "start_date": [
        df["date"].min(),
        crisis_emerging_start,
        crisis_full_impact_start,
        crisis_recovery_start,
        crisis_recovery_end + pd.Timedelta(days=1)
    ],
    "end_date": [
        crisis_emerging_start - pd.Timedelta(days=1),
        crisis_full_impact_start - pd.Timedelta(days=1),
        crisis_full_impact_end,
        crisis_recovery_end,
        df["date"].max()
    ],
    "expected_factor": [
        "0.0",
        "Linear increase: 0.0 to 1.0",
        "1.0",
        "Linear decrease: 1.0 to 0.0",
        "0.0"
    ]
})
print("PROPOSED ECONOMIC-CRISIS IMPACT PHASES")
print("-" * 70)
display(crisis_phase_summary)

# Inspect actual target values around each proposed transition.
boundary_dates = [
    "2021-03-31",
    "2021-04-01",
    "2022-03-31",
    "2022-04-01",
    "2022-12-31",
    "2023-01-01",
    "2024-12-31",
    "2025-01-01"
]
print("\nDATASET RECORDS NEAR PROPOSED CRISIS BOUNDARIES")
for date_value in boundary_dates:
    date_value = pd.Timestamp(date_value)
    boundary_window = df.loc[
        (df["date"] >= date_value - pd.Timedelta(days=1)) &
        (df["date"] <= date_value + pd.Timedelta(days=1)),
        ["date", "arrivals"]
    ]
    print(f"\nAround {date_value.date()}:")
    display(boundary_window)

# ------------------------------------------------------------
# STEP 4.2: CREATE ECONOMIC-CRISIS IMPACT FACTOR
# ------------------------------------------------------------

def compute_crisis_impact(date_value):
    """
    Returns an economic-crisis disruption factor from 0.0 to 1.0.
    The calculation depends only on the date, not on arrivals.
    """
    # Phase 1: Before the modeled economic crisis.
    if date_value < crisis_emerging_start:
        return 0.0
    # Phase 2: Linear escalation from 0.0 on 2021-04-01
    # to near 1.0 immediately before 2022-04-01.
    elif date_value < crisis_full_impact_start:
        total_days = (
            crisis_full_impact_start - crisis_emerging_start
        ).days
        elapsed_days = (
            date_value - crisis_emerging_start
        ).days
        return elapsed_days / total_days
    # Phase 3: Maximum modeled crisis impact.
    elif date_value <= crisis_full_impact_end:
        return 1.0
    # Phase 4: Linear recovery from 1.0 on 2023-01-01
    # to 0.0 on 2024-12-31.
    elif date_value <= crisis_recovery_end:
        total_days = (
            crisis_recovery_end - crisis_recovery_start
        ).days
        elapsed_days = (
            date_value - crisis_recovery_start
        ).days
        return 1.0 - (elapsed_days / total_days)
    # Phase 5: After the modeled recovery.
    else:
        return 0.0

# Create the feature using date only.
df["crisis_impact_factor"] = (
    df["date"]
    .apply(compute_crisis_impact)
    .round(6)
)

# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

print("ECONOMIC-CRISIS IMPACT FACTOR VERIFICATION")
print("-" * 70)
print(f"Minimum factor value: {df['crisis_impact_factor'].min():.6f}")
print(f"Maximum factor value: {df['crisis_impact_factor'].max():.6f}")
print(f"Missing factor values: {df['crisis_impact_factor'].isna().sum():,}")
crisis_verification_dates = [
    "2021-03-31",
    "2021-04-01",
    "2022-03-30",
    "2022-03-31",
    "2022-04-01",
    "2022-12-31",
    "2023-01-01",
    "2024-12-30",
    "2024-12-31",
    "2025-01-01"
]
print("\nBoundary-date verification:")
display(
    df.loc[
        df["date"].isin(pd.to_datetime(crisis_verification_dates)),
        ["date", "crisis_impact_factor"]
    ]
)
print("\nFactor value counts by broad category:")
crisis_factor_category = pd.cut(
    df["crisis_impact_factor"],
    bins=[-0.001, 0, 0.999999, 1.0],
    labels=["No impact (0)", "Partial impact (0 to 1)", "Full impact (1)"]
)
display(
    crisis_factor_category
    .value_counts(dropna=False)
    .rename_axis("impact_category")
    .reset_index(name="row_count")
)
