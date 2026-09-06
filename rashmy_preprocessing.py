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

# ------------------------------------------------------------
# STEP 4.3: SAFE CATEGORICAL PREPARATION AND DATE-BASED FEATURES
# ------------------------------------------------------------
# 1. Preserve original missingness as audit flags before filling labels.
df["event_name_was_missing"] = df["event_name"].isna().astype("int8")
df["holiday_name_was_missing"] = df["holiday_name"].isna().astype("int8")

# 2. Replace verified "no event" and "no holiday" missing labels.
df["event_name"] = df["event_name"].fillna("NoEvent")
df["holiday_name"] = df["holiday_name"].fillna("NoHoliday")

# 3. Convert Boolean flags to compact integer form.
df["is_tourist_event"] = df["is_tourist_event"].astype("int8")
df["is_holiday"] = df["is_holiday"].astype("int8")

# 4. Create a rainfall indicator using the threshold in your plan.
#    A value of 1 indicates rainfall greater than 0.5 units.
df["is_rainy_day"] = (df["precipitation"] > 0.5).astype("int8")

# 5. Identify actual holiday dates from the existing holiday indicator.
holiday_dates = df.loc[
    df["is_holiday"] == 1,
    "date"
].sort_values().reset_index(drop=True)

# 6. For every record, calculate days since the most recent holiday.
last_holiday_date = df["date"].where(
    df["is_holiday"] == 1
).ffill()
df["days_since_last_holiday"] = (
    df["date"] - last_holiday_date
).dt.days

# 7. For every record, calculate days until the next holiday.
#    bfill is used here only on the known holiday calendar to construct
#    a calendar-proximity feature; it does not use arrivals or target values.
next_holiday_date = df["date"].where(
    df["is_holiday"] == 1
).bfill()
df["days_to_next_holiday"] = (
    next_holiday_date - df["date"]
).dt.days

# 8. Mark a ±3-day holiday window.
df["in_holiday_window"] = (
    (
        df["days_since_last_holiday"].between(0, 3)
    )
    |
    (
        df["days_to_next_holiday"].between(0, 3)
    )
).astype("int8")

# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

print("CATEGORICAL AND CALENDAR FEATURE VERIFICATION")
print("-" * 70)
print("\nRemaining missing event names:", df["event_name"].isna().sum())
print("Remaining missing holiday names:", df["holiday_name"].isna().sum())
print("\nEvent-label counts:")
display(
    df["event_name"]
    .value_counts()
    .head(10)
    .rename_axis("event_name")
    .reset_index(name="row_count")
)
print("\nHoliday-label counts:")
display(
    df["holiday_name"]
    .value_counts()
    .head(10)
    .rename_axis("holiday_name")
    .reset_index(name="row_count")
)
print("\nBinary feature counts:")
binary_columns = [
    "is_holiday",
    "is_tourist_event",
    "is_rainy_day",
    "in_holiday_window"
]
for column in binary_columns:
    print(f"\n{column}:")
    display(
        df[column]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(column)
        .reset_index(name="row_count")
    )
print("\nHoliday proximity examples:")
display(
    df.loc[
        (df["is_holiday"] == 1) |
        (df["in_holiday_window"] == 1),
        [
            "date",
            "holiday_name",
            "is_holiday",
            "days_since_last_holiday",
            "days_to_next_holiday",
            "in_holiday_window"
        ]
    ].head(30)
)
print("\nFirst rows where days_since_last_holiday is unavailable:")
display(
    df.loc[
        df["days_since_last_holiday"].isna(),
        [
            "date",
            "holiday_name",
            "is_holiday",
            "days_since_last_holiday",
            "days_to_next_holiday"
        ]
    ].head(20)
)

# ------------------------------------------------------------
# STEP 4.3: FINALIZE HOLIDAY-DISTANCE FEATURES
# ------------------------------------------------------------

holiday_distance_columns = [
    "days_since_last_holiday",
    "days_to_next_holiday"
]

# 1. Preserve whether each holiday-distance value was originally unavailable.
for column in holiday_distance_columns:
    df[f"{column}_was_missing"] = df[column].isna().astype("int8")
# 2. Count missing values before finalization.
holiday_distance_missing_before = df[holiday_distance_columns].isna().sum()
# 3. Use -1 only where no previous/next holiday is available
#    inside the dataset boundary. Do not use future target information.
df[holiday_distance_columns] = df[holiday_distance_columns].fillna(-1)
# 4. Convert values to compact integers because they represent day counts.
df[holiday_distance_columns] = df[holiday_distance_columns].astype("int16")
# 5. Verify the completed columns.
holiday_distance_missing_after = df[holiday_distance_columns].isna().sum()
holiday_distance_check = pd.DataFrame({
    "missing_before": holiday_distance_missing_before,
    "missing_after": holiday_distance_missing_after,
    "filled_with_minus_1": (
        df[holiday_distance_columns] == -1
    ).sum()
})
print("HOLIDAY-DISTANCE FINALIZATION RESULTS")
print("-" * 70)
display(holiday_distance_check)
print("\nRows using -1 because holiday history/calendar is unavailable:")
display(
    df.loc[
        (df["days_since_last_holiday"] == -1) |
        (df["days_to_next_holiday"] == -1),
        [
            "date",
            "holiday_name",
            "is_holiday",
            "days_since_last_holiday",
            "days_to_next_holiday",
            "days_since_last_holiday_was_missing",
            "days_to_next_holiday_was_missing"
        ]
    ]
)
print("\nFirst 15 rows after finalization:")
display(
    df[
        [
            "date",
            "holiday_name",
            "is_holiday",
            "days_since_last_holiday",
            "days_to_next_holiday",
            "in_holiday_window"
        ]
    ].head(15)
)
print("\nLast 15 rows after finalization:")
display(
    df[
        [
            "date",
            "holiday_name",
            "is_holiday",
            "days_since_last_holiday",
            "days_to_next_holiday",
            "in_holiday_window"
        ]
    ].tail(15)
)

# ------------------------------------------------------------
# STEP 4.3: INSPECT CATEGORY CARDINALITY AND RARE LABELS
# Inspection only — this code does NOT modify df.
# ------------------------------------------------------------

event_category_counts = (
    df["event_name"]
    .value_counts(dropna=False)
    .rename_axis("event_name")
    .reset_index(name="row_count")
)
holiday_category_counts = (
    df["holiday_name"]
    .value_counts(dropna=False)
    .rename_axis("holiday_name")
    .reset_index(name="row_count")
)
print("EVENT CATEGORY SUMMARY")
print("-" * 70)
print(f"Total unique event labels: {len(event_category_counts):,}")
print(f"Event labels occurring fewer than 5 times: {(event_category_counts['row_count'] < 5).sum():,}")
print(f"Rows belonging to event labels occurring fewer than 5 times: {event_category_counts.loc[event_category_counts['row_count'] < 5, 'row_count'].sum():,}")
print("\nAll rare event labels: fewer than 5 rows")
display(
    event_category_counts[
        event_category_counts["row_count"] < 5
    ].sort_values("row_count")
)
print("\n" + "=" * 70)
print("HOLIDAY CATEGORY SUMMARY")
print("-" * 70)
print(f"Total unique holiday labels: {len(holiday_category_counts):,}")
print(f"Holiday labels occurring fewer than 5 times: {(holiday_category_counts['row_count'] < 5).sum():,}")
print(f"Rows belonging to holiday labels occurring fewer than 5 times: {holiday_category_counts.loc[holiday_category_counts['row_count'] < 5, 'row_count'].sum():,}")
print("\nAll rare holiday labels: fewer than 5 rows")
display(
    holiday_category_counts[
        holiday_category_counts["row_count"] < 5
    ].sort_values("row_count")
)
print("\nCATEGORY COUNT DISTRIBUTION")
print("-" * 70)
category_distribution = pd.DataFrame({
    "event_name": [
        (event_category_counts["row_count"] == 1).sum(),
        ((event_category_counts["row_count"] >= 2) & (event_category_counts["row_count"] <= 4)).sum(),
        ((event_category_counts["row_count"] >= 5) & (event_category_counts["row_count"] <= 9)).sum(),
        (event_category_counts["row_count"] >= 10).sum()
    ],
    "holiday_name": [
        (holiday_category_counts["row_count"] == 1).sum(),
        ((holiday_category_counts["row_count"] >= 2) & (holiday_category_counts["row_count"] <= 4)).sum(),
        ((holiday_category_counts["row_count"] >= 5) & (holiday_category_counts["row_count"] <= 9)).sum(),
        (holiday_category_counts["row_count"] >= 10).sum()
    ]
}, index=[
    "Exactly 1 occurrence",
    "2 to 4 occurrences",
    "5 to 9 occurrences",
    "10 or more occurrences"
])
display(category_distribution)

# ------------------------------------------------------------
# STEP 4.3: CREATE RARE-CATEGORY GROUPS
# ------------------------------------------------------------

rare_threshold = 5

# Identify event names that occur fewer than 5 times.
rare_event_labels = event_category_counts.loc[
    (event_category_counts["row_count"] < rare_threshold) &
    (event_category_counts["event_name"] != "NoEvent"),
    "event_name"
].tolist()

# Identify holiday names that occur fewer than 5 times.
rare_holiday_labels = holiday_category_counts.loc[
    (holiday_category_counts["row_count"] < rare_threshold) &
    (holiday_category_counts["holiday_name"] != "NoHoliday"),
    "holiday_name"
].tolist()

# Create grouped categorical fields.
df["event_name_grouped"] = df["event_name"].where(
    ~df["event_name"].isin(rare_event_labels),
    "OtherEvent"
)
df["holiday_name_grouped"] = df["holiday_name"].where(
    ~df["holiday_name"].isin(rare_holiday_labels),
    "OtherHoliday"
)

# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

print("RARE-CATEGORY GROUPING VERIFICATION")
print("-" * 70)
print(f"Rare-event threshold: fewer than {rare_threshold} rows")
print(f"Rare event labels grouped: {len(rare_event_labels):,}")
print(f"Rare holiday labels grouped: {len(rare_holiday_labels):,}")
print("\nEvent categories before and after grouping:")
print(f"Before grouping: {df['event_name'].nunique(dropna=False):,}")
print(f"After grouping: {df['event_name_grouped'].nunique(dropna=False):,}")
print("\nHoliday categories before and after grouping:")
print(f"Before grouping: {df['holiday_name'].nunique(dropna=False):,}")
print(f"After grouping: {df['holiday_name_grouped'].nunique(dropna=False):,}")
print("\nGrouped event category counts:")
display(
    df["event_name_grouped"]
    .value_counts()
    .rename_axis("event_name_grouped")
    .reset_index(name="row_count")
    .tail(15)
)
print("\nGrouped holiday category counts:")
display(
    df["holiday_name_grouped"]
    .value_counts()
    .rename_axis("holiday_name_grouped")
    .reset_index(name="row_count")
    .tail(20)
)
print("\nVerification of special categories:")
special_category_check = pd.DataFrame({
    "category": [
        "NoEvent",
        "OtherEvent",
        "NoHoliday",
        "OtherHoliday"
    ],
    "row_count": [
        (df["event_name_grouped"] == "NoEvent").sum(),
        (df["event_name_grouped"] == "OtherEvent").sum(),
        (df["holiday_name_grouped"] == "NoHoliday").sum(),
        (df["holiday_name_grouped"] == "OtherHoliday").sum()
    ]
})
display(special_category_check)

# ------------------------------------------------------------
# Save Rashmy output for ahamed
step4_path = "/content/step_4_feature_engineering_pre_target_encoding.csv"
df.to_csv(step4_path, index=False, date_format="%Y-%m-%d")
print(f"Saved Rashmy output: {step4_path}")
