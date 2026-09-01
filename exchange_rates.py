# %%

import pandas as pd

# Load the CBSL exchange-rate dataset
cbsl = pd.read_csv("/content/data.csv")

# Show the first few rows
print(cbsl.head())
print("\nColumns:")
print(cbsl.columns.tolist())

# %%
# Clean column names
cbsl.columns = cbsl.columns.str.strip()

# Clean currency names/codes
cbsl["Currency"] = cbsl["Currency"].str.strip().str.upper()

# Convert date
cbsl["Date"] = pd.to_datetime(cbsl["Date"], errors="coerce")

# Convert exchange rate to numeric
cbsl["Exchange Rate"] = pd.to_numeric(
    cbsl["Exchange Rate"],
    errors="coerce"
)

# Keep only the currencies needed for our dataset
currencies = ["CNY", "EUR", "GBP", "INR", "RUB", "USD"]

cbsl = cbsl[cbsl["Currency"].isin(currencies)].copy()

# Convert from:
# Currency | Date | Exchange Rate
#
# into:
# Date | CNY | EUR | GBP | INR | RUB | USD

exchange_rates = cbsl.pivot(
    index="Date",
    columns="Currency",
    values="Exchange Rate"
).reset_index()

# Rename columns
exchange_rates = exchange_rates.rename(columns={
    "Date": "date",
    "USD": "usd_lkr",
    "RUB": "rub_lkr",
    "INR": "inr_lkr",
    "GBP": "gbp_lkr",
    "EUR": "eur_lkr",
    "CNY": "cny_lkr"
})

# Format date as M/D/YYYY
exchange_rates["date"] = exchange_rates["date"].dt.strftime("%-m/%-d/%Y")

# Exact column order
exchange_rates = exchange_rates[
    [
        "date",
        "usd_lkr",
        "rub_lkr",
        "inr_lkr",
        "gbp_lkr",
        "eur_lkr",
        "cny_lkr"
    ]
]

print(exchange_rates.head(10).to_string(index=False))
print("\nLast 10 rows:")
print(exchange_rates.tail(10).to_string(index=False))

# %%
import os

print(os.listdir("/content"))

# %%
import os

print(os.listdir("/content"))

# %%
import pandas as pd

# Load your main tourism dataset
main_df = pd.read_csv("/content/0.  Research Data.csv")

print("Shape:", main_df.shape)
print("\nColumns:")
print(main_df.columns.tolist())

print("\nLast 10 rows:")
print(main_df.tail(10).to_string(index=False))

# %%
print("First date:", main_df["date"].iloc[0])
print("Last date:", main_df["date"].iloc[-1])

print("\nNumber of rows:", len(main_df))

# %%
import pandas as pd

# --------------------------------------------------
# 1. Load both datasets
# --------------------------------------------------

main_df = pd.read_csv("/content/0.  Research Data.csv")
cbsl = pd.read_csv("/content/data.csv")


# --------------------------------------------------
# 2. Prepare dates
# --------------------------------------------------

main_df["_date"] = pd.to_datetime(
    main_df["date"],
    format="%m/%d/%Y"
)

cbsl["Date"] = pd.to_datetime(
    cbsl["Date"],
    errors="coerce"
)


# --------------------------------------------------
# 3. Clean CBSL data
# --------------------------------------------------

cbsl["Currency"] = cbsl["Currency"].str.strip().str.upper()

cbsl["Exchange Rate"] = pd.to_numeric(
    cbsl["Exchange Rate"],
    errors="coerce"
)

# Keep only our six currencies
cbsl = cbsl[
    cbsl["Currency"].isin(
        ["USD", "RUB", "INR", "GBP", "EUR", "CNY"]
    )
].copy()


# --------------------------------------------------
# 4. Convert CBSL data from long → wide
# --------------------------------------------------

exchange_rates = cbsl.pivot(
    index="Date",
    columns="Currency",
    values="Exchange Rate"
).reset_index()

exchange_rates = exchange_rates.rename(columns={
    "Date": "_date",
    "USD": "usd_lkr",
    "RUB": "rub_lkr",
    "INR": "inr_lkr",
    "GBP": "gbp_lkr",
    "EUR": "eur_lkr",
    "CNY": "cny_lkr"
})


# --------------------------------------------------
# 5. Only use CBSL data for the new period
# --------------------------------------------------

exchange_rates = exchange_rates[
    (exchange_rates["_date"] >= "2025-09-19") &
    (exchange_rates["_date"] <= "2026-07-31")
].copy()


# --------------------------------------------------
# 6. Merge CBSL rates into the main dataset
# --------------------------------------------------

rate_columns = [
    "usd_lkr",
    "rub_lkr",
    "inr_lkr",
    "gbp_lkr",
    "eur_lkr",
    "cny_lkr"
]

main_df = main_df.merge(
    exchange_rates[["_date"] + rate_columns],
    on="_date",
    how="left",
    suffixes=("", "_cbsl")
)


# --------------------------------------------------
# 7. Use CBSL values for 9/19/2025 onward
# --------------------------------------------------

extension_start = pd.Timestamp("2025-09-19")

for col in rate_columns:
    main_df.loc[
        main_df["_date"] >= extension_start,
        col
    ] = main_df.loc[
        main_df["_date"] >= extension_start,
        col + "_cbsl"
    ]


# --------------------------------------------------
# 8. Remove temporary columns
# --------------------------------------------------

temporary_columns = [
    col + "_cbsl"
    for col in rate_columns
]

main_df.drop(columns=temporary_columns, inplace=True)

main_df.drop(columns=["_date"], inplace=True)


# --------------------------------------------------
# 9. Restore the original column order
# --------------------------------------------------

original_columns = [
    "date",
    "arrivals",
    "gdp_per_capita",
    "inflation_rate",
    "brent_crude_price",
    "usd_lkr",
    "rub_lkr",
    "inr_lkr",
    "gbp_lkr",
    "eur_lkr",
    "cny_lkr",
    "web_search",
    "youtube_search",
    "image_search",
    "is_holiday",
    "holiday_name",
    "event_name",
    "is_tourist_event",
    "location",
    "temperature",
    "precipitation",
    "humidity",
    "wind_speed"
]

main_df = main_df[original_columns]


# --------------------------------------------------
# 10. Check the result
# --------------------------------------------------

print("Shape:", main_df.shape)

print("\nLast 15 rows:")
print(
    main_df.tail(15).to_string(index=False)
)

# %%
main_df.to_csv(
    "/content/0. Research Data - exchange rates updated.csv",
    index=False
)

print("Saved successfully.")

# %%
check = pd.read_csv(
    "/content/0. Research Data - exchange rates updated.csv"
)

print("Shape:", check.shape)
print("Last date:", check["date"].iloc[-1])
print(check.tail(3).to_string(index=False))

# %%
from google.colab import files

files.download("/content/0. Research Data - exchange rates updated.csv")

# %%
main_df.to_csv(
    "/content/0. Research Data - exchange rates updated.csv",
    index=False,
    na_rep="null"
)

print("Saved with missing values written as 'null'.")

# %%
with open("/content/0. Research Data - exchange rates updated.csv", "r") as f:
    for _ in range(3):
        print(f.readline().strip())

# %%


# %%
from google.colab import files

files.download("/content/0. Research Data - exchange rates updated.csv")


