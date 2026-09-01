# %%
import requests
import pandas as pd
import time

# %%
url = "https://api.worldbank.org/v2/country/LKA/indicator/NY.GDP.PCAP.CD"

params = {
    "format": "json",
    "per_page": 100
}

response = requests.get(url, params=params, timeout=60)

print("Status code:", response.status_code)
print("URL:", response.url)

data = response.json()

print("Number of records:", len(data[1]))
print(data[1][:3])

# %%
url = "https://api.worldbank.org/v2/country/LKA/indicator/FP.CPI.TOTL.ZG"

params = {
    "format": "json",
    "per_page": 100
}

response = requests.get(url, params=params, timeout=60)

print("Status code:", response.status_code)

data = response.json()

print("Number of records:", len(data[1]))
print(data[1][:3])

# %%
import requests
import pandas as pd

def get_world_bank_indicator(indicator_code):
    url = f"https://api.worldbank.org/v2/country/LKA/indicator/{indicator_code}"

    params = {
        "format": "json",
        "per_page": 100
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    records = data[1]

    result = {}

    for record in records:
        year = int(record["date"])
        value = record["value"]

        if value is not None:
            result[year] = float(value)

    return result


gdp_data = get_world_bank_indicator("NY.GDP.PCAP.CD")
inflation_data = get_world_bank_indicator("FP.CPI.TOTL.ZG")

print("GDP:")
print(gdp_data)

print("\nInflation:")
print(inflation_data)

# %%
import pandas as pd

dates = pd.date_range(
    start="2025-09-10",
    end="2026-07-31",
    freq="D"
)

df_wb = pd.DataFrame({
    "date": dates
})

print(df_wb.head())
print(df_wb.tail())
print("Number of dates:", len(df_wb))

# %%
import requests
import pandas as pd
import time

def get_world_bank_indicator(indicator_code):
    url = f"https://api.worldbank.org/v2/country/LKA/indicator/{indicator_code}"

    params = {
        "format": "json",
        "per_page": 100
    }

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            if len(data) < 2 or data[1] is None:
                raise ValueError("World Bank returned no indicator data.")

            records = data[1]

            result = {}

            for record in records:
                year = int(record["date"])
                value = record["value"]

                if value is not None:
                    result[year] = float(value)

            return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < 2:
                time.sleep(5)

    raise RuntimeError(
        f"Could not retrieve World Bank indicator: {indicator_code}"
    )


# GDP per capita
gdp_data = get_world_bank_indicator(
    "NY.GDP.PCAP.CD"
)

# Inflation
inflation_data = get_world_bank_indicator(
    "FP.CPI.TOTL.ZG"
)

print("GDP years available:")
print(sorted(gdp_data.keys(), reverse=True))

print("\nInflation years available:")
print(sorted(inflation_data.keys(), reverse=True))

# %%
df_wb["year"] = df_wb["date"].dt.year

df_wb["gdp_per_capita"] = df_wb["year"].map(gdp_data)
df_wb["inflation_rate"] = df_wb["year"].map(inflation_data)

df_wb = df_wb.drop(columns=["year"])

print(df_wb.head())
print(df_wb.tail())

# %%
# Create the required date range
df_wb = pd.DataFrame({
    "date": pd.date_range(
        start="2025-09-10",
        end="2026-07-31",
        freq="D"
    )
})

# Get the year for each date
df_wb["year"] = df_wb["date"].dt.year

# Map World Bank annual values to each date
df_wb["gdp_per_capita"] = df_wb["year"].map(gdp_data)
df_wb["inflation_rate"] = df_wb["year"].map(inflation_data)

# Remove helper column
df_wb = df_wb.drop(columns=["year"])

# Format date exactly like your existing dataset: M/D/YYYY
df_wb["date"] = df_wb["date"].dt.strftime("%-m/%-d/%Y")

print(df_wb.head(10))
print()
print(df_wb.tail(10))
print()
print("Rows:", len(df_wb))

# %%
print(df_wb[df_wb["date"].isin([
    "9/10/2025",
    "12/31/2025",
    "1/1/2026",
    "7/31/2026"
])])

# %%
# Save the World Bank data
output_file = "world_bank_sri_lanka_2025_2026.csv"

df_wb.to_csv(
    output_file,
    index=False,
    na_rep="null"
)

print(f"Saved: {output_file}")

# %%
# Read the CSV back and verify it
test_df = pd.read_csv(
    output_file,
    na_values=["null"]
)

print(test_df.head())
print()
print(test_df.tail())
print()
print("Rows:", len(test_df))
print("Columns:", list(test_df.columns))
print()
print("Missing values:")
print(test_df.isna().sum())

# %%
from google.colab import files

files.download("world_bank_sri_lanka_2025_2026.csv")


