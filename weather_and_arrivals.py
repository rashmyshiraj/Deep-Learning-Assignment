# %%
import pandas as pd

# Load the current working dataset
main_df = pd.read_csv(
    "/content/0. Research Data - exchange rates updated.csv"
)

print("Shape:", main_df.shape)
print("Date range:", main_df["date"].iloc[0], "to", main_df["date"].iloc[-1])

# %%
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 6.9271,
    "longitude": 79.8612,
    "start_date": "2025-09-19",
    "end_date": "2026-07-31",

    "daily": ",".join([
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "relative_humidity_2m_mean"
    ]),

    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",

    # Important: daily values should use Colombo/Sri Lankan local time
    "timezone": "Asia/Colombo"
}

response = requests.get(url, params=params, timeout=60)

print("Status code:", response.status_code)

response.raise_for_status()

weather_json = response.json()

print(weather_json.keys())

# %%
weather_df = pd.DataFrame({
    "date": weather_json["daily"]["time"],
    "temperature": weather_json["daily"]["temperature_2m_mean"],
    "precipitation": weather_json["daily"]["precipitation_sum"],
    "humidity": weather_json["daily"]["relative_humidity_2m_mean"],
    "wind_speed": weather_json["daily"]["wind_speed_10m_max"]
})

print(weather_df.head(10).to_string(index=False))

# %%
print("Weather rows:", len(weather_df))
print("First date:", weather_df["date"].iloc[0])
print("Last date:", weather_df["date"].iloc[-1])

print("\nMissing values:")
print(weather_df.isna().sum())

# %%
print("\nFirst 5:")
print(weather_df.head().to_string(index=False))

print("\nLast 5:")
print(weather_df.tail().to_string(index=False))

# %%
# Convert both date columns to datetime for matching
main_df["_date"] = pd.to_datetime(main_df["date"], format="%m/%d/%Y")
weather_df["_date"] = pd.to_datetime(weather_df["date"])

# Merge weather data
main_df = main_df.merge(
    weather_df[
        ["_date", "temperature", "precipitation", "humidity", "wind_speed"]
    ],
    on="_date",
    how="left",
    suffixes=("", "_weather")
)

# Only replace the weather values for the extension period
extension_start = pd.Timestamp("2025-09-19")

weather_columns = [
    "temperature",
    "precipitation",
    "humidity",
    "wind_speed"
]

for col in weather_columns:
    main_df.loc[
        main_df["_date"] >= extension_start,
        col
    ] = main_df.loc[
        main_df["_date"] >= extension_start,
        col + "_weather"
    ]

# Remove temporary columns
main_df.drop(
    columns=[col + "_weather" for col in weather_columns],
    inplace=True
)

main_df.drop(columns=["_date"], inplace=True)

print("Shape:", main_df.shape)
print("Date range:", main_df["date"].iloc[0], "to", main_df["date"].iloc[-1])

# %%
print(
    main_df[
        main_df["date"].isin([
            "9/19/2025",
            "9/20/2025",
            "7/30/2026",
            "7/31/2026"
        ])
    ][
        ["date", "temperature", "precipitation", "humidity", "wind_speed"]
    ].to_string(index=False)
)

# %%
import pandas as pd

arrivals_sep_2025 = pd.DataFrame({
    "date": pd.date_range("2025-09-19", "2025-09-30"),
    "arrivals": [
        5389,
        4642,
        5954,
        4614,
        5379,
        4561,
        4666,
        5891,
        5588,
        6150,
        4959,
        5339
    ]
})

print(arrivals_sep_2025.to_string(index=False))

# %%
print("Rows:", len(arrivals_sep_2025))
print("First:", arrivals_sep_2025["date"].min())
print("Last:", arrivals_sep_2025["date"].max())
print("Total:", arrivals_sep_2025["arrivals"].sum())

# %%
arrivals_oct_2025 = pd.DataFrame({
    "date": pd.date_range("2025-10-01", "2025-10-31"),
    "arrivals": [
        5784,
        6078,
        5998,
        6274,
        5874,
        5891,
        6096,
        5891,
        5888,
        5973,
        5954,
        6198,
        6117,
        6263,
        6417,
        6078,
        6062,
        6275,
        6073,
        6045,
        6366,
        6372,
        6475,
        6488,
        6428,
        6425,
        6540,
        6574,
        6690,
        6710,
        6598
    ]
})

print("Rows:", len(arrivals_oct_2025))
print("First:", arrivals_oct_2025["date"].min())
print("Last:", arrivals_oct_2025["date"].max())
print("Total:", arrivals_oct_2025["arrivals"].sum())

# %%
arrivals_nov_2025 = pd.DataFrame({
    "date": pd.date_range("2025-11-01", "2025-11-30"),
    "arrivals": [
        6502,
        6523,
        6541,
        6678,
        6458,
        6591,
        6712,
        6684,
        6597,
        6734,
        6645,
        6789,
        6711,
        6823,
        6756,
        6891,
        6812,
        6945,
        6876,
        7012,
        6954,
        7089,
        7018,
        7134,
        7067,
        7211,
        7145,
        7289,
        7218,
        7356
    ]
})

print("Rows:", len(arrivals_nov_2025))
print("First:", arrivals_nov_2025["date"].min())
print("Last:", arrivals_nov_2025["date"].max())
print("Total:", arrivals_nov_2025["arrivals"].sum())

# %%
arrivals_oct_2025 = pd.DataFrame({
    "date": pd.date_range("2025-10-01", "2025-10-31"),
    "arrivals": [
        6751, 6021, 6146, 5802, 4847, 4478, 4429,
        4216, 4177, 5259, 5378, 5235, 4376, 4098,
        4442, 4911, 5768, 6552, 5976, 4626, 5041,
        5535, 5604, 7145, 5443, 5618, 4583, 5577,
        5028, 5216, 6915
    ]
})

print("Rows:", len(arrivals_oct_2025))
print("First:", arrivals_oct_2025["date"].min())
print("Last:", arrivals_oct_2025["date"].max())
print("Total:", arrivals_oct_2025["arrivals"].sum())

# %%
arrivals_nov_2025 = pd.DataFrame({
    "date": pd.date_range("2025-11-01", "2025-11-30"),
    "arrivals": [
        7412, 7205, 6173, 6330, 5695, 7021, 7159,
        7742, 7153, 7292, 7102, 5986, 6922, 6510,
        7463, 7558, 6757, 7245, 8058, 8530, 7562,
        7764, 7653, 6856, 6512, 6272, 7425, 6410,
        6793, 8346
    ]
})

print("Rows:", len(arrivals_nov_2025))
print("First:", arrivals_nov_2025["date"].min())
print("Last:", arrivals_nov_2025["date"].max())
print("Total:", arrivals_nov_2025["arrivals"].sum())

# %%
arrivals_dec_2025 = pd.DataFrame({
    "date": pd.date_range("2025-12-01", "2025-12-31"),
    "arrivals": [
        6328,
        8178,
        5221,
        5319,
        5927,
        6026,
        6977,
        6245,
        6768,
        6271,
        7610,
        6763,
        7461,
        7936,
        7305,
        7720,
        9142,
        7561,
        9525,
        10081,
        10245,
        10639,
        9215,
        10281,
        11111,
        12397,
        11203,
        11756,
        10177,
        10425,
        7118
    ]
})

print("Rows:", len(arrivals_dec_2025))
print("First:", arrivals_dec_2025["date"].min())
print("Last:", arrivals_dec_2025["date"].max())
print("Total:", arrivals_dec_2025["arrivals"].sum())

# %%
arrivals_jan_2026 = pd.DataFrame({

    "date": pd.date_range("2026-01-01", "2026-01-31"),

    "arrivals": [
        8377,
        8858,
        8344,
        7497,
        7987,
        9275,
        8484,
        8940,
        8514,
        8789,
        8974,
        10005,
        9350,
        8018,
        10483,
        8699,
        8986,
        9206,
        8846,
        9111,
        8746,
        9637,
        10068,
        9863,
        9157,
        9203,
        8369,
        8172,
        9447,
        8648,
        9274
    ]

})

print("Rows:", len(arrivals_jan_2026))
print("First:", arrivals_jan_2026["date"].min())
print("Last:", arrivals_jan_2026["date"].max())
print("Total:", arrivals_jan_2026["arrivals"].sum())

# %%
arrivals_feb_2026 = pd.DataFrame({

    "date": pd.date_range("2026-02-01", "2026-02-28"),

    "arrivals": [
        9752,
        11817,
        9650,
        8835,
        9918,
        9660,
        10723,
        10420,
        9846,
        10746,
        12731,
        11201,
        10497,
        11114,
        12565,
        12070,
        10199,
        8927,
        9462,
        9376,
        9953,
        9097,
        8803,
        8673,
        7393,
        8503,
        8505,
        9027
    ]

})

print("Rows:", len(arrivals_feb_2026))
print("First:", arrivals_feb_2026["date"].min())
print("Last:", arrivals_feb_2026["date"].max())
print("Total:", arrivals_feb_2026["arrivals"].sum())

# %%
arrivals_mar_2026 = pd.DataFrame({

    "date": pd.date_range("2026-03-01", "2026-03-31"),

    "arrivals": [
        5426,
        6376,
        5651,
        5663,
        6285,
        5596,
        6278,
        6370,
        6350,
        6493,
        6509,
        5612,
        5882,
        7318,
        6762,
        5115,
        5890,
        5844,
        6858,
        6540,
        6752,
        6519,
        5234,
        5480,
        4890,
        6182,
        5567,
        6523,
        5696,
        4455,
        3863
    ]

})

print("Rows:", len(arrivals_mar_2026))
print("First:", arrivals_mar_2026["date"].min())
print("Last:", arrivals_mar_2026["date"].max())
print("Total:", arrivals_mar_2026["arrivals"].sum())

# %%
arrivals_apr_2026 = pd.DataFrame({

    "date": pd.date_range("2026-04-01", "2026-04-30"),

    "arrivals": [
        4534,
        6624,
        5484,
        5534,
        4624,
        4021,
        4691,
        4115,
        5150,
        4459,
        5091,
        3873,
        3408,
        3779,
        4565,
        4179,
        4656,
        5022,
        4890,
        3977,
        3677,
        3424,
        4467,
        4476,
        5111,
        4062,
        3449,
        3991,
        4270,
        6040
    ]

})

print("Rows:", len(arrivals_apr_2026))
print("First:", arrivals_apr_2026["date"].min())
print("Last:", arrivals_apr_2026["date"].max())
print("Total:", arrivals_apr_2026["arrivals"].sum())

# %%
arrivals_may_2026 = pd.DataFrame({

    "date": pd.date_range("2026-05-01", "2026-05-31"),

    "arrivals": [
        5656,
        5383,
        5333,
        3935,
        4023,
        4248,
        4937,
        4002,
        4457,
        4631,
        3992,
        3970,
        3742,
        4547,
        4190,
        4504,
        3916,
        3426,
        5913,
        3711,
        5315,
        6263,
        7310,
        6350,
        5468,
        5593,
        4645,
        5070,
        3913,
        3875,
        3427
    ]

})

print("Rows:", len(arrivals_may_2026))
print("First:", arrivals_may_2026["date"].min())
print("Last:", arrivals_may_2026["date"].max())
print("Total:", arrivals_may_2026["arrivals"].sum())

# %%
arrivals_june_2026 = pd.DataFrame({

    "date": pd.date_range("2026-06-01", "2026-06-30"),

    "arrivals": [
        4037,
        3693,
        3523,
        4068,
        4024,
        3990,
        3708,
        3475,
        3576,
        3367,
        3985,
        4498,
        4442,
        4080,
        3702,
        3798,
        3714,
        3909,
        4062,
        5173,
        4227,
        4307,
        4476,
        3913,
        5060,
        5421,
        5016,
        4326,
        4578,
        4403
    ]

})

print("Rows:", len(arrivals_june_2026))
print("First:", arrivals_june_2026["date"].min())
print("Last:", arrivals_june_2026["date"].max())
print("Total:", arrivals_june_2026["arrivals"].sum())

# %%
arrivals_july_2026 = pd.DataFrame({

    "date": pd.date_range("2026-07-01", "2026-07-31"),

    "arrivals": [
        4702,
        6293,
        5800,
        6291,
        5510,
        6125,
        6100,
        5312,
        6394,
        6443,
        6635,
        5777,
        6509,
        6316,
        5929,
        7217,
        7372,
        7690,
        6519,
        6934,
        6500,
        6361,
        7827,
        7464,
        7371,
        6009,
        6156,
        5739,
        5287,
        5873,
        6390
    ]

})

print("Rows:", len(arrivals_july_2026))
print("First:", arrivals_july_2026["date"].min())
print("Last:", arrivals_july_2026["date"].max())
print("Total:", arrivals_july_2026["arrivals"].sum())

# %%
arrivals_df = pd.concat([
    arrivals_sep_2025,
    arrivals_oct_2025,
    arrivals_nov_2025,
    arrivals_dec_2025,
    arrivals_jan_2026,
    arrivals_feb_2026,
    arrivals_mar_2026,
    arrivals_apr_2026,
    arrivals_may_2026,
    arrivals_june_2026,
    arrivals_july_2026
], ignore_index=True)

print("Shape:", arrivals_df.shape)
print("First date:", arrivals_df["date"].iloc[0])
print("Last date:", arrivals_df["date"].iloc[-1])

# %%
print("Duplicate dates:", arrivals_df["date"].duplicated().sum())

expected_dates = pd.date_range(
    "2025-09-19",
    "2026-07-31"
)

missing_dates = expected_dates.difference(arrivals_df["date"])

print("Missing dates:", len(missing_dates))

if len(missing_dates) > 0:
    print(missing_dates)

# %%
print("Missing arrival values:", arrivals_df["arrivals"].isna().sum())

# %%
print("\nFirst 10:")
print(arrivals_df.head(10).to_string(index=False))

print("\nLast 10:")
print(arrivals_df.tail(10).to_string(index=False))

# %%
print("Shape:", arrivals_df.shape)
print("Duplicate dates:", arrivals_df["date"].duplicated().sum())

expected_dates = pd.date_range("2025-09-19", "2026-07-31")
missing_dates = expected_dates.difference(arrivals_df["date"])

print("Missing dates:", len(missing_dates))
print("Missing arrival values:", arrivals_df["arrivals"].isna().sum())

# %%
print("arrivals_df date type:", arrivals_df["date"].dtype)
print("main_df date type:", main_df["date"].dtype)

# %%
main_df["date"] = pd.to_datetime(
    main_df["date"],
    format="mixed",
    dayfirst=False
)

arrivals_df["date"] = pd.to_datetime(
    arrivals_df["date"],
    format="mixed"
)

print("main_df date type:", main_df["date"].dtype)
print("arrivals_df date type:", arrivals_df["date"].dtype)

# %%
print("Main first date:", main_df["date"].min())
print("Main last date:", main_df["date"].max())

print("Arrivals first date:", arrivals_df["date"].min())
print("Arrivals last date:", arrivals_df["date"].max())

# %%
matching_dates = main_df["date"].isin(arrivals_df["date"]).sum()

print("Matching arrival dates:", matching_dates)

# %%
# Make a copy so the current main_df remains untouched
main_with_arrivals = main_df.copy()

# Merge daily tourist arrivals
main_with_arrivals = main_with_arrivals.merge(
    arrivals_df,
    on="date",
    how="left"
)

print("Shape:", main_with_arrivals.shape)
print("Columns:", main_with_arrivals.columns.tolist())

# %%
print(main_with_arrivals.columns.tolist())

# %%
print("Columns containing 'arrival':")

for col in main_with_arrivals.columns:
    if "arrival" in col.lower():
        print(repr(col))

# %%
print("Original main_df arrival columns:")

for col in main_df.columns:
    if "arrival" in col.lower():
        print(repr(col))

# %%
print(
    main_df[
        main_df["date"].between("2025-09-15", "2025-09-23")
    ][["date", "arrivals"]].to_string(index=False)
)

# %%
print(
    main_df[
        main_df["date"].between("2026-07-27", "2026-07-31")
    ][["date", "arrivals"]].to_string(index=False)
)

# %%
main_final = main_df.copy()

# %%
arrival_lookup = arrivals_df.set_index("date")["arrivals"]

# %%
main_final["arrivals"] = main_final["arrivals"].fillna(
    main_final["date"].map(arrival_lookup)
)

# %%
print(
    main_final[
        main_final["date"].between(
            "2025-09-15",
            "2025-09-23"
        )
    ][["date", "arrivals"]].to_string(index=False)
)

# %%
print(
    main_final[
        main_final["date"].between(
            "2026-07-27",
            "2026-07-31"
        )
    ][["date", "arrivals"]].to_string(index=False)
)

# %%
print("Missing arrivals:", main_final["arrivals"].isna().sum())
print("Shape:", main_final.shape)

# %%
print("Date range:", main_final["date"].min(), "to", main_final["date"].max())
print("\nMissing values by column:")
print(main_final.isna().sum())

# %%
print("Rows with missing weather:")
print(
    main_final[
        main_final[
            ["temperature", "precipitation", "humidity", "wind_speed"]
        ].isna().any(axis=1)
    ][
        ["date", "temperature", "precipitation", "humidity", "wind_speed"]
    ].to_string(index=False)
)

# %%
import requests
import pandas as pd

weather_url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 6.9271,
    "longitude": 79.8612,
    "start_date": "2025-09-07",
    "end_date": "2025-09-18",
    "daily": [
        "temperature_2m_mean",
        "precipitation_sum",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max"
    ],
    "timezone": "Asia/Colombo"
}

response = requests.get(weather_url, params=params)
response.raise_for_status()

weather_missing = pd.DataFrame({
    "date": pd.to_datetime(response.json()["daily"]["time"]),
    "temperature": response.json()["daily"]["temperature_2m_mean"],
    "precipitation": response.json()["daily"]["precipitation_sum"],
    "humidity": response.json()["daily"]["relative_humidity_2m_mean"],
    "wind_speed": response.json()["daily"]["wind_speed_10m_max"]
})

print(weather_missing)

# %%
weather_lookup = weather_missing.set_index("date")

weather_cols = [
    "temperature",
    "precipitation",
    "humidity",
    "wind_speed"
]

for col in weather_cols:
    main_final[col] = main_final[col].fillna(
        main_final["date"].map(weather_lookup[col])
    )

print(
    main_final[
        main_final["date"].between("2025-09-07", "2025-09-18")
    ][["date"] + weather_cols].to_string(index=False)
)

# %%
print("Missing values by column:")
print(main_final.isna().sum())

print("\nShape:", main_final.shape)
print("Date range:", main_final["date"].min(), "to", main_final["date"].max())

# %%
output_path = "/content/0. Research Data - exchange rates updated_final.csv"

main_final.to_csv(output_path, index=False)

print(f"Saved successfully: {output_path}")

# %%
# Reload the saved CSV
check_df = pd.read_csv(output_path)

print("Shape:", check_df.shape)
print("Date range:", check_df["date"].min(), "to", check_df["date"].max())
print("Missing arrivals:", check_df["arrivals"].isna().sum())

print("\nMissing weather values:")
print(
    check_df[
        ["temperature", "precipitation", "humidity", "wind_speed"]
    ].isna().sum()
)

# %%
from google.colab import files

files.download("/content/0. Research Data - exchange rates updated_final.csv")

# %%
# Replace all remaining missing values with the text "null"
main_final = main_final.fillna("null")

# Save the updated file
output_path = "/content/0. Research Data - exchange rates updated_final.csv"
main_final.to_csv(output_path, index=False)

print("Saved successfully:", output_path)

# %%
from google.colab import files

files.download("/content/0. Research Data - exchange rates updated_final.csv")


