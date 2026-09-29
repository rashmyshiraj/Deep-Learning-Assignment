# Dataset Access and Provenance

## Dataset Files Included in the Repository

### 1. Original/Base Dataset

File: `0.  Research Data.csv`

GitHub:
https://github.com/rashmyshiraj/Deep-Learning-Assignment/blob/main/0.%20%20Research%20Data.csv

This is the initial dataset obtained from a senior student. The base dataset was not created from scratch by the project group.

It originally covered daily data from 01 January 2010 to 18 September 2025.

### 2. Final Encoded Dataset

File: `Sri_Lanka_Tourism_Arrivals_Encoded.csv`

GitHub:
https://github.com/rashmyshiraj/Deep-Learning-Assignment/blob/main/Sri_Lanka_Tourism_Arrivals_Encoded.csv

This is the final cleaned, feature-engineered and one-hot encoded dataset used as the common modelling dataset.

It contains:

- 6,053 daily observations
- 110 columns
- Date range: 04 January 2010 to 31 July 2026

The reduction from 6,056 raw rows to 6,053 modelling rows occurs because the first three dates did not have complete historical exchange-rate information after past-only forward filling.

## Dataset Extension

The project group extended the base dataset from 19 September 2025 to 31 July 2026.

After extension, the complete raw daily range was 01 January 2010 to 31 July 2026 with 6,056 daily observations.

Some extension values, including Google Trends values, were added manually. The exact values used are preserved in the committed project datasets.

## External Data Sources Used for Extension / Verification

### Tourist Arrivals
Sri Lanka Tourism Development Authority (SLTDA)
https://www.sltda.gov.lk/

### Weather
Open-Meteo Historical Weather API
https://open-meteo.com/

Weather variables include mean temperature, precipitation, relative humidity and wind speed for Colombo, Sri Lanka.

### GDP Per Capita and Inflation
World Bank API
https://api.worldbank.org/

Annual Sri Lankan GDP-per-capita and inflation values were associated with the corresponding daily observations.

For 2026, newly published annual World Bank values were not available when the project dataset was prepared, so the preprocessing pipeline handled the unavailable trailing values instead of treating them as observed 2026 statistics.

### Exchange Rates
Central Bank of Sri Lanka (CBSL)
https://www.cbsl.gov.lk/

Exchange-rate variables include USD/LKR, RUB/LKR, INR/LKR, GBP/LKR, EUR/LKR and CNY/LKR.

### Brent Crude Oil Price
FRED / IMF Brent crude-oil series
https://fred.stlouisfed.org/

Series identifier: `POILBREUSDM`

### Search Interest
Google Trends
https://trends.google.com/

Search interest for the term `Sri Lanka` was represented using Web Search, YouTube Search and Image Search. Part of the extension was entered manually, so the committed datasets are the authoritative copies for exact reproduction.

### Holidays and Tourism Events
Sri Lankan holiday and tourism-event calendar information was incorporated into the daily dataset using Google Calendar and project-maintained event information.

## Original 23 Columns

1. `date`
2. `arrivals`
3. `gdp_per_capita`
4. `inflation_rate`
5. `brent_crude_price`
6. `usd_lkr`
7. `rub_lkr`
8. `inr_lkr`
9. `gbp_lkr`
10. `eur_lkr`
11. `cny_lkr`
12. `web_search`
13. `youtube_search`
14. `image_search`
15. `is_holiday`
16. `holiday_name`
17. `event_name`
18. `is_tourist_event`
19. `location`
20. `temperature`
21. `precipitation`
22. `humidity`
23. `wind_speed`

## Processing Summary

The group applied a staged preprocessing workflow including date validation, chronological sorting, missing-value handling, past-only exchange-rate forward filling, missingness indicators, holiday/event cleaning, weather-related feature creation, COVID-19 and economic-crisis disruption features, holiday-distance features, rare-category grouping, calendar/cyclical features and one-hot encoding.

The final categorical encoding uses `pandas.get_dummies()` on:

- `holiday_name_grouped`
- `event_name_grouped`

## Recommended Dataset for Model Reproduction

To reproduce the deep-learning experiments directly, use:

`Sri_Lanka_Tourism_Arrivals_Encoded.csv`

This avoids rerunning external APIs and avoids differences caused by later revisions to external web data.
