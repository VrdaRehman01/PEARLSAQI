import pandas as pd

# Also referenced by download_weather_history.py but never provided.
# Same caveat as validate_historical_data.py: verify POLLUTANT_CANDIDATES
# against your actual downloaded CSV columns (print df.columns after
# load_historical_data() to check) before trusting this blindly.

POLLUTANT_CANDIDATES = ["aqi", "AQI", "pm25", "PM2.5", "pm2_5"]


def create_daily_aqi(df):
    """
    Collapse historical pollution readings (which may be hourly/sub-daily)
    into one row per city per day, since the free historical weather API
    (Open-Meteo) only provides daily granularity -- so both sides of the
    eventual merge need to be at the same time resolution.
    """
    value_col = next((c for c in POLLUTANT_CANDIDATES if c in df.columns), None)

    if value_col is None:
        raise ValueError(
            f"None of {POLLUTANT_CANDIDATES} found in columns: {list(df.columns)}. "
            "Update POLLUTANT_CANDIDATES in merge_historical_data.py to match "
            "your actual dataset's pollutant column name."
        )

    daily = (
        df.groupby(["city", df["date"].dt.date])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "aqi"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def merge_aqi_and_weather(aqi_daily_df, weather_df):
    """
    Merge daily AQI with historical daily weather on city + date.
    Open-Meteo returns a 'time' column for its daily series --
    normalise that to 'date' before joining.
    """
    weather_df = weather_df.copy()
    date_col = "time" if "time" in weather_df.columns else "date"
    weather_df = weather_df.rename(columns={date_col: "date"})
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    merged = pd.merge(
        aqi_daily_df,
        weather_df,
        on=["city", "date"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "Merge produced 0 rows -- check that city names match exactly "
            "between the AQI dataset and CITY_COORDINATES (case-sensitive), "
            "and that the date ranges actually overlap."
        )

    print(f"Merged dataset: {len(merged)} rows, {merged['city'].nunique()} cities")
    return merged
