import pandas as pd


def add_lag_features(df, target_col="aqi", lags=(1, 3, 7)):
    """
    Adds aqi_lag_1, aqi_lag_3, aqi_lag_7 -- the AQI value N days before
    the current row, per city. Operates on daily-resolution data sorted
    by city + date (as backfill.py's merged dataset is).
    """
    df = df.sort_values(["city", "date"])
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby("city")[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col="aqi", windows=(3, 7)):
    """
    Adds rolling mean + std over the past N days, per city. Uses
    .shift(1) before rolling so the current day's own value never leaks
    into its own rolling statistic (that would be a lookahead bug).
    """
    df = df.sort_values(["city", "date"])
    for window in windows:
        shifted = df.groupby("city")[target_col].shift(1)
        df[f"{target_col}_rolling_mean_{window}"] = (
            shifted.groupby(df["city"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            shifted.groupby(df["city"]).rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        )
    return df


def add_calendar_features(df):
    """Adds is_weekend and season (meteorological, adjusted for Pakistan's
    climate: Dec-Feb winter, Mar-May spring, Jun-Sep monsoon/summer,
    Oct-Nov autumn -- roughly maps to how AQI actually varies seasonally
    here, unlike a generic 4-season split)."""
    df = df.copy()
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    def _season(month):
        if month in (12, 1, 2):
            return 0  # winter -- typically worst AQI (smog season)
        if month in (3, 4, 5):
            return 1  # spring
        if month in (6, 7, 8, 9):
            return 2  # monsoon/summer
        return 3  # autumn

    df["season"] = df["month"].apply(_season)
    return df
