import os
import numpy as np
import pandas as pd

from src.database.connection import get_connection


# ==========================================================
# Configuration
# ==========================================================

OUTPUT_FILE = "data/processed/features_v3.parquet"

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Helper
# ==========================================================

def add_group_features(df):

    # Always sort by city and date first.
    df = df.sort_values(
        ["city_id", "date"]
    ).copy()

    grouped = df.groupby(
        "city_id",
        group_keys=False
    )

    # ======================================================
    # AQI MOMENTUM
    # ======================================================

    df["aqi_change_1d"] = (
        grouped["aqi"].diff(1)
    )

    df["aqi_change_2d"] = (
        grouped["aqi"].diff(2)
    )

    df["aqi_change_3d"] = (
        grouped["aqi"].diff(3)
    )

    df["aqi_change_7d"] = (
        grouped["aqi"].diff(7)
    )

    # ======================================================
    # AQI ACCELERATION
    # ======================================================

    df["aqi_acceleration_1d"] = (
        df["aqi_change_1d"]
        - grouped["aqi_change_1d"].shift(1)
    )

    df["aqi_acceleration_2d"] = (
        df["aqi_change_2d"]
        - grouped["aqi_change_2d"].shift(1)
    )

    # ======================================================
    # AQI ROLLING STATISTICS
    # ======================================================

    df["aqi_rolling_3"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    df["aqi_rolling_7"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .mean()
        )
    )

    df["aqi_rolling_14"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .mean()
        )
    )

    # ======================================================
    # AQI VOLATILITY
    # ======================================================

    df["aqi_std_3"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .std()
        )
    )

    df["aqi_std_7"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .std()
        )
    )

    df["aqi_std_14"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .std()
        )
    )

    # ======================================================
    # AQI RANGE
    # ======================================================

    df["aqi_min_7"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .min()
        )
    )

    df["aqi_max_7"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .max()
        )
    )

    df["aqi_range_7"] = (
        df["aqi_max_7"]
        - df["aqi_min_7"]
    )

    df["aqi_min_14"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .min()
        )
    )

    df["aqi_max_14"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .max()
        )
    )

    df["aqi_range_14"] = (
        df["aqi_max_14"]
        - df["aqi_min_14"]
    )

    # ======================================================
    # POLLUTANT MOMENTUM
    # ======================================================

    pollutants = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3"
    ]

    for pollutant in pollutants:

        df[f"{pollutant}_change_1d"] = (
            grouped[pollutant].diff(1)
        )

        df[f"{pollutant}_change_3d"] = (
            grouped[pollutant].diff(3)
        )

        df[f"{pollutant}_acceleration"] = (
            df[f"{pollutant}_change_1d"]
            - grouped[
                f"{pollutant}_change_1d"
            ].shift(1)
        )

    # ======================================================
    # POLLUTANT ROLLING MEANS
    # ======================================================

    rolling_pollutants = [
        "pm25",
        "pm10",
        "no2",
        "o3"
    ]

    for pollutant in rolling_pollutants:

        df[f"{pollutant}_rolling_3"] = (
            grouped[pollutant]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(3)
                .mean()
            )
        )

        df[f"{pollutant}_rolling_7"] = (
            grouped[pollutant]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(7)
                .mean()
            )
        )

    # ======================================================
    # POLLUTION PRESSURE FEATURES
    # ======================================================

    # PM2.5 contribution relative to total particulate matter.

    df["pm25_pm10_ratio"] = (
        df["pm25"]
        / (df["pm10"] + 1e-6)
    )

    # Combined pollution indicator.

    df["pollution_sum"] = (
        df["pm25"]
        + df["pm10"]
        + df["no2"]
        + df["so2"]
        + df["o3"]
    )

    # ======================================================
    # WEATHER MOMENTUM
    # ======================================================

    weather_columns = [
        "temperature",
        "humidity",
        "windspeed",
        "precipitation"
    ]

    for column in weather_columns:

        df[f"{column}_change_1d"] = (
            grouped[column].diff(1)
        )

    df["precipitation_rolling_3"] = (
        grouped["precipitation"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .sum()
        )
    )

    df["windspeed_rolling_3"] = (
        grouped["windspeed"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    # ======================================================
    # CYCLICAL SEASONAL FEATURES
    # ======================================================

    df["month_sin"] = (
        np.sin(
            2 * np.pi * df["month"] / 12
        )
    )

    df["month_cos"] = (
        np.cos(
            2 * np.pi * df["month"] / 12
        )
    )

    df["day_of_year_sin"] = (
        np.sin(
            2 * np.pi
            * df["day_of_year"]
            / 365.25
        )
    )

    df["day_of_year_cos"] = (
        np.cos(
            2 * np.pi
            * df["day_of_year"]
            / 365.25
        )
    )

    # ======================================================
    # HIGH POLLUTION INDICATORS
    # ======================================================

    df["high_aqi_recent"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .apply(
                lambda values:
                np.mean(values >= 200),
                raw=True
            )
        )
    )

    df["extreme_aqi_recent"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .apply(
                lambda values:
                np.mean(values >= 300),
                raw=True
            )
        )
    )

    return df


# ==========================================================
# Main Feature Builder
# ==========================================================

def build_features():

    print("=" * 60)
    print("AQI FEATURE ENGINEERING - V3")
    print("=" * 60)

    # ------------------------------------------------------
    # Connect to DuckDB
    # ------------------------------------------------------

    conn = get_connection()

    # ------------------------------------------------------
    # Load AQI + weather data
    # ------------------------------------------------------

    query = """
        SELECT
            a.city_id,
            c.city_name,
            a.date,

            a.aqi,
            a.pm25,
            a.pm10,
            a.no2,
            a.so2,
            a.co,
            a.o3,

            w.temperature,
            w.humidity,
            w.precipitation,
            w.windspeed

        FROM aqi a

        INNER JOIN cities c
            ON a.city_id = c.city_id

        INNER JOIN weather w
            ON a.city_id = w.city_id
            AND a.date = w.date

        ORDER BY
            a.city_id,
            a.date
    """

    df = conn.execute(
        query
    ).fetchdf()

    conn.close()

    print()
    print(
        f"Initial rows: {len(df)}"
    )

    # ------------------------------------------------------
    # Create next-day target
    # ------------------------------------------------------

    df = df.sort_values(
        ["city_id", "date"]
    ).copy()

    df[TARGET_COLUMN] = (
        df.groupby("city_id")["aqi"]
        .shift(-1)
    )

    # ------------------------------------------------------
    # Base lag features
    # ------------------------------------------------------

    grouped = df.groupby(
        "city_id",
        group_keys=False
    )

    df["aqi_lag_1"] = (
        grouped["aqi"].shift(1)
    )

    df["aqi_lag_2"] = (
        grouped["aqi"].shift(2)
    )

    df["aqi_lag_3"] = (
        grouped["aqi"].shift(3)
    )

    df["aqi_lag_7"] = (
        grouped["aqi"].shift(7)
    )

    # ------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------

    df["year"] = (
        df["date"].dt.year
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["day"] = (
        df["date"].dt.day
    )

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ------------------------------------------------------
    # Advanced V3 features
    # ------------------------------------------------------

    df = add_group_features(
        df
    )

    # ------------------------------------------------------
    # Remove rows without enough history
    # ------------------------------------------------------

    # 14 days of history are required because
    # several V3 features use 14-day windows.

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "aqi_lag_7",
            "aqi_rolling_14",
            "aqi_std_14",
            "aqi_min_14",
            "aqi_max_14"
        ]
    ).copy()

    # ------------------------------------------------------
    # Clean infinite values
    # ------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ------------------------------------------------------
    # Remove remaining missing rows
    # ------------------------------------------------------

    df = df.dropna().copy()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # ======================================================
    # Report
    # ======================================================

    print()
    print("=" * 60)
    print("V3 FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Final rows: {len(df)}"
    )

    print(
        f"Features: {len(df.columns)}"
    )

    print()
    print("New V3 features include:")

    v3_features = [
        "aqi_acceleration_1d",
        "aqi_acceleration_2d",
        "aqi_rolling_14",
        "aqi_std_14",
        "aqi_range_7",
        "aqi_range_14",
        "pm25_acceleration",
        "pm10_acceleration",
        "no2_acceleration",
        "o3_acceleration",
        "pm25_pm10_ratio",
        "pollution_sum",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "high_aqi_recent",
        "extreme_aqi_recent"
    ]

    for feature in v3_features:

        if feature in df.columns:

            print(
                f"  ✓ {feature}"
            )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    build_features()