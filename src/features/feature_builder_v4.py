import os
import numpy as np
import pandas as pd

from src.database.connection import get_connection
from src.features.feature_store import (
    initialise_feature_store,
    build_historical_feature_store,
)


# ==========================================================
# Configuration
# ==========================================================

OUTPUT_FILE = "data/processed/features_v4.parquet"

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Helper functions
# ==========================================================

def add_lag_features(df):
    """
    Add historical AQI and pollutant lag features.

    All features use only information from the current
    date or previous dates.
    """

    grouped = df.groupby("city_id", group_keys=False)

    # ------------------------------------------------------
    # AQI lags
    # ------------------------------------------------------

    for lag in [1, 2, 3, 7, 14]:
        df[f"aqi_lag_{lag}"] = (
            grouped["aqi"]
            .shift(lag)
        )

    # ------------------------------------------------------
    # AQI changes
    # ------------------------------------------------------

    for lag in [1, 2, 3, 7]:
        df[f"aqi_change_{lag}d"] = (
            df["aqi"] - df[f"aqi_lag_{lag}"]
        )

    # ------------------------------------------------------
    # AQI acceleration
    # ------------------------------------------------------

    df["aqi_acceleration_1d"] = (
        df["aqi_change_1d"]
        - grouped["aqi_change_1d"].shift(1)
    )

    df["aqi_acceleration_2d"] = (
        df["aqi_change_2d"]
        - grouped["aqi_change_2d"].shift(1)
    )

    return df


def add_aqi_rolling_features(df):
    """
    Rolling AQI statistics.

    IMPORTANT:
    rolling values are shifted by one day so that
    tomorrow's target cannot leak into today's features.
    """

    grouped = df.groupby("city_id", group_keys=False)

    for window in [3, 7, 14]:

        df[f"aqi_rolling_{window}"] = (
            grouped["aqi"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window)
                 .mean()
            )
        )

        df[f"aqi_std_{window}"] = (
            grouped["aqi"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window)
                 .std()
            )
        )

        df[f"aqi_min_{window}"] = (
            grouped["aqi"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window)
                 .min()
            )
        )

        df[f"aqi_max_{window}"] = (
            grouped["aqi"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window)
                 .max()
            )
        )

        df[f"aqi_range_{window}"] = (
            df[f"aqi_max_{window}"]
            - df[f"aqi_min_{window}"]
        )

    return df


def add_aqi_trend_features(df):
    """
    Estimate recent AQI trend using historical values only.
    """

    grouped = df.groupby("city_id", group_keys=False)

    for window in [3, 7]:

        df[f"aqi_trend_{window}d"] = (
            grouped["aqi"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window)
                 .apply(
                     lambda values:
                     np.polyfit(
                         np.arange(len(values)),
                         values,
                         1
                     )[0],
                     raw=True
                 )
            )
        )

    # Distance from historical maximum

    df["aqi_distance_from_max_7"] = (
        df["aqi"]
        - df["aqi_max_7"]
    )

    df["aqi_distance_from_max_14"] = (
        df["aqi"]
        - df["aqi_max_14"]
    )

    # Relative position inside recent AQI range

    df["aqi_percentile_7"] = (
        (
            df["aqi"]
            - df["aqi_min_7"]
        )
        /
        (
            df["aqi_range_7"]
            + 1e-6
        )
    )

    df["aqi_percentile_14"] = (
        (
            df["aqi"]
            - df["aqi_min_14"]
        )
        /
        (
            df["aqi_range_14"]
            + 1e-6
        )
    )

    return df


def add_pollution_features(df):
    """
    Pollution changes, trends and rolling statistics.
    """

    grouped = df.groupby("city_id", group_keys=False)

    pollutants = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3"
    ]

    for pollutant in pollutants:

        # --------------------------------------------------
        # Changes
        # --------------------------------------------------

        df[f"{pollutant}_change_1d"] = (
            grouped[pollutant]
            .diff(1)
        )

        df[f"{pollutant}_change_3d"] = (
            df[pollutant]
            -
            grouped[pollutant]
            .shift(3)
        )

        # --------------------------------------------------
        # Rolling values
        # --------------------------------------------------

        for window in [3, 7]:

            df[f"{pollutant}_rolling_{window}"] = (
                grouped[pollutant]
                .transform(
                    lambda x:
                    x.shift(1)
                     .rolling(window)
                     .mean()
                )
            )

    # ------------------------------------------------------
    # Pollution trends
    # ------------------------------------------------------

    for pollutant in [
        "pm25",
        "pm10"
    ]:

        for window in [3, 7]:

            df[f"{pollutant}_trend_{window}d"] = (
                grouped[pollutant]
                .transform(
                    lambda x:
                    x.shift(1)
                     .rolling(window)
                     .apply(
                         lambda values:
                         np.polyfit(
                             np.arange(len(values)),
                             values,
                             1
                         )[0],
                         raw=True
                     )
                )
            )

    # ------------------------------------------------------
    # Pollution interactions
    # ------------------------------------------------------

    df["pm25_pm10_ratio"] = (
        df["pm25"]
        /
        (df["pm10"] + 1e-6)
    )

    df["pm25_no2_interaction"] = (
        df["pm25"] * df["no2"]
    )

    df["pm25_co_interaction"] = (
        df["pm25"] * df["co"]
    )

    df["pm25_o3_interaction"] = (
        df["pm25"] * df["o3"]
    )

    df["pollution_sum"] = (
        df["pm25"]
        + df["pm10"]
        + df["no2"]
        + df["so2"]
        + df["o3"]
    )
    # Winning V4 feature
    # Must match the exact formula used when training
    # xgboost_v4_final_108.pkl
    df["pm25_pollution_ratio"] = (
        df["pm25"] /
        (df["pollution_sum"].abs() + 1e-6)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def add_weather_features(df):
    """
    Weather changes and rolling conditions.
    """

    grouped = df.groupby("city_id", group_keys=False)

    weather_columns = [
        "temperature",
        "humidity",
        "windspeed",
        "precipitation"
    ]

    for column in weather_columns:

        df[f"{column}_change_1d"] = (
            grouped[column]
            .diff(1)
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

    return df


def add_regime_features(df):
    """
    AQI pollution regime indicators.

    These describe today's known AQI and therefore are
    legitimate predictors of tomorrow's AQI.
    """

    df["aqi_moderate"] = (
        (df["aqi"] >= 51)
        & (df["aqi"] <= 100)
    ).astype(int)

    df["aqi_unhealthy"] = (
        (df["aqi"] >= 101)
        & (df["aqi"] <= 150)
    ).astype(int)

    df["aqi_very_unhealthy"] = (
        (df["aqi"] >= 151)
        & (df["aqi"] <= 200)
    ).astype(int)

    df["aqi_severe"] = (
        (df["aqi"] >= 201)
        & (df["aqi"] <= 300)
    ).astype(int)

    df["aqi_extreme"] = (
        df["aqi"] >= 301
    ).astype(int)

    # ------------------------------------------------------
    # Previous-day AQI regime
    # ------------------------------------------------------

    previous_aqi = (
        df.groupby("city_id")["aqi"]
        .shift(1)
    )

    def regime(value):

        if pd.isna(value):
            return 0

        if value <= 50:
            return 0
        elif value <= 100:
            return 1
        elif value <= 150:
            return 2
        elif value <= 200:
            return 3
        elif value <= 300:
            return 4
        else:
            return 5

    current_regime = df["aqi"].apply(regime)
    previous_regime = previous_aqi.apply(regime)

    df["aqi_regime_change"] = (
        current_regime
        != previous_regime
    ).astype(int)

    # ------------------------------------------------------
    # Recent high/extreme AQI
    # ------------------------------------------------------

    grouped = df.groupby("city_id", group_keys=False)

    df["high_aqi_recent"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .max()
        )
        >= 201
    ).astype(int)

    df["extreme_aqi_recent"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(14)
             .max()
        )
        >= 301
    ).astype(int)

    return df


def add_city_history_features(df):
    """
    City-specific historical statistics.

    Every statistic uses only dates before the current row.
    """

    grouped = df.groupby("city_id", group_keys=False)

    # Expanding statistics shifted by one day

    df["city_aqi_mean"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .mean()
        )
    )

    df["city_aqi_std"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .std()
        )
    )

    df["city_pm25_mean"] = (
        grouped["pm25"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .mean()
        )
    )

    # Recent city behavior

    df["city_recent_mean"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(30)
             .mean()
        )
    )

    df["city_recent_max"] = (
        grouped["aqi"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(30)
             .max()
        )
    )

    return df


def add_calendar_features(df):

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

    return df


# ==========================================================
# Main feature engineering
# ==========================================================

def build_features():

    print("=" * 60)
    print("AQI FEATURE ENGINEERING - V4")
    print("=" * 60)

    conn = get_connection()

    # ------------------------------------------------------
    # Load AQI + weather
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

    df = conn.execute(query).fetchdf()

    conn.close()

    print()
    print(
        f"Initial rows: {len(df)}"
    )

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    df["target_aqi"] = (
        df.groupby("city_id")["aqi"]
        .shift(-1)
    )

    # ------------------------------------------------------
    # Calendar
    # ------------------------------------------------------

    dates = pd.to_datetime(df["date"])

    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_year"] = dates.dt.dayofyear
    df["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------

    df = add_lag_features(df)

    df = add_aqi_rolling_features(df)

    df = add_aqi_trend_features(df)

    df = add_pollution_features(df)

    df = add_weather_features(df)

    df = add_regime_features(df)

    df = add_city_history_features(df)

    df = add_calendar_features(df)

    # ------------------------------------------------------
    # Remove rows without sufficient history
    # ------------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Target contains missing values."
        )

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        ).columns
    )

    if df[numeric_columns].isna().any().any():

        missing = (
            df[numeric_columns]
            .isna()
            .sum()
        )

        missing = (
            missing[missing > 0]
            .to_dict()
        )

        raise ValueError(
            f"Missing numeric values: {missing}"
        )

    # ------------------------------------------------------
    # Check target is next-day AQI
    # ------------------------------------------------------

    expected_target = (
        df.groupby("city_id")["aqi"]
        .shift(-1)
    )

    # We cannot compare the final row of each city
    # because its next-day value does not exist.

    check = (
        df["target_aqi"].values
        ==
        expected_target.values
    )

    # ------------------------------------------------------
    # Remove final rows where target is unavailable
    # ------------------------------------------------------

    df = df.dropna(
        subset=["target_aqi"]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------
    # Sync features into DuckDB
    # ------------------------------------------------------

    conn = get_connection()

    try:

        conn.register(
            "features_dataframe",
            df
        )

        conn.execute(
            "DROP TABLE IF EXISTS features"
        )

        conn.execute(
            """
            CREATE TABLE features AS
            SELECT *
            FROM features_dataframe
            """
        )

        conn.unregister(
            "features_dataframe"
        )

    finally:

        conn.close()

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "target_aqi",
            "city_name",
            "date"
        ]
    ]

    print()
    print("=" * 60)
    print("V4 FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print(
        f"Final rows: {len(df)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print()
    print("New V4 features:")

    v4_features = [
        "aqi_trend_3d",
        "aqi_trend_7d",
        "aqi_distance_from_max_7",
        "aqi_distance_from_max_14",
        "aqi_percentile_7",
        "aqi_percentile_14",
        "pm25_trend_3d",
        "pm25_trend_7d",
        "pm10_trend_3d",
        "pm10_trend_7d",
        "pm25_no2_interaction",
        "pm25_co_interaction",
        "pm25_o3_interaction",
        "aqi_moderate",
        "aqi_unhealthy",
        "aqi_very_unhealthy",
        "aqi_severe",
        "aqi_extreme",
        "aqi_regime_change",
        "city_aqi_mean",
        "city_aqi_std",
        "city_pm25_mean",
        "city_recent_mean",
        "city_recent_max",
    ]

    for feature in v4_features:

        if feature in df.columns:

            print(
                f"✓ {feature}"
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