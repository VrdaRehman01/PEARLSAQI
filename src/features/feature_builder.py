import pandas as pd

from src.database.connection import get_connection


def build_features():

    conn = get_connection()

    print("=" * 60)
    print("FEATURE ENGINEERING V2")
    print("=" * 60)

    # ======================================================
    # LOAD AQI + WEATHER
    # ======================================================

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

    INNER JOIN weather w
        ON a.city_id = w.city_id
        AND a.date = w.date

    INNER JOIN cities c
        ON a.city_id = c.city_id

    ORDER BY
        a.city_id,
        a.date
    """

    df = conn.execute(query).fetchdf()

    print()
    print(f"Initial rows: {len(df)}")

    if df.empty:
        raise RuntimeError(
            "No AQI + weather data found."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # ======================================================
    # GROUP BY CITY
    # ======================================================

    grouped = df.groupby(
        "city_id",
        group_keys=False
    )

    # ======================================================
    # AQI LAG FEATURES
    # ======================================================

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

    # ======================================================
    # AQI TREND FEATURES
    # ======================================================

    df["aqi_change_1d"] = (
        df["aqi"]
        - df["aqi_lag_1"]
    )

    df["aqi_change_2d"] = (
        df["aqi"]
        - df["aqi_lag_2"]
    )

    df["aqi_change_3d"] = (
        df["aqi"]
        - df["aqi_lag_3"]
    )

    df["aqi_change_7d"] = (
        df["aqi"]
        - df["aqi_lag_7"]
    )

    # ======================================================
    # AQI ROLLING FEATURES
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

    # ======================================================
    # POLLUTANT CHANGE FEATURES
    # ======================================================

    pollutant_columns = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3"
    ]

    for column in pollutant_columns:

        df[f"{column}_change_1d"] = (
            grouped[column]
            .diff(1)
        )

        df[f"{column}_change_3d"] = (
            df[column]
            - grouped[column].shift(3)
        )

    # ======================================================
    # POLLUTANT ROLLING FEATURES
    # ======================================================

    rolling_pollutants = [
        "pm25",
        "pm10",
        "no2",
        "o3"
    ]

    for column in rolling_pollutants:

        df[f"{column}_rolling_3"] = (
            grouped[column]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(3)
                .mean()
            )
        )

        df[f"{column}_rolling_7"] = (
            grouped[column]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(7)
                .mean()
            )
        )

    # ======================================================
    # WEATHER CHANGE FEATURES
    # ======================================================

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

    # ======================================================
    # WEATHER ROLLING FEATURES
    # ======================================================

    df["precipitation_rolling_3"] = (
        grouped["precipitation"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
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
    # CALENDAR FEATURES
    # ======================================================

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
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ======================================================
    # SEASON
    # ======================================================

    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        if month in [3, 4, 5]:
            return "Spring"

        if month in [6, 7, 8]:
            return "Summer"

        return "Autumn"

    df["season"] = (
        df["month"].apply(get_season)
    )

    # ======================================================
    # TARGET
    # ======================================================

    df["target_aqi"] = (
        grouped["aqi"].shift(-1)
    )

    # ======================================================
    # REMOVE INVALID ROWS
    # ======================================================

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    print()
    print(
        f"Final training rows: {len(df)}"
    )

    print(
        f"Features: {len(df.columns)}"
    )

    # ======================================================
    # SAVE TO DUCKDB
    # ======================================================

    conn.execute(
        "DROP TABLE IF EXISTS features"
    )

    conn.register(
        "features_dataframe",
        df
    )

    conn.execute("""
        CREATE TABLE features AS
        SELECT *
        FROM features_dataframe
    """)

    conn.unregister(
        "features_dataframe"
    )

    # ======================================================
    # PREVIEW
    # ======================================================

    print()
    print("Dataset preview:")

    print(
        df.head(10).to_string(
            index=False
        )
    )

    print()
    print("Columns:")

    print(
        df.columns.tolist()
    )

    # ======================================================
    # MISSING VALUES
    # ======================================================

    print()
    print("Missing values:")

    print(
        df.isnull().sum()
    )

    # ======================================================
    # VERIFY DATABASE
    # ======================================================

    count = conn.execute("""
        SELECT COUNT(*)
        FROM features
    """).fetchone()[0]

    print()
    print("=" * 60)
    print("FEATURE TABLE SAVED")
    print("=" * 60)

    print(
        f"Rows in DuckDB: {count}"
    )

    conn.close()


if __name__ == "__main__":

    build_features()