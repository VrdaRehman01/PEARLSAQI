import pandas as pd

from src.database.connection import get_connection


def build_features():

    conn = get_connection()

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------
    # Load AQI + Weather
    # --------------------------------------------------

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

    print(f"Initial rows: {len(df)}")

    if df.empty:
        raise RuntimeError(
            "No AQI + weather data found."
        )

    # --------------------------------------------------
    # Make sure date is datetime
    # --------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------
    # Lag features
    # --------------------------------------------------

    df["aqi_lag_1"] = (
        df.groupby("city_id")["aqi"]
        .shift(1)
    )

    df["aqi_lag_2"] = (
        df.groupby("city_id")["aqi"]
        .shift(2)
    )

    df["aqi_lag_3"] = (
        df.groupby("city_id")["aqi"]
        .shift(3)
    )

    df["aqi_lag_7"] = (
        df.groupby("city_id")["aqi"]
        .shift(7)
    )

    # --------------------------------------------------
    # Rolling features
    # --------------------------------------------------

    df["aqi_rolling_3"] = (
        df.groupby("city_id")["aqi"]
        .transform(
            lambda x:
            x.shift(1).rolling(3).mean()
        )
    )

    df["aqi_rolling_7"] = (
        df.groupby("city_id")["aqi"]
        .transform(
            lambda x:
            x.shift(1).rolling(7).mean()
        )
    )

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Season
    # --------------------------------------------------

    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        if month in [3, 4, 5]:
            return "Spring"

        if month in [6, 7, 8]:
            return "Summer"

        return "Autumn"

    df["season"] = df["month"].apply(get_season)

    # --------------------------------------------------
    # Target
    # Predict next day's AQI
    # --------------------------------------------------

    df["target_aqi"] = (
        df.groupby("city_id")["aqi"]
        .shift(-1)
    )

    # --------------------------------------------------
    # Remove rows without enough history/target
    # --------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    print(f"Final training rows: {len(df)}")

    print(
        f"Features: {len(df.columns)}"
    )

    # --------------------------------------------------
    # Save features to DuckDB
    # --------------------------------------------------

    conn.execute("""
        DROP TABLE IF EXISTS features
    """)

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

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    print()
    print("Dataset preview:")
    print(
        df.head(10).to_string(index=False)
    )

    # --------------------------------------------------
    # Columns
    # --------------------------------------------------

    print()
    print("Columns:")
    print(
        df.columns.tolist()
    )

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print()
    print("Missing values:")
    print(
        df.isnull().sum()
    )

    # --------------------------------------------------
    # Database verification
    # --------------------------------------------------

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