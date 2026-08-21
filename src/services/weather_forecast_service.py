"""
PEARLSAQI FUTURE WEATHER FORECAST SERVICE

Production-safe future weather ingestion.

Purpose
-------
Downloads future weather from Open-Meteo and stores it separately
from historical weather.

Database rule
-------------
There must be exactly ONE canonical row for:

    city_id + date

If a newer forecast is downloaded for an existing date, the old
forecast is replaced.

Example:

Forecast generated on Aug 14:
    Aug 15
    Aug 16
    Aug 17

Forecast generated on Aug 15:
    Aug 16
    Aug 17
    Aug 18

After the second run the database contains:

    Aug 15
    Aug 16  <- updated
    Aug 17  <- updated
    Aug 18  <- new

There are never two rows for the same city/date.
"""

from datetime import date, datetime, timedelta
import time

import pandas as pd

from src.database.connection import get_connection
from src.ingestion.api_client import APIClient
from src.ingestion.city_loader import load_cities


class WeatherForecastService:

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    BASE_URL = (
        "https://api.open-meteo.com/v1/forecast"
    )

    FORECAST_DAYS = 3

    TIMEZONE = "Asia/Karachi"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self):

        self.client = APIClient(
            timeout=60,
            max_retries=5,
            retry_delay=5,
        )

    # ==========================================================
    # DATABASE TABLE
    # ==========================================================

    def ensure_table(self):

        conn = get_connection()

        try:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_forecasts (

                    id BIGINT,

                    city_id INTEGER,

                    city_name VARCHAR,

                    date DATE,

                    temperature DOUBLE,

                    humidity DOUBLE,

                    precipitation DOUBLE,

                    windspeed DOUBLE,

                    forecast_origin DATE,

                    created_at TIMESTAMP
                )
                """
            )

        finally:

            conn.close()

    # ==========================================================
    # GET ONE CITY FORECAST
    # ==========================================================

    def get_city_forecast(self, city):

        today = date.today()

        start_date = (
            today
            + timedelta(days=1)
        )

        end_date = (
            today
            + timedelta(days=self.FORECAST_DAYS)
        )

        params = {

            "latitude":
                float(city["latitude"]),

            "longitude":
                float(city["longitude"]),

            "start_date":
                start_date.isoformat(),

            "end_date":
                end_date.isoformat(),

            "daily":
                (
                    "temperature_2m_mean,"
                    "relative_humidity_2m_mean,"
                    "precipitation_sum,"
                    "wind_speed_10m_max"
                ),

            "timezone":
                self.TIMEZONE,
        }

        data = self.client.get(
            self.BASE_URL,
            params,
        )

        daily = data.get("daily")

        if not daily:

            raise RuntimeError(
                "Open-Meteo returned no daily "
                "forecast data."
            )

        dates = daily.get(
            "time",
            [],
        )

        temperatures = daily.get(
            "temperature_2m_mean",
            [],
        )

        humidities = daily.get(
            "relative_humidity_2m_mean",
            [],
        )

        precipitation = daily.get(
            "precipitation_sum",
            [],
        )

        windspeeds = daily.get(
            "wind_speed_10m_max",
            [],
        )

        records = []

        for i, current_date in enumerate(
            dates
        ):

            records.append(
                {

                    "city_id":
                        int(
                            city["city_id"]
                        ),

                    "city_name":
                        str(
                            city["city_name"]
                        ),

                    "date":
                        pd.to_datetime(
                            current_date
                        ).date(),

                    "temperature":
                        (
                            temperatures[i]
                            if i < len(temperatures)
                            else None
                        ),

                    "humidity":
                        (
                            humidities[i]
                            if i < len(humidities)
                            else None
                        ),

                    "precipitation":
                        (
                            precipitation[i]
                            if i < len(precipitation)
                            else None
                        ),

                    "windspeed":
                        (
                            windspeeds[i]
                            if i < len(windspeeds)
                            else None
                        ),

                    "forecast_origin":
                        today,
                }
            )

        return records

    # ==========================================================
    # VALIDATE ONE CITY
    # ==========================================================

    def validate_city_forecast(
        self,
        records,
    ):

        if not records:

            raise RuntimeError(
                "City forecast returned zero rows."
            )

        today = date.today()

        expected_dates = [

            today
            + timedelta(days=i)

            for i in range(
                1,
                self.FORECAST_DAYS + 1,
            )
        ]

        actual_dates = [

            record["date"]

            for record in records
        ]

        if actual_dates != expected_dates:

            raise RuntimeError(
                "Future weather dates are incorrect.\n"
                f"Expected: {expected_dates}\n"
                f"Received: {actual_dates}"
            )

        for record in records:

            required_fields = [
                "temperature",
                "humidity",
                "precipitation",
                "windspeed",
            ]

            for field in required_fields:

                value = record[field]

                if value is None:

                    raise RuntimeError(
                        f"Missing {field} for "
                        f"{record['city_name']} "
                        f"{record['date']}"
                    )

            if float(
                record["precipitation"]
            ) < 0:

                raise RuntimeError(
                    "Negative precipitation detected "
                    f"for {record['city_name']} "
                    f"{record['date']}."
                )

            if float(
                record["windspeed"]
            ) < 0:

                raise RuntimeError(
                    "Negative windspeed detected "
                    f"for {record['city_name']} "
                    f"{record['date']}."
                )

        return True

    # ==========================================================
    # GET NEXT SAFE ID
    # ==========================================================

    def get_next_id(
        self,
        conn,
    ):

        result = conn.execute(
            """
            SELECT
                COALESCE(
                    MAX(id),
                    0
                )
            FROM weather_forecasts
            """
        ).fetchone()

        return int(result[0]) + 1

    # ==========================================================
    # SAVE / UPSERT FORECAST
    # ==========================================================

    def save_forecasts(
        self,
        df,
    ):

        self.ensure_table()

        conn = get_connection()

        try:

            next_id = self.get_next_id(
                conn
            )

            for _, row in df.iterrows():

                city_id = int(
                    row["city_id"]
                )

                forecast_date = row[
                    "date"
                ]

                # ------------------------------------------------
                # Remove previous forecast for this exact
                # city/date.
                #
                # This is the key protection against duplicates.
                # ------------------------------------------------

                conn.execute(
                    """
                    DELETE FROM weather_forecasts

                    WHERE
                        city_id = ?
                        AND date = ?
                    """,
                    [
                        city_id,
                        forecast_date,
                    ],
                )

                # ------------------------------------------------
                # Insert the latest forecast.
                # ------------------------------------------------

                conn.execute(
                    """
                    INSERT INTO weather_forecasts (

                        id,
                        city_id,
                        city_name,
                        date,
                        temperature,
                        humidity,
                        precipitation,
                        windspeed,
                        forecast_origin,
                        created_at

                    )

                    VALUES (

                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?
                    )
                    """,
                    [

                        next_id,

                        city_id,

                        str(
                            row["city_name"]
                        ),

                        forecast_date,

                        float(
                            row["temperature"]
                        ),

                        float(
                            row["humidity"]
                        ),

                        float(
                            row["precipitation"]
                        ),

                        float(
                            row["windspeed"]
                        ),

                        row[
                            "forecast_origin"
                        ],

                        datetime.now(),
                    ],
                )

                next_id += 1

        finally:

            conn.close()

    # ==========================================================
    # REMOVE OLD DUPLICATES
    # ==========================================================

    def cleanup_duplicates(self):

        conn = get_connection()

        try:

            conn.execute(
                """
                DELETE FROM weather_forecasts

                WHERE id IN (

                    SELECT id

                    FROM (

                        SELECT

                            id,

                            ROW_NUMBER() OVER (

                                PARTITION BY
                                    city_id,
                                    date

                                ORDER BY
                                    forecast_origin DESC,
                                    created_at DESC,
                                    id DESC

                            ) AS rn

                        FROM weather_forecasts

                    )

                    WHERE rn > 1
                )
                """
            )

        finally:

            conn.close()

    # ==========================================================
    # VERIFY DATABASE
    # ==========================================================

    def verify_database(
        self,
        expected_cities,
    ):

        conn = get_connection()

        try:

            summary = conn.execute(
                """
                SELECT

                    COUNT(*) AS rows,

                    COUNT(
                        DISTINCT city_id
                    ) AS cities,

                    COUNT(
                        DISTINCT date
                    ) AS dates,

                    MIN(date) AS first_date,

                    MAX(date) AS last_date

                FROM weather_forecasts
                """
            ).fetchdf()

            duplicates = conn.execute(
                """
                SELECT

                    city_id,

                    date,

                    COUNT(*) AS rows

                FROM weather_forecasts

                GROUP BY
                    city_id,
                    date

                HAVING COUNT(*) > 1

                ORDER BY
                    city_id,
                    date
                """
            ).fetchdf()

        finally:

            conn.close()

        print()
        print("=" * 70)
        print("WEATHER FORECAST DATABASE VERIFICATION")
        print("=" * 70)

        print(
            summary.to_string(
                index=False
            )
        )

        print()

        if duplicates.empty:

            print(
                "Duplicate city/date rows : NONE"
            )

        else:

            print(
                "Duplicate city/date rows : "
                f"{len(duplicates)}"
            )

            print(
                duplicates.to_string(
                    index=False
                )
            )

            raise RuntimeError(
                "Weather forecast database "
                "still contains duplicate "
                "city/date rows."
            )

        actual_cities = int(
            summary.iloc[0]["cities"]
        )

        if actual_cities < expected_cities:

            raise RuntimeError(
                "Weather forecast database "
                "is missing cities. "
                f"Expected {expected_cities}, "
                f"found {actual_cities}."
            )

        return summary

    # ==========================================================
    # DOWNLOAD ALL CITIES
    # ==========================================================

    def download_forecast(self):

        print()
        print("=" * 70)
        print("PEARLSAQI FUTURE WEATHER FORECAST")
        print("=" * 70)

        today = date.today()

        start_date = (
            today
            + timedelta(days=1)
        )

        end_date = (
            today
            + timedelta(
                days=self.FORECAST_DAYS
            )
        )

        print()
        print(
            f"Forecast origin : {today}"
        )

        print(
            f"Forecast dates  : "
            f"{start_date} -> {end_date}"
        )

        cities = load_cities()

        expected_cities = len(cities)

        all_records = []

        # ======================================================
        # DOWNLOAD EACH CITY
        # ======================================================

        for _, city in cities.iterrows():

            city_name = str(
                city["city_name"]
            )

            print()
            print("-" * 70)
            print(city_name)

            try:

                records = (
                    self.get_city_forecast(
                        city
                    )
                )

                self.validate_city_forecast(
                    records
                )

                all_records.extend(
                    records
                )

                print(
                    f"Downloaded "
                    f"{len(records)} "
                    f"forecast days."
                )

            except Exception as error:

                print(
                    "WARNING: Could not download "
                    f"forecast weather for "
                    f"{city_name}"
                )

                print(
                    f"Reason: {error}"
                )

            time.sleep(1)

        # ======================================================
        # GLOBAL VALIDATION
        # ======================================================

        if not all_records:

            raise RuntimeError(
                "No future weather forecasts "
                "were downloaded."
            )

        df = pd.DataFrame(
            all_records
        )

        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.date

        df["forecast_origin"] = (
            pd.to_datetime(
                df["forecast_origin"]
            ).dt.date
        )

        # ------------------------------------------------------
        # Remove duplicates inside this API download.
        # ------------------------------------------------------

        df = (
            df
            .sort_values(
                [
                    "city_id",
                    "date",
                ]
            )
            .drop_duplicates(
                subset=[
                    "city_id",
                    "date",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        expected_rows = (
            expected_cities
            * self.FORECAST_DAYS
        )

        if len(df) != expected_rows:

            raise RuntimeError(
                "Future weather download "
                "row count mismatch.\n"
                f"Expected: {expected_rows}\n"
                f"Received: {len(df)}"
            )

        # ======================================================
        # SAVE
        # ======================================================

        self.save_forecasts(
            df
        )

        # ======================================================
        # CLEAN EXISTING DUPLICATES
        # ======================================================

        self.cleanup_duplicates()

        # ======================================================
        # VERIFY
        # ======================================================

        summary = self.verify_database(
            expected_cities
        )

        # ======================================================
        # OUTPUT
        # ======================================================

        print()
        print("=" * 70)
        print(
            "FUTURE WEATHER FORECAST COMPLETE"
        )
        print("=" * 70)

        print(
            f"Downloaded rows : {len(df)}"
        )

        print(
            f"Database rows   : "
            f"{int(summary.iloc[0]['rows'])}"
        )

        print(
            f"Cities          : "
            f"{int(summary.iloc[0]['cities'])}"
        )

        print(
            f"Unique dates    : "
            f"{int(summary.iloc[0]['dates'])}"
        )

        print(
            f"First date      : "
            f"{summary.iloc[0]['first_date']}"
        )

        print(
            f"Last date       : "
            f"{summary.iloc[0]['last_date']}"
        )

        print()
        print(
            "Downloaded forecast rows:"
        )

        print(
            df.to_string(
                index=False
            )
        )

        return df


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    service = WeatherForecastService()

    service.download_forecast()