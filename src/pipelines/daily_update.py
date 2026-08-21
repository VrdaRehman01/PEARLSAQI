"""
PEARLSAQI - DAILY PRODUCTION UPDATE PIPELINE

Purpose
-------
Incrementally update the AQI + weather history and generate
the next 24h / 48h / 72h recursive forecasts.

IMPORTANT
---------
- Historical AQI/weather data is NEVER deleted.
- Only missing dates are downloaded.
- Duplicate city/date records are prevented.
- Features are rebuilt after new actual data arrives.
- Forecasts always start from the latest actual AQI available.
- The pipeline can safely be executed every day.
"""

from datetime import date, timedelta
import time
import subprocess
import sys

import numpy as np
import pandas as pd

from src.ingestion.api_client import APIClient
from src.ingestion.city_loader import load_cities
from src.database.connection import get_connection
from src.services.weather_forecast_service import WeatherForecastService

# ============================================================
# CONFIGURATION
# ============================================================

AQI_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

CHUNK_DAYS = 30

TIMEZONE = "Asia/Karachi"

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


# ============================================================
# API CLIENT
# ============================================================

client = APIClient(
    timeout=60,
    max_retries=5,
    retry_delay=10
)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_database_connection():
    return get_connection()


def get_latest_dates():

    conn = get_database_connection()

    aqi_latest = conn.execute(
        """
        SELECT MAX(date)
        FROM aqi
        """
    ).fetchone()[0]

    weather_latest = conn.execute(
        """
        SELECT MAX(date)
        FROM weather
        """
    ).fetchone()[0]

    conn.close()

    return aqi_latest, weather_latest


def get_latest_date_for_city(
    table_name,
    city_id
):

    if table_name not in {
        "aqi",
        "weather",
    }:

        raise ValueError(
            f"Invalid table name: {table_name}"
        )

    conn = get_database_connection()

    result = conn.execute(
        f"""
        SELECT MAX(date)
        FROM {table_name}
        WHERE city_id = ?
        """,
        [int(city_id)]
    ).fetchone()[0]

    conn.close()

    return result

def get_existing_dates(table_name, city_id):

    conn = get_database_connection()

    result = conn.execute(
        f"""
        SELECT date
        FROM {table_name}
        WHERE city_id = ?
        """,
        [city_id]
    ).fetchall()

    conn.close()

    return {
        row[0]
        for row in result
    }


# ============================================================
# DATE CHUNKS
# ============================================================

def generate_date_chunks(start_date, end_date):

    current = start_date

    while current <= end_date:

        chunk_end = min(
            current + timedelta(
                days=CHUNK_DAYS - 1
            ),
            end_date
        )

        yield current, chunk_end

        current = chunk_end + timedelta(days=1)


# ============================================================
# AQI DOWNLOAD
# ============================================================

def download_aqi_for_city(
    city,
    start_date,
    end_date,
):

    if start_date > end_date:
        return []

    city_id = int(city["city_id"])
    city_name = str(city["city_name"])

    existing_dates = get_existing_dates(
        "aqi",
        city_id
    )

    records = []

    print()
    print("-" * 70)
    print(
        f"AQI UPDATE: {city_name}"
    )
    print("-" * 70)

    for chunk_start, chunk_end in generate_date_chunks(
        start_date,
        end_date
    ):

        print(
            f"Downloading AQI "
            f"{chunk_start} -> {chunk_end}"
        )

        params = {

            "latitude":
                city["latitude"],

            "longitude":
                city["longitude"],

            "start_date":
                chunk_start.isoformat(),

            "end_date":
                chunk_end.isoformat(),

            "hourly": (
                "pm2_5,"
                "pm10,"
                "carbon_monoxide,"
                "nitrogen_dioxide,"
                "sulphur_dioxide,"
                "ozone,"
                "us_aqi"
            ),

            "timezone":
                TIMEZONE
        }

        try:

            data = client.get(
                AQI_URL,
                params
            )

        except Exception as error:

            print(
                f"WARNING: AQI request failed "
                f"for {city_name}: {error}"
            )

            continue

        hourly = data.get("hourly")

        if not hourly:

            print(
                "WARNING: No AQI data returned."
            )

            continue

        times = hourly.get(
            "time",
            []
        )

        pm25 = hourly.get(
            "pm2_5",
            []
        )

        pm10 = hourly.get(
            "pm10",
            []
        )

        co = hourly.get(
            "carbon_monoxide",
            []
        )

        no2 = hourly.get(
            "nitrogen_dioxide",
            []
        )

        so2 = hourly.get(
            "sulphur_dioxide",
            []
        )

        o3 = hourly.get(
            "ozone",
            []
        )

        us_aqi = hourly.get(
            "us_aqi",
            []
        )

        daily = {}

        # ----------------------------------------------------
        # Hourly -> daily
        # ----------------------------------------------------

        for i, timestamp in enumerate(times):

            current_date = date.fromisoformat(
                timestamp[:10]
            )

            if current_date in existing_dates:
                continue

            if current_date not in daily:

                daily[current_date] = {

                    "aqi": [],
                    "pm25": [],
                    "pm10": [],
                    "co": [],
                    "no2": [],
                    "so2": [],
                    "o3": []

                }

            # AQI

            if i < len(us_aqi):

                value = us_aqi[i]

                if value is not None:

                    daily[current_date][
                        "aqi"
                    ].append(
                        float(value)
                    )

            # PM2.5

            if i < len(pm25):

                value = pm25[i]

                if value is not None:

                    daily[current_date][
                        "pm25"
                    ].append(
                        float(value)
                    )

            # PM10

            if i < len(pm10):

                value = pm10[i]

                if value is not None:

                    daily[current_date][
                        "pm10"
                    ].append(
                        float(value)
                    )

            # CO

            if i < len(co):

                value = co[i]

                if value is not None:

                    daily[current_date][
                        "co"
                    ].append(
                        float(value)
                    )

            # NO2

            if i < len(no2):

                value = no2[i]

                if value is not None:

                    daily[current_date][
                        "no2"
                    ].append(
                        float(value)
                    )

            # SO2

            if i < len(so2):

                value = so2[i]

                if value is not None:

                    daily[current_date][
                        "so2"
                    ].append(
                        float(value)
                    )

            # O3

            if i < len(o3):

                value = o3[i]

                if value is not None:

                    daily[current_date][
                        "o3"
                    ].append(
                        float(value)
                    )

        # ----------------------------------------------------
        # Build database records
        # ----------------------------------------------------

        for current_date, values in daily.items():

            if not values["aqi"]:
                continue

            records.append(
                (
                    None,
                    city_id,
                    current_date,
                    float(
                        np.median(
                            values["aqi"]
                        )
                    ),
                    safe_average(
                        values["pm25"]
                    ),
                    safe_average(
                        values["pm10"]
                    ),
                    safe_average(
                        values["no2"]
                    ),
                    safe_average(
                        values["so2"]
                    ),
                    safe_average(
                        values["co"]
                    ),
                    safe_average(
                        values["o3"]
                    ),
                    "Open-Meteo"
                )
            )

        time.sleep(1)

    return records


# ============================================================
# WEATHER DOWNLOAD
# ============================================================

def download_weather_for_city(
    city,
    start_date,
    end_date,
):

    if start_date > end_date:
        return []

    city_id = int(city["city_id"])
    city_name = str(city["city_name"])

    existing_dates = get_existing_dates(
        "weather",
        city_id
    )

    records = []

    print()
    print("-" * 70)
    print(
        f"WEATHER UPDATE: {city_name}"
    )
    print("-" * 70)

    for chunk_start, chunk_end in generate_date_chunks(
        start_date,
        end_date
    ):

        print(
            f"Downloading weather "
            f"{chunk_start} -> {chunk_end}"
        )

        params = {

            "latitude":
                city["latitude"],

            "longitude":
                city["longitude"],

            "start_date":
                chunk_start.isoformat(),

            "end_date":
                chunk_end.isoformat(),

            "daily": (
                "temperature_2m_mean,"
                "relative_humidity_2m_mean,"
                "precipitation_sum,"
                "windspeed_10m_max"
            ),

            "timezone":
                TIMEZONE

        }

        try:

            data = client.get(
                WEATHER_URL,
                params
            )

        except Exception as error:

            print(
                f"WARNING: Weather request failed "
                f"for {city_name}: {error}"
            )

            continue

        daily = data.get("daily")

        if not daily:

            print(
                "WARNING: No weather data returned."
            )

            continue

        dates = daily.get(
            "time",
            []
        )

        temperature = daily.get(
            "temperature_2m_mean",
            []
        )

        humidity = daily.get(
            "relative_humidity_2m_mean",
            []
        )

        precipitation = daily.get(
            "precipitation_sum",
            []
        )

        windspeed = daily.get(
            "windspeed_10m_max",
            []
        )

        for i, current_date_string in enumerate(
            dates
        ):

            current_date = date.fromisoformat(
                current_date_string
            )

            if current_date in existing_dates:
                continue

            records.append(
                (
                    None,
                    city_id,
                    current_date,

                    safe_index(
                        temperature,
                        i
                    ),

                    safe_index(
                        humidity,
                        i
                    ),

                    safe_index(
                        precipitation,
                        i
                    ),

                    safe_index(
                        windspeed,
                        i
                    )
                )
            )

        time.sleep(1)

    return records


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_average(values):

    if not values:
        return None

    return float(
        np.mean(values)
    )


def safe_index(values, index):

    if index >= len(values):
        return None

    value = values[index]

    if value is None:
        return None

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


# ============================================================
# INSERT AQI
# ============================================================

def insert_aqi_records(records):

    if not records:
        return 0

    conn = get_database_connection()

    next_id = conn.execute(
        """
        SELECT COALESCE(
            MAX(id),
            0
        ) + 1
        FROM aqi
        """
    ).fetchone()[0]

    prepared = []

    for record in records:

        prepared.append(
            (
                next_id,
                *record[1:]
            )
        )

        next_id += 1

    conn.executemany(
        """
        INSERT INTO aqi (
            id,
            city_id,
            date,
            aqi,
            pm25,
            pm10,
            no2,
            so2,
            co,
            o3,
            aqi_source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        prepared
    )

    conn.close()

    return len(prepared)


# ============================================================
# INSERT WEATHER
# ============================================================

def insert_weather_records(records):

    if not records:
        return 0

    conn = get_database_connection()

    next_id = conn.execute(
        """
        SELECT COALESCE(
            MAX(id),
            0
        ) + 1
        FROM weather
        """
    ).fetchone()[0]

    prepared = []

    for record in records:

        prepared.append(
            (
                next_id,
                *record[1:]
            )
        )

        next_id += 1

    conn.executemany(
        """
        INSERT INTO weather (
            id,
            city_id,
            date,
            temperature,
            humidity,
            precipitation,
            windspeed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        prepared
    )

    conn.close()

    return len(prepared)


# ============================================================
# DATABASE COVERAGE
# ============================================================

def print_database_coverage():

    conn = get_database_connection()

    aqi = conn.execute(
        """
        SELECT
            MIN(date),
            MAX(date),
            COUNT(*)
        FROM aqi
        """
    ).fetchone()

    weather = conn.execute(
        """
        SELECT
            MIN(date),
            MAX(date),
            COUNT(*)
        FROM weather
        """
    ).fetchone()

    conn.close()

    print()
    print("=" * 70)
    print("DATABASE COVERAGE")
    print("=" * 70)

    print(
        f"AQI     : {aqi[0]} -> {aqi[1]} "
        f"({aqi[2]:,} rows)"
    )

    print(
        f"Weather : {weather[0]} -> {weather[1]} "
        f"({weather[2]:,} rows)"
    )


# ============================================================
# REBUILD FEATURES
# ============================================================

def rebuild_features():

    print()
    print("=" * 70)
    print("REBUILDING V4 FEATURES")
    print("=" * 70)

    from src.features.feature_builder_v4 import (
        build_features
    )

    from src.features.feature_store import (
        initialise_feature_store,
        build_feature_store
    )

    # --------------------------------------------------------
    # Step 1: Build the latest V4 production features
    # --------------------------------------------------------

    build_features()

    # --------------------------------------------------------
    # Step 2: Load the freshly generated V4 features
    # --------------------------------------------------------

    feature_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "features_v4.parquet"
    )

    if not feature_file.exists():

        raise FileNotFoundError(
            f"V4 feature file was not created: "
            f"{feature_file}"
        )

    latest_features = pd.read_parquet(
        feature_file
    )

    if latest_features.empty:

        raise ValueError(
            "V4 feature pipeline produced 0 rows."
        )

    # --------------------------------------------------------
    # Step 3: Update Local Feature Store incrementally
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("UPDATING LOCAL FEATURE STORE")
    print("=" * 70)

    initialise_feature_store()

    build_feature_store(
        latest_features
    )

    print()
    print("V4 features successfully synchronized")
    print("with the Local Feature Store.")

# ============================================================
# GENERATE FORECAST
# ============================================================

def generate_forecast():

    print()
    print("=" * 70)
    print("GENERATING PRODUCTION V9 FORECAST")
    print("=" * 70)

    from src.ml.forecast_engine_v9 import main

    main()


# ============================================================
# MAIN DAILY PIPELINE
# ============================================================

def run_daily_update():

    print()
    print("=" * 70)
    print("PEARLSAQI DAILY PRODUCTION UPDATE")
    print("=" * 70)

    print(
        "Mode: INCREMENTAL"
    )

    print(
        "Historical data will NOT be deleted."
    )

    # --------------------------------------------------------
    # Current overall coverage
    # --------------------------------------------------------

    aqi_latest, weather_latest = (
        get_latest_dates()
    )

    print()
    print(
        f"Current AQI latest date     : "
        f"{aqi_latest}"
    )

    print(
        f"Current weather latest date : "
        f"{weather_latest}"
    )

    # --------------------------------------------------------
    # Determine update end date
    # --------------------------------------------------------

    today = date.today()

    target_end = (
        today - timedelta(days=1)
    )

    print(
        f"Update target end date      : "
        f"{target_end}"
    )

    cities = load_cities()

    total_aqi = 0
    total_weather = 0

    # --------------------------------------------------------
    # Download missing data
    # --------------------------------------------------------

    for _, city in cities.iterrows():

        city_id = int(
            city["city_id"]
        )

        city_name = str(
            city["city_name"]
        )

        # ----------------------------------------------------
        # CITY-SPECIFIC LATEST DATES
        # ----------------------------------------------------

        city_aqi_latest = (
            get_latest_date_for_city(
                "aqi",
                city_id
            )
        )

        city_weather_latest = (
            get_latest_date_for_city(
                "weather",
                city_id
            )
        )

        print()
        print("=" * 70)
        print(
            f"SYNCING CITY: {city_name}"
        )
        print("=" * 70)

        print(
            f"Latest AQI for {city_name}: "
            f"{city_aqi_latest}"
        )

        print(
            f"Latest weather for {city_name}: "
            f"{city_weather_latest}"
        )

        # ----------------------------------------------------
        # AQI START DATE
        # ----------------------------------------------------

        if city_aqi_latest is None:

            aqi_start = date(
                2023,
                1,
                1
            )

        else:

            aqi_start = (
                city_aqi_latest
                + timedelta(days=1)
            )

        # ----------------------------------------------------
        # WEATHER START DATE
        # ----------------------------------------------------

        if city_weather_latest is None:

            weather_start = date(
                2023,
                1,
                1
            )

        else:

            weather_start = (
                city_weather_latest
                + timedelta(days=1)
            )

        # ----------------------------------------------------
        # AQI DOWNLOAD
        # ----------------------------------------------------

        aqi_records = (
            download_aqi_for_city(
                city,
                aqi_start,
                target_end
            )
        )

        inserted_aqi = (
            insert_aqi_records(
                aqi_records
            )
        )

        total_aqi += inserted_aqi

        # ----------------------------------------------------
        # WEATHER DOWNLOAD
        # ----------------------------------------------------

        weather_records = (
            download_weather_for_city(
                city,
                weather_start,
                target_end
            )
        )

        inserted_weather = (
            insert_weather_records(
                weather_records
            )
        )

        total_weather += inserted_weather

    # --------------------------------------------------------
    # Coverage after update
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INCREMENTAL UPDATE COMPLETE")
    print("=" * 70)

    print(
        f"New AQI rows     : {total_aqi}"
    )

    print(
        f"New weather rows : {total_weather}"
    )

    print_database_coverage()

    # --------------------------------------------------------
    # --------------------------------------------------------
    # REFRESH PRODUCTION FORECAST
    #
    # Forecasting MUST run even when no new historical rows
    # were inserted during this execution.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REFRESHING PRODUCTION FORECAST")
    print("=" * 70)

    if total_aqi > 0 or total_weather > 0:

        print(
            f"New historical AQI rows     : {total_aqi}"
        )

        print(
            f"New historical weather rows : {total_weather}"
        )

    else:

        print(
            "No new historical rows were inserted."
        )

    print(
        "Forecast generation will run regardless."
    )

    # --------------------------------------------------------
    # STEP 1: Update future weather
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("UPDATING FUTURE WEATHER FORECAST")
    print("=" * 70)

    from src.services.weather_forecast_service import (
        WeatherForecastService
    )

    future_weather_service = (
        WeatherForecastService()
    )

    future_weather_service.download_forecast()

    print(
        "Future weather forecast updated."
    )

    # --------------------------------------------------------
    # STEP 2: Rebuild V4 production features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REBUILDING V4 FEATURES")
    print("=" * 70)


    rebuild_features()

    print(
        "V4 features rebuilt successfully."
    )

    # --------------------------------------------------------
    # STEP 3: Generate 24h / 48h / 72h forecast
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING V9 PRODUCTION FORECAST")
    print("=" * 70)

    generate_forecast()

    print(
        "24h / 48h / 72h forecast generated successfully."
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    final_aqi, final_weather = (
        get_latest_dates()
    )

    print()
    print("=" * 70)
    print("DAILY PRODUCTION PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Latest actual AQI date     : "
        f"{final_aqi}"
    )

    print(
        f"Latest actual weather date : "
        f"{final_weather}"
    )

    print(
        "Forecast horizons          : "
        "24h / 48h / 72h"
    )

    print()
    print(
        "The system is ready for the "
        "next automated run."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_daily_update()

