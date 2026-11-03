import os
import sys
import time
import requests
import duckdb
from datetime import datetime, timedelta, timezone

# ============================================================
# PEARLSAQI HOURLY LIVE INGESTION
# ============================================================
#
# Fetches fresh hourly AQI + weather data from Open-Meteo.
#
# IMPORTANT:
# - Does NOT retrain the model.
# - Does NOT modify the production model.
# - Stores hourly data separately from the existing daily tables.
# - Safe to run repeatedly because records are deduplicated.
#
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "database",
    "aqi.duckdb",
)

OPEN_METEO_AIR_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

OPEN_METEO_WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# CITIES
# ============================================================

CITIES = [
    {
        "city_id": 1,
        "city_name": "Karachi",
        "latitude": 24.8607,
        "longitude": 67.0011,
    },
    {
        "city_id": 2,
        "city_name": "Lahore",
        "latitude": 31.5204,
        "longitude": 74.3587,
    },
    {
        "city_id": 3,
        "city_name": "Islamabad",
        "latitude": 33.6844,
        "longitude": 73.0479,
    },
    {
        "city_id": 4,
        "city_name": "Rawalpindi",
        "latitude": 33.5651,
        "longitude": 73.0169,
    },
    {
        "city_id": 5,
        "city_name": "Faisalabad",
        "latitude": 31.4504,
        "longitude": 73.1350,
    },
    {
        "city_id": 6,
        "city_name": "Multan",
        "latitude": 30.1575,
        "longitude": 71.5249,
    },
    {
        "city_id": 7,
        "city_name": "Peshawar",
        "latitude": 34.0151,
        "longitude": 71.5249,
    },
    {
        "city_id": 8,
        "city_name": "Quetta",
        "latitude": 30.1798,
        "longitude": 66.9750,
    },
    {
        "city_id": 9,
        "city_name": "Hyderabad",
        "latitude": 25.3960,
        "longitude": 68.3578,
    },
    {
        "city_id": 10,
        "city_name": "Gujranwala",
        "latitude": 32.1877,
        "longitude": 74.1945,
    },
    {
        "city_id": 11,
        "city_name": "Sialkot",
        "latitude": 32.4945,
        "longitude": 74.5229,
    },
    {
        "city_id": 12,
        "city_name": "Bahawalpur",
        "latitude": 29.3956,
        "longitude": 71.6836,
    },
]


PAST_DAYS = int(os.environ.get("HOURLY_PAST_DAYS", "2"))

# Open-Meteo returns local times for timezone=Asia/Karachi (UTC+5, no DST).
PAKISTAN_TZ = timezone(timedelta(hours=5))


def is_future_hour(timestamp):
    """
    True for hours later than the current Pakistan hour. Those values
    are forecasts and must not be stored as observations.
    """

    now = datetime.now(PAKISTAN_TZ).replace(tzinfo=None)

    return datetime.fromisoformat(timestamp) > now


REQUEST_ATTEMPTS = 4


def get_json(url, params):
    """
    GET with retries. Large backfills (HOURLY_PAST_DAYS up to 92) can
    time out on Open-Meteo, so retry with a growing wait and a longer
    timeout before giving up on a city.
    """

    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == REQUEST_ATTEMPTS:
                raise
            wait = 10 * attempt
            print(f"       Request failed ({type(error).__name__}), retrying in {wait}s "
                  f"[{attempt}/{REQUEST_ATTEMPTS}]")
            time.sleep(wait)


# ============================================================
# DATABASE
# ============================================================

def connect_database():

    os.makedirs(
        os.path.dirname(DATABASE_PATH),
        exist_ok=True,
    )

    return duckdb.connect(
        DATABASE_PATH
    )


# ============================================================
# TABLES
# ============================================================

def create_tables(conn):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aqi_hourly (
            id BIGINT,
            city_id INTEGER NOT NULL,
            city_name VARCHAR,
            timestamp TIMESTAMP NOT NULL,

            aqi DOUBLE,
            pm25 DOUBLE,
            pm10 DOUBLE,
            no2 DOUBLE,
            so2 DOUBLE,
            co DOUBLE,
            o3 DOUBLE,

            aqi_source VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(city_id, timestamp)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_hourly (
            id BIGINT,
            city_id INTEGER NOT NULL,
            city_name VARCHAR,
            timestamp TIMESTAMP NOT NULL,

            temperature DOUBLE,
            relative_humidity DOUBLE,
            precipitation DOUBLE,
            windspeed DOUBLE,

            weather_source VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(city_id, timestamp)
        )
        """
    )

    # Upcoming hours (CAMS forecast) for the dashboard's hourly outlook.
    # Kept separate from aqi_hourly, which holds observations only, and
    # replaced per city on every run.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aqi_hourly_forecast (
            city_id INTEGER NOT NULL,
            city_name VARCHAR,
            timestamp TIMESTAMP NOT NULL,
            aqi DOUBLE,
            pm25 DOUBLE,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(city_id, timestamp)
        )
        """
    )


# ============================================================
# AIR QUALITY API
# ============================================================

def fetch_air_quality(city):

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],

        "hourly": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "carbon_monoxide,"
            "ozone"
        ),

        "timezone": "Asia/Karachi",

        # Recent/current data; HOURLY_PAST_DAYS (max 92) re-fetches
        # older hours, e.g. to repair a backfill.
        "past_days": PAST_DAYS,

        # Today plus two days ahead: future hours feed the forecast
        # table, never the observations.
        "forecast_days": 3,
    }

    return get_json(OPEN_METEO_AIR_URL, params)


# ============================================================
# WEATHER API
# ============================================================

def fetch_weather(city):

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),

        "timezone": "Asia/Karachi",

        "past_days": PAST_DAYS,

        "forecast_days": 1,
    }

    return get_json(OPEN_METEO_WEATHER_URL, params)


# ============================================================
# SAVE AIR QUALITY FORECAST
# ============================================================

def save_air_forecast(conn, city, payload):
    """Replace the city's stored forecast with this run's future hours."""

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    aqi = hourly.get("us_aqi", [])
    pm25 = hourly.get("pm2_5", [])

    rows = [
        [
            city["city_id"],
            city["city_name"],
            timestamp,
            aqi[i] if i < len(aqi) else None,
            pm25[i] if i < len(pm25) else None,
        ]
        for i, timestamp in enumerate(times)
        if is_future_hour(timestamp)
    ]

    conn.execute(
        "DELETE FROM aqi_hourly_forecast WHERE city_id = ?",
        [city["city_id"]],
    )

    if rows:
        conn.executemany(
            """
            INSERT INTO aqi_hourly_forecast
                (city_id, city_name, timestamp, aqi, pm25)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    return len(rows)


# ============================================================
# INSERT AIR QUALITY
# ============================================================

def insert_air_quality(
    conn,
    city,
    payload,
):

    hourly = payload.get(
        "hourly",
        {},
    )

    times = hourly.get(
        "time",
        [],
    )

    values = {
        "aqi": hourly.get(
            "us_aqi",
            [],
        ),

        "pm25": hourly.get(
            "pm2_5",
            [],
        ),

        "pm10": hourly.get(
            "pm10",
            [],
        ),

        "no2": hourly.get(
            "nitrogen_dioxide",
            [],
        ),

        "so2": hourly.get(
            "sulphur_dioxide",
            [],
        ),

        "co": hourly.get(
            "carbon_monoxide",
            [],
        ),

        "o3": hourly.get(
            "ozone",
            [],
        ),
    }

    inserted = 0

    for i, timestamp in enumerate(times):

        if is_future_hour(timestamp):
            continue

        row = [
            city["city_id"],
            city["city_name"],
            timestamp,
            values["aqi"][i]
            if i < len(values["aqi"])
            else None,
            values["pm25"][i]
            if i < len(values["pm25"])
            else None,
            values["pm10"][i]
            if i < len(values["pm10"])
            else None,
            values["no2"][i]
            if i < len(values["no2"])
            else None,
            values["so2"][i]
            if i < len(values["so2"])
            else None,
            values["co"][i]
            if i < len(values["co"])
            else None,
            values["o3"][i]
            if i < len(values["o3"])
            else None,
            "Open-Meteo",
        ]

        result = conn.execute(
            """
            INSERT OR REPLACE INTO aqi_hourly (
                city_id,
                city_name,
                timestamp,
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
            row,
        )

        if result.rowcount:
            inserted += 1

    return inserted


# ============================================================
# INSERT WEATHER
# ============================================================

def insert_weather(
    conn,
    city,
    payload,
):

    hourly = payload.get(
        "hourly",
        {},
    )

    times = hourly.get(
        "time",
        [],
    )

    temperature = hourly.get(
        "temperature_2m",
        [],
    )

    humidity = hourly.get(
        "relative_humidity_2m",
        [],
    )

    precipitation = hourly.get(
        "precipitation",
        [],
    )

    windspeed = hourly.get(
        "wind_speed_10m",
        [],
    )

    inserted = 0

    for i, timestamp in enumerate(times):

        if is_future_hour(timestamp):
            continue

        row = [
            city["city_id"],
            city["city_name"],
            timestamp,

            temperature[i]
            if i < len(temperature)
            else None,

            humidity[i]
            if i < len(humidity)
            else None,

            precipitation[i]
            if i < len(precipitation)
            else None,

            windspeed[i]
            if i < len(windspeed)
            else None,

            "Open-Meteo",
        ]

        result = conn.execute(
            """
            INSERT OR REPLACE INTO weather_hourly (
                city_id,
                city_name,
                timestamp,
                temperature,
                relative_humidity,
                precipitation,
                windspeed,
                weather_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )

        if result.rowcount:
            inserted += 1

    return inserted


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLSAQI HOURLY LIVE INGESTION")
    print("=" * 70)

    print()
    print(
        "Database:",
        DATABASE_PATH,
    )

    print(
        "Cities  :",
        len(CITIES),
    )

    conn = connect_database()

    try:

        create_tables(conn)

        # Observation tables must never hold hours that haven't happened.
        # Older versions stored forecast hours there; drop any leftovers
        # so they can't be read as observations.
        now = datetime.now(PAKISTAN_TZ).replace(tzinfo=None)
        for table in ("aqi_hourly", "weather_hourly"):
            conn.execute(f"DELETE FROM {table} WHERE timestamp > ?", [now])

        total_aqi = 0
        total_weather = 0
        failed_cities = []

        for index, city in enumerate(
            CITIES,
            start=1,
        ):

            print()
            print(
                f"[{index:02d}/{len(CITIES)}] "
                f"{city['city_name']}"
            )

            try:

                air_payload = fetch_air_quality(
                    city
                )

                weather_payload = fetch_weather(
                    city
                )

                aqi_count = insert_air_quality(
                    conn,
                    city,
                    air_payload,
                )

                forecast_count = save_air_forecast(
                    conn,
                    city,
                    air_payload,
                )

                print(
                    f"       Forecast hours  : "
                    f"{forecast_count}"
                )

                weather_count = insert_weather(
                    conn,
                    city,
                    weather_payload,
                )

                total_aqi += aqi_count
                total_weather += weather_count

                print(
                    f"       AQI inserted    : "
                    f"{aqi_count}"
                )

                print(
                    f"       Weather inserted: "
                    f"{weather_count}"
                )

            except Exception as e:

                failed_cities.append(city["city_name"])

                print(
                    f"       FAILED: "
                    f"{type(e).__name__}: {e}"
                )

        conn.commit()

        print()
        print("=" * 70)
        print("HOURLY INGESTION COMPLETE")
        print("=" * 70)

        print(
            "AQI rows inserted    :",
            total_aqi,
        )

        print(
            "Weather rows inserted:",
            total_weather,
        )

        print(
            "Run time             :",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )

        # Rows for the cities that worked are already committed. Exit
        # non-zero so the workflow retries instead of silently leaving
        # some cities stale.
        if failed_cities:
            print("FAILED CITIES        :", ", ".join(failed_cities))
            sys.exit(1)

    finally:

        conn.close()


if __name__ == "__main__":
    main()
