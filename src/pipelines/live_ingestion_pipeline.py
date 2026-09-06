import requests
import duckdb
import time
from datetime import datetime, timezone



# ============================================================
# PEARLSAQI LIVE AQI INGESTION PIPELINE
# ============================================================
#
# Purpose:
#
#   Fetch the newest available AQI + pollutant observations
#   from Open-Meteo and store them in the local DuckDB database.
#
# This pipeline does NOT train a model.
#
# It only updates:
#
#   database/aqi.duckdb
#
# ============================================================


DB_PATH = "database/aqi.duckdb"

OPEN_METEO_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# CITY CONFIGURATION
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


# ============================================================
# FETCH ONE CITY
# ============================================================

def fetch_city(city):

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],

        "hourly": (
            "pm2_5,"
            "pm10,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "carbon_monoxide,"
            "ozone,"
            "us_aqi"
        ),

        "timezone": "UTC",

        # Get enough recent data so the pipeline can
        # determine the newest available observation.
        "past_days": 2,

        "forecast_days": 0,
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# DAILY AGGREGATION
# ============================================================

def aggregate_daily(city, data):

    hourly = data.get("hourly")

    if not hourly:
        raise RuntimeError(
            f"No hourly data returned for {city['city_name']}"
        )

    times = hourly.get("time", [])

    if not times:
        raise RuntimeError(
            f"No timestamps returned for {city['city_name']}"
        )

    rows = []

    for i, timestamp in enumerate(times):

        def value(key):

            values = hourly.get(key, [])

            if i >= len(values):
                return None

            return values[i]

        rows.append(
            {
                "city_id": city["city_id"],
                "date": timestamp[:10],

                "aqi": value("us_aqi"),
                "pm25": value("pm2_5"),
                "pm10": value("pm10"),
                "no2": value("nitrogen_dioxide"),
                "so2": value("sulphur_dioxide"),
                "co": value("carbon_monoxide"),
                "o3": value("ozone"),
            }
        )

    # --------------------------------------------------------
    # Aggregate hourly observations to daily means.
    #
    # AQI is averaged in the same way as the existing
    # database convention.
    # --------------------------------------------------------

    daily = {}

    for row in rows:

        date = row["date"]

        if date not in daily:
            daily[date] = {
                "city_id": row["city_id"],
                "date": date,
                "aqi": [],
                "pm25": [],
                "pm10": [],
                "no2": [],
                "so2": [],
                "co": [],
                "o3": [],
            }

        for key in (
            "aqi",
            "pm25",
            "pm10",
            "no2",
            "so2",
            "co",
            "o3",
        ):

            value = row[key]

            if value is not None:
                daily[date][key].append(
                    float(value)
                )

    output = []

    for date, values in daily.items():

        def mean(key):

            numbers = values[key]

            if not numbers:
                return None

            return sum(numbers) / len(numbers)

        output.append(
            {
                "city_id": values["city_id"],
                "date": date,

                "aqi": mean("aqi"),
                "pm25": mean("pm25"),
                "pm10": mean("pm10"),
                "no2": mean("no2"),
                "so2": mean("so2"),
                "co": mean("co"),
                "o3": mean("o3"),

                "aqi_source": "Open-Meteo",
            }
        )

    return output


# ============================================================
# INSERT INTO DUCKDB
# ============================================================

def insert_rows(conn, rows):

    inserted = 0
    skipped = 0

    for row in rows:

        existing = conn.execute(
            """
            SELECT COUNT(*)
            FROM aqi
            WHERE city_id = ?
              AND date = ?
            """,
            [
                row["city_id"],
                row["date"],
            ],
        ).fetchone()[0]

        if existing:

            skipped += 1
            continue

        next_id = conn.execute(
            """
            SELECT COALESCE(
                MAX(id),
                0
            ) + 1
            FROM aqi
            """
        ).fetchone()[0]

        conn.execute(
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
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                next_id,
                row["city_id"],
                row["date"],
                row["aqi"],
                row["pm25"],
                row["pm10"],
                row["no2"],
                row["so2"],
                row["co"],
                row["o3"],
                row["aqi_source"],
            ],
        )

        inserted += 1

    return inserted, skipped


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PEARLSAQI LIVE AQI INGESTION")
    print("=" * 70)

    print()
    print("Database:", DB_PATH)
    print("Cities  :", len(CITIES))
    print()

    conn = duckdb.connect(DB_PATH)

    all_rows = []

    for index, city in enumerate(CITIES, start=1):

        print(
            f"[{index:02d}/{len(CITIES)}] "
            f"Fetching {city['city_name']}..."
        )

        try:

            data = fetch_city(city)

            rows = aggregate_daily(
                city,
                data,
            )

            all_rows.extend(rows)

            if rows:
                latest = max(
                    row["date"]
                    for row in rows
                )

                print(
                    f"       Latest available: "
                    f"{latest}"
                )

        except Exception as e:

            print(
                f"       FAILED: "
                f"{type(e).__name__}: {e}"
            )

        # Be polite to the API.
        time.sleep(0.5)

    print()
    print("=" * 70)
    print("DATABASE UPDATE")
    print("=" * 70)

    inserted, skipped = insert_rows(
        conn,
        all_rows,
    )

    conn.commit()

    latest_db = conn.execute(
        """
        SELECT MAX(date)
        FROM aqi
        """
    ).fetchone()[0]

    city_count = conn.execute(
        """
        SELECT COUNT(DISTINCT city_id)
        FROM aqi
        WHERE date = ?
        """,
        [latest_db],
    ).fetchone()[0]

    conn.close()

    print()
    print("Rows fetched :", len(all_rows))
    print("Inserted     :", inserted)
    print("Skipped      :", skipped)
    print("Latest date  :", latest_db)
    print("Cities       :", city_count)

    print()
    print("=" * 70)
    print("LIVE INGESTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()