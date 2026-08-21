from pathlib import Path
import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "aqi.duckdb"


def main():

    print("=" * 70)
    print("PEARLSAQI HOURLY → DAILY SYNCHRONIZATION")
    print("=" * 70)

    print()
    print("Database:", DB_PATH)

    conn = duckdb.connect(str(DB_PATH))

    try:

        # ============================================================
        # 1. REQUIRED TABLES
        # ============================================================

        required_tables = [
            "aqi",
            "weather",
            "aqi_hourly",
            "weather_hourly",
        ]

        existing_tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

        missing = [
            table
            for table in required_tables
            if table not in existing_tables
        ]

        if missing:
            raise RuntimeError(
                "Missing required tables: " + ", ".join(missing)
            )

        # ============================================================
        # 2. HOURLY COVERAGE
        # ============================================================

        hourly_range = conn.execute(
            """
            SELECT
                MIN(CAST(timestamp AS DATE)),
                MAX(CAST(timestamp AS DATE))
            FROM aqi_hourly
            """
        ).fetchone()

        min_date, max_date = hourly_range

        if min_date is None:
            print()
            print("No hourly AQI data found.")
            return

        print()
        print("Hourly AQI coverage:")
        print("  From:", min_date)
        print("  To  :", max_date)

        # ============================================================
        # 3. DAILY AQI TEMP TABLE
        # ============================================================

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE daily_aqi_temp AS

            SELECT
                city_id,
                CAST(timestamp AS DATE) AS date,
                AVG(aqi)  AS aqi,
                AVG(pm25) AS pm25,
                AVG(pm10) AS pm10,
                AVG(no2)  AS no2,
                AVG(so2)  AS so2,
                AVG(co)   AS co,
                AVG(o3)   AS o3

            FROM aqi_hourly

            GROUP BY
                city_id,
                CAST(timestamp AS DATE)
            """
        )

        # ============================================================
        # 4. DAILY WEATHER TEMP TABLE
        # ============================================================

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE daily_weather_temp AS

            SELECT
                city_id,
                CAST(timestamp AS DATE) AS date,
                AVG(temperature)       AS temperature,
                AVG(relative_humidity) AS humidity,
                SUM(precipitation)     AS precipitation,
                AVG(windspeed)         AS windspeed

            FROM weather_hourly

            GROUP BY
                city_id,
                CAST(timestamp AS DATE)
            """
        )

        daily_aqi_count = conn.execute(
            "SELECT COUNT(*) FROM daily_aqi_temp"
        ).fetchone()[0]

        daily_weather_count = conn.execute(
            "SELECT COUNT(*) FROM daily_weather_temp"
        ).fetchone()[0]

        print()
        print("Daily AQI aggregates    :", daily_aqi_count)
        print("Daily weather aggregates:", daily_weather_count)

        # ============================================================
        # 5. KEEP ONLY COMPLETE 12-CITY DATES
        # ============================================================

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE complete_dates AS

            SELECT a.date

            FROM (
                SELECT
                    CAST(timestamp AS DATE) AS date
                FROM aqi_hourly
                GROUP BY CAST(timestamp AS DATE)
                HAVING COUNT(DISTINCT city_id) = 12
            ) a

            INNER JOIN (
                SELECT
                    CAST(timestamp AS DATE) AS date
                FROM weather_hourly
                GROUP BY CAST(timestamp AS DATE)
                HAVING COUNT(DISTINCT city_id) = 12
            ) w

            ON a.date = w.date
            """
        )

        complete_count = conn.execute(
            "SELECT COUNT(*) FROM complete_dates"
        ).fetchone()[0]

        print()
        print("Complete 12-city dates:", complete_count)

        if complete_count == 0:
            raise RuntimeError(
                "No dates have complete 12-city AQI + weather coverage."
            )

        # ============================================================
        # 6. COUNT BEFORE
        # ============================================================

        before_aqi = conn.execute(
            "SELECT COUNT(*) FROM aqi"
        ).fetchone()[0]

        before_weather = conn.execute(
            "SELECT COUNT(*) FROM weather"
        ).fetchone()[0]

        # ============================================================
        # 7. REBUILD DAILY AQI FOR COMPLETE DATES
        # ============================================================

        conn.execute(
            """
            DELETE FROM aqi
            WHERE date IN (
                SELECT date
                FROM complete_dates
            )
            """
        )

        conn.execute(
            """
            INSERT INTO aqi (
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

            SELECT
                d.city_id,
                d.date,
                d.aqi,
                d.pm25,
                d.pm10,
                d.no2,
                d.so2,
                d.co,
                d.o3,
                'Open-Meteo-Hourly-Aggregated'

            FROM daily_aqi_temp d

            INNER JOIN complete_dates c
                ON d.date = c.date
            """
        )

        # ============================================================
        # 8. REBUILD DAILY WEATHER FOR COMPLETE DATES
        # ============================================================

        conn.execute(
            """
            DELETE FROM weather
            WHERE date IN (
                SELECT date
                FROM complete_dates
            )
            """
        )

        conn.execute(
            """
            INSERT INTO weather (
                city_id,
                date,
                temperature,
                humidity,
                precipitation,
                windspeed
            )

            SELECT
                d.city_id,
                d.date,
                d.temperature,
                d.humidity,
                d.precipitation,
                d.windspeed

            FROM daily_weather_temp d

            INNER JOIN complete_dates c
                ON d.date = c.date
            """
        )

        # ============================================================
        # 9. FINAL COUNTS
        # ============================================================

        after_aqi = conn.execute(
            "SELECT COUNT(*) FROM aqi"
        ).fetchone()[0]

        after_weather = conn.execute(
            "SELECT COUNT(*) FROM weather"
        ).fetchone()[0]

        latest_aqi = conn.execute(
            "SELECT MAX(date) FROM aqi"
        ).fetchone()[0]

        latest_weather = conn.execute(
            "SELECT MAX(date) FROM weather"
        ).fetchone()[0]

        # ============================================================
        # 10. LATEST COVERAGE
        # ============================================================

        aqi_validation = conn.execute(
            """
            SELECT
                date,
                COUNT(DISTINCT city_id),
                COUNT(*)
            FROM aqi
            WHERE date = (SELECT MAX(date) FROM aqi)
            GROUP BY date
            """
        ).fetchone()

        weather_validation = conn.execute(
            """
            SELECT
                date,
                COUNT(DISTINCT city_id),
                COUNT(*)
            FROM weather
            WHERE date = (SELECT MAX(date) FROM weather)
            GROUP BY date
            """
        ).fetchone()

        # ============================================================
        # 11. DISPLAY
        # ============================================================

        print()
        print("=" * 70)
        print("DATABASE SYNCHRONIZATION")
        print("=" * 70)

        print()
        print("AQI rows before    :", before_aqi)
        print("AQI rows after     :", after_aqi)
        print("Weather rows before:", before_weather)
        print("Weather rows after :", after_weather)

        print()
        print("Latest AQI date    :", latest_aqi)
        print("Latest weather date:", latest_weather)

        print()
        print("=" * 70)
        print("LATEST AQI COVERAGE")
        print("=" * 70)

        print("date  :", aqi_validation[0])
        print("cities:", aqi_validation[1])
        print("rows  :", aqi_validation[2])

        print()
        print("=" * 70)
        print("LATEST WEATHER COVERAGE")
        print("=" * 70)

        print("date  :", weather_validation[0])
        print("cities:", weather_validation[1])
        print("rows  :", weather_validation[2])

        # ============================================================
        # 12. SAFETY VALIDATION
        # ============================================================

        if latest_aqi != latest_weather:
            raise RuntimeError(
                "AQI and weather latest dates do not match."
            )

        if aqi_validation[1] != 12:
            raise RuntimeError(
                "Latest AQI date does not contain all 12 cities."
            )

        if weather_validation[1] != 12:
            raise RuntimeError(
                "Latest weather date does not contain all 12 cities."
            )

        if aqi_validation[2] != 12:
            raise RuntimeError(
                "Latest AQI date does not contain exactly 12 rows."
            )

        if weather_validation[2] != 12:
            raise RuntimeError(
                "Latest weather date does not contain exactly 12 rows."
            )

        print()
        print("=" * 70)
        print("HOURLY → DAILY SYNCHRONIZATION PASSED")
        print("=" * 70)

        print()
        print("AQI latest date    :", latest_aqi)
        print("Weather latest date:", latest_weather)
        print("Cities             : 12")
        print()
        print("Existing model was NOT retrained.")
        print("Production model was NOT modified.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
