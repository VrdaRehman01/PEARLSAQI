import os
import duckdb


DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "database",
        "aqi.duckdb",
    )
)


def main():

    print("=" * 70)
    print("PEARLSAQI HOURLY → DAILY SYNCHRONIZATION")
    print("=" * 70)

    print()
    print("Database:", DB_PATH)

    conn = duckdb.connect(DB_PATH)

    try:

        # --------------------------------------------------------
        # 1. Verify required tables
        # --------------------------------------------------------

        required_tables = [
            "aqi",
            "weather",
            "aqi_hourly",
            "weather_hourly",
        ]

        existing_tables = set(
            conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            )
            .fetchdf()["table_name"]
            .tolist()
        )

        missing = [
            table
            for table in required_tables
            if table not in existing_tables
        ]

        if missing:
            raise RuntimeError(
                "Missing required tables: "
                + ", ".join(missing)
            )

        # --------------------------------------------------------
        # 2. Determine completed hourly dates
        # --------------------------------------------------------

        hourly_dates = conn.execute(
            """
            SELECT DISTINCT
                CAST(timestamp AS DATE) AS date
            FROM aqi_hourly
            ORDER BY date
            """
        ).fetchdf()

        if hourly_dates.empty:
            print()
            print("No hourly AQI data found.")
            return

        min_date = hourly_dates["date"].min()
        max_date = hourly_dates["date"].max()

        print()
        print("Hourly AQI coverage:")
        print("  From:", min_date)
        print("  To  :", max_date)

        # --------------------------------------------------------
        # 3. Aggregate hourly AQI → daily AQI
        # --------------------------------------------------------

        daily_aqi = conn.execute(
            """
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
        ).fetchdf()

        # --------------------------------------------------------
        # 4. Aggregate hourly weather → daily weather
        # --------------------------------------------------------

        daily_weather = conn.execute(
            """
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
        ).fetchdf()

        print()
        print("Daily AQI aggregates    :", len(daily_aqi))
        print("Daily weather aggregates:", len(daily_weather))

        # --------------------------------------------------------
        # 5. Only use dates that have complete city coverage
        # --------------------------------------------------------

        complete_aqi_dates = conn.execute(
            """
            SELECT
                CAST(timestamp AS DATE) AS date
            FROM aqi_hourly
            GROUP BY CAST(timestamp AS DATE)
            HAVING COUNT(DISTINCT city_id) = 12
            ORDER BY date
            """
        ).fetchdf()

        complete_weather_dates = conn.execute(
            """
            SELECT
                CAST(timestamp AS DATE) AS date
            FROM weather_hourly
            GROUP BY CAST(timestamp AS DATE)
            HAVING COUNT(DISTINCT city_id) = 12
            ORDER BY date
            """
        ).fetchdf()

        complete_dates = set(
            complete_aqi_dates["date"].tolist()
        ).intersection(
            set(
                complete_weather_dates["date"].tolist()
            )
        )

        print()
        print(
            "Complete 12-city dates:",
            len(complete_dates),
        )

        if not complete_dates:
            print()
            print(
                "No dates have complete "
                "12-city coverage."
            )
            return

        # --------------------------------------------------------
        # 6. Create temporary daily tables
        # --------------------------------------------------------

        conn.register(
            "daily_aqi_df",
            daily_aqi,
        )

        conn.register(
            "daily_weather_df",
            daily_weather,
        )

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            daily_aqi_temp AS

            SELECT *
            FROM daily_aqi_df
            WHERE date IN (
                SELECT *
                FROM UNNEST(?)
            )
            """,
            [list(complete_dates)],
        )

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            daily_weather_temp AS

            SELECT *
            FROM daily_weather_df
            WHERE date IN (
                SELECT *
                FROM UNNEST(?)
            )
            """,
            [list(complete_dates)],
        )

        # --------------------------------------------------------
        # 7. Count rows before synchronization
        # --------------------------------------------------------

        before_aqi = conn.execute(
            """
            SELECT COUNT(*)
            FROM aqi
            """
        ).fetchone()[0]

        before_weather = conn.execute(
            """
            SELECT COUNT(*)
            FROM weather
            """
        ).fetchone()[0]

        # --------------------------------------------------------
        # 8. Upsert daily AQI
        # --------------------------------------------------------

        conn.execute(
            """
            DELETE FROM aqi
            WHERE date IN (
                SELECT DISTINCT date
                FROM daily_aqi_temp
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
                city_id,
                date,
                aqi,
                pm25,
                pm10,
                no2,
                so2,
                co,
                o3,
                'Open-Meteo-Hourly-Aggregated'

            FROM daily_aqi_temp
            """
        )

        # --------------------------------------------------------
        # 9. Upsert daily weather
        # --------------------------------------------------------

        conn.execute(
            """
            DELETE FROM weather
            WHERE date IN (
                SELECT DISTINCT date
                FROM daily_weather_temp
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
                city_id,
                date,
                temperature,
                humidity,
                precipitation,
                windspeed

            FROM daily_weather_temp
            """
        )

        # --------------------------------------------------------
        # 10. Verify final coverage
        # --------------------------------------------------------

        after_aqi = conn.execute(
            """
            SELECT COUNT(*)
            FROM aqi
            """
        ).fetchone()[0]

        after_weather = conn.execute(
            """
            SELECT COUNT(*)
            FROM weather
            """
        ).fetchone()[0]

        latest_aqi = conn.execute(
            """
            SELECT MAX(date)
            FROM aqi
            """
        ).fetchone()[0]

        latest_weather = conn.execute(
            """
            SELECT MAX(date)
            FROM weather
            """
        ).fetchone()[0]

        print()
        print("=" * 70)
        print("DATABASE SYNCHRONIZATION")
        print("=" * 70)

        print()
        print(
            "AQI rows before    :", before_aqi
        )
        print(
            "AQI rows after     :", after_aqi
        )
        print(
            "Weather rows before:", before_weather
        )
        print(
            "Weather rows after :", after_weather
        )

        print()
        print(
            "Latest AQI date    :",
            latest_aqi,
        )
        print(
            "Latest weather date:",
            latest_weather,
        )

        # --------------------------------------------------------
        # 11. Final city coverage validation
        # --------------------------------------------------------

        validation = conn.execute(
            """
            SELECT
                date,
                COUNT(DISTINCT city_id) AS cities,
                COUNT(*) AS rows
            FROM aqi
            WHERE date = (
                SELECT MAX(date)
                FROM aqi
            )
            GROUP BY date
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("LATEST AQI COVERAGE")
        print("=" * 70)

        print(
            validation.to_string(
                index=False
            )
        )

        validation_weather = conn.execute(
            """
            SELECT
                date,
                COUNT(DISTINCT city_id) AS cities,
                COUNT(*) AS rows
            FROM weather
            WHERE date = (
                SELECT MAX(date)
                FROM weather
            )
            GROUP BY date
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("LATEST WEATHER COVERAGE")
        print("=" * 70)

        print(
            validation_weather.to_string(
                index=False
            )
        )

        # --------------------------------------------------------
        # 12. Safety validation
        # --------------------------------------------------------

        if latest_aqi != latest_weather:
            raise RuntimeError(
                "AQI and weather latest dates "
                "do not match."
            )

        if validation.empty:
            raise RuntimeError(
                "AQI validation returned no rows."
            )

        if int(
            validation.iloc[0]["cities"]
        ) != 12:
            raise RuntimeError(
                "Latest AQI date does not contain "
                "all 12 cities."
            )

        if validation_weather.empty:
            raise RuntimeError(
                "Weather validation returned no rows."
            )

        if int(
            validation_weather.iloc[0]["cities"]
        ) != 12:
            raise RuntimeError(
                "Latest weather date does not "
                "contain all 12 cities."
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
        print(
            "Existing model was NOT retrained."
        )
        print(
            "Production model was NOT modified."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
