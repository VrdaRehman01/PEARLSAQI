from src.database.connection import get_connection


def main():

    conn = get_connection()

    try:

        print("=" * 70)
        print("AQI / HISTORICAL WEATHER GAP AUDIT")
        print("=" * 70)

        df = conn.execute(
            """
            WITH latest_aqi AS (

                SELECT
                    city_id,
                    MAX(date) AS latest_aqi_date
                FROM aqi
                GROUP BY
                    city_id
            ),

            latest_weather AS (

                SELECT
                    city_id,
                    MAX(date) AS latest_weather_date
                FROM weather
                GROUP BY
                    city_id
            )

            SELECT

                c.city_id,

                c.city_name,

                a.latest_aqi_date,

                w.latest_weather_date,

                CASE
                    WHEN w.latest_weather_date IS NULL
                    THEN NULL

                    ELSE DATE_DIFF(
                        'day',
                        w.latest_weather_date,
                        a.latest_aqi_date
                    )
                END AS weather_lag_days

            FROM cities c

            LEFT JOIN latest_aqi a
                ON c.city_id = a.city_id

            LEFT JOIN latest_weather w
                ON c.city_id = w.city_id

            ORDER BY
                c.city_id
            """
        ).fetchdf()

        print()
        print("=== CITY-LEVEL ALIGNMENT ===")
        print()

        print(
            df.to_string(
                index=False
            )
        )

        # ------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(
            f"Cities checked : {len(df)}"
        )

        missing_aqi = df[
            df["latest_aqi_date"].isna()
        ]

        missing_weather = df[
            df["latest_weather_date"].isna()
        ]

        print(
            f"Cities missing AQI : "
            f"{len(missing_aqi)}"
        )

        print(
            f"Cities missing historical weather : "
            f"{len(missing_weather)}"
        )

        valid_gap = df[
            df["weather_lag_days"].notna()
        ]

        if not valid_gap.empty:

            print(
                f"Maximum weather lag : "
                f"{valid_gap['weather_lag_days'].max()} days"
            )

            print(
                f"Minimum weather lag : "
                f"{valid_gap['weather_lag_days'].min()} days"
            )

        # ------------------------------------------------------
        # AQI NEWER THAN WEATHER
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("AQI NEWER THAN HISTORICAL WEATHER")
        print("=" * 70)

        gap = df[
            df["weather_lag_days"] > 0
        ]

        if gap.empty:

            print("NONE")

        else:

            print(
                gap.to_string(
                    index=False
                )
            )

        # ------------------------------------------------------
        # EXACT LATEST DATES
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("LATEST DATES")
        print("=" * 70)

        latest_dates = conn.execute(
            """
            SELECT

                c.city_id,

                c.city_name,

                (
                    SELECT MAX(a.date)
                    FROM aqi a
                    WHERE a.city_id = c.city_id
                ) AS latest_aqi_date,

                (
                    SELECT MAX(w.date)
                    FROM weather w
                    WHERE w.city_id = c.city_id
                ) AS latest_weather_date

            FROM cities c

            ORDER BY
                c.city_id
            """
        ).fetchdf()

        print(
            latest_dates.to_string(
                index=False
            )
        )

    finally:

        conn.close()


if __name__ == "__main__":

    main()