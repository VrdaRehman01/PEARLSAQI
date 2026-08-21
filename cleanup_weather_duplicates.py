from src.database.connection import get_connection


def cleanup_weather_duplicates():

    conn = get_connection()

    try:

        print()
        print("=" * 70)
        print("WEATHER FORECAST DATABASE CLEANUP")
        print("=" * 70)

        # ------------------------------------------------------
        # 1. CHECK CURRENT STATE
        # ------------------------------------------------------

        before = conn.execute(
            """
            SELECT COUNT(*)
            FROM weather_forecasts
            """
        ).fetchone()[0]

        print(f"Rows before cleanup : {before}")

        # ------------------------------------------------------
        # 2. CREATE CLEAN TEMPORARY TABLE
        #
        # For every city + date:
        #
        # newest forecast_origin wins
        # newest created_at wins if tied
        # highest id wins if still tied
        # ------------------------------------------------------

        print()
        print("Creating deduplicated forecast table...")

        conn.execute(
            """
            DROP TABLE IF EXISTS weather_forecasts_clean
            """
        )

        conn.execute(
            """
            CREATE TABLE weather_forecasts_clean AS

            SELECT
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

            FROM weather_forecasts

            QUALIFY
                ROW_NUMBER() OVER (
                    PARTITION BY
                        city_id,
                        date

                    ORDER BY
                        forecast_origin DESC,
                        created_at DESC,
                        id DESC
                ) = 1
            """
        )

        clean_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM weather_forecasts_clean
            """
        ).fetchone()[0]

        print(
            f"Clean rows created    : {clean_count}"
        )

        # ------------------------------------------------------
        # 3. SHOW WHAT WILL BE REMOVED
        # ------------------------------------------------------

        removed_count = before - clean_count

        print(
            f"Duplicate rows removed: {removed_count}"
        )

        # ------------------------------------------------------
        # 4. REPLACE ORIGINAL CONTENT
        # ------------------------------------------------------

        print()
        print("Replacing weather_forecasts contents...")

        conn.execute(
            """
            BEGIN TRANSACTION
            """
        )

        try:

            conn.execute(
                """
                DELETE FROM weather_forecasts
                """
            )

            conn.execute(
                """
                INSERT INTO weather_forecasts
                (
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

                SELECT
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

                FROM weather_forecasts_clean
                """
            )

            conn.execute(
                """
                COMMIT
                """
            )

        except Exception:

            conn.execute(
                """
                ROLLBACK
                """
            )

            raise

        # ------------------------------------------------------
        # 5. REMOVE TEMPORARY TABLE
        # ------------------------------------------------------

        conn.execute(
            """
            DROP TABLE weather_forecasts_clean
            """
        )

        # ------------------------------------------------------
        # 6. VERIFY FINAL ROW COUNT
        # ------------------------------------------------------

        after = conn.execute(
            """
            SELECT COUNT(*)
            FROM weather_forecasts
            """
        ).fetchone()[0]

        print()
        print("=" * 70)
        print("CLEANUP RESULT")
        print("=" * 70)

        print(
            f"Rows before cleanup : {before}"
        )

        print(
            f"Rows after cleanup  : {after}"
        )

        print(
            f"Rows removed        : {before - after}"
        )

        if after != clean_count:

            raise RuntimeError(
                "Final row count does not match "
                "deduplicated row count."
            )

        # ------------------------------------------------------
        # 7. DUPLICATE CHECK
        # ------------------------------------------------------

        duplicates = conn.execute(
            """
            SELECT
                city_id,
                city_name,
                date,
                COUNT(*) AS rows

            FROM weather_forecasts

            GROUP BY
                city_id,
                city_name,
                date

            HAVING COUNT(*) > 1

            ORDER BY
                city_id,
                date
            """
        ).fetchdf()

        print()

        if duplicates.empty:

            print(
                "Duplicate city/date rows : NONE"
            )

        else:

            print(
                "ERROR: Duplicate city/date rows remain:"
            )

            print(
                duplicates.to_string(
                    index=False
                )
            )

            raise RuntimeError(
                "Duplicate city/date rows remain."
            )

        # ------------------------------------------------------
        # 8. VERIFY FORECAST TIMELINE
        # ------------------------------------------------------

        timeline = conn.execute(
            """
            SELECT
                city_id,
                city_name,
                MIN(date) AS first_date,
                MAX(date) AS last_date,
                COUNT(*) AS rows,
                COUNT(DISTINCT date) AS unique_dates

            FROM weather_forecasts

            GROUP BY
                city_id,
                city_name

            ORDER BY
                city_id
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("FINAL WEATHER FORECAST TIMELINE")
        print("=" * 70)

        print(
            timeline.to_string(
                index=False
            )
        )

        # ------------------------------------------------------
        # 9. FINAL UNIQUENESS ASSERTION
        # ------------------------------------------------------

        bad = timeline[
            timeline["rows"]
            != timeline["unique_dates"]
        ]

        if not bad.empty:

            raise RuntimeError(
                "City/date uniqueness validation failed."
            )

        print()
        print("=" * 70)
        print("WEATHER FORECAST CLEANUP PASSED")
        print("=" * 70)

    finally:

        conn.close()


if __name__ == "__main__":

    cleanup_weather_duplicates()