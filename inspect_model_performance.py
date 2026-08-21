from src.database.connection import get_connection


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    conn = get_connection()

    print_section("PEARLSAQI MODEL PERFORMANCE MONITOR")

    # ---------------------------------------------------------
    # Overall performance
    # ---------------------------------------------------------

    print_section("OVERALL PERFORMANCE")

    overall = conn.execute("""
        SELECT
            COUNT(*) AS evaluated_forecasts,

            ROUND(AVG(absolute_error), 4) AS mae,

            ROUND(
                SQRT(AVG(error * error)),
                4
            ) AS rmse,

            ROUND(AVG(error), 4) AS bias,

            ROUND(
                AVG(
                    CASE
                        WHEN absolute_error <= 10
                        THEN 1.0
                        ELSE 0.0
                    END
                ) * 100,
                2
            ) AS within10,

            ROUND(
                AVG(
                    CASE
                        WHEN absolute_error <= 20
                        THEN 1.0
                        ELSE 0.0
                    END
                ) * 100,
                2
            ) AS within20

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL
    """).fetchdf()

    print(overall.to_string(index=False))

    # ---------------------------------------------------------
    # Performance by horizon
    # ---------------------------------------------------------

    print_section("PERFORMANCE BY HORIZON")

    horizon = conn.execute("""
        SELECT
            horizon,

            COUNT(*) AS n,

            ROUND(
                AVG(absolute_error),
                4
            ) AS mae,

            ROUND(
                SQRT(AVG(error * error)),
                4
            ) AS rmse,

            ROUND(
                AVG(error),
                4
            ) AS bias,

            ROUND(
                AVG(
                    CASE
                        WHEN absolute_error <= 20
                        THEN 1.0
                        ELSE 0.0
                    END
                ) * 100,
                2
            ) AS within20

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        GROUP BY horizon

        ORDER BY horizon
    """).fetchdf()

    if len(horizon) == 0:
        print("No evaluated forecasts yet.")
    else:
        print(horizon.to_string(index=False))

    # ---------------------------------------------------------
    # Performance by city
    # ---------------------------------------------------------

    print_section("PERFORMANCE BY CITY")

    city = conn.execute("""
        SELECT
            city_name,

            COUNT(*) AS n,

            ROUND(
                AVG(absolute_error),
                4
            ) AS mae,

            ROUND(
                SQRT(AVG(error * error)),
                4
            ) AS rmse,

            ROUND(
                AVG(error),
                4
            ) AS bias,

            ROUND(
                AVG(
                    CASE
                        WHEN absolute_error <= 20
                        THEN 1.0
                        ELSE 0.0
                    END
                ) * 100,
                2
            ) AS within20

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        GROUP BY city_name

        ORDER BY mae
    """).fetchdf()

    if len(city) == 0:
        print("No evaluated forecasts yet.")
    else:
        print(city.to_string(index=False))

    # ---------------------------------------------------------
    # Forecast status
    # ---------------------------------------------------------

    print_section("FORECAST STATUS")

    status = conn.execute("""
        SELECT
            COUNT(*) AS total_forecasts,

            COUNT(actual_aqi)
                AS evaluated_forecasts,

            COUNT(*) - COUNT(actual_aqi)
                AS pending_forecasts,

            MIN(origin_date)
                AS first_origin_date,

            MAX(origin_date)
                AS latest_origin_date,

            MIN(forecast_date)
                AS first_forecast_date,

            MAX(forecast_date)
                AS latest_forecast_date

        FROM forecast_predictions
    """).fetchdf()

    print(status.to_string(index=False))

    # ---------------------------------------------------------
    # Latest evaluated forecasts
    # ---------------------------------------------------------

    print_section("LATEST EVALUATED FORECASTS")

    latest = conn.execute("""
        SELECT
            city_name,
            origin_date,
            forecast_date,
            horizon,
            predicted_aqi,
            actual_aqi,
            error,
            absolute_error

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        ORDER BY
            origin_date DESC,
            city_name,
            horizon

        LIMIT 36
    """).fetchdf()

    if len(latest) == 0:
        print("No evaluated forecasts yet.")
    else:
        print(latest.to_string(index=False))

    # ---------------------------------------------------------
    # Worst recent errors
    # ---------------------------------------------------------

    print_section("LARGEST RECENT FORECAST ERRORS")

    worst = conn.execute("""
        SELECT
            city_name,
            origin_date,
            forecast_date,
            horizon,
            predicted_aqi,
            actual_aqi,
            error,
            absolute_error

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        ORDER BY absolute_error DESC

        LIMIT 10
    """).fetchdf()

    if len(worst) == 0:
        print("No evaluated forecasts yet.")
    else:
        print(worst.to_string(index=False))

    # ---------------------------------------------------------
    # Recent origin performance
    # ---------------------------------------------------------

    print_section("PERFORMANCE BY FORECAST ORIGIN")

    origins = conn.execute("""
        SELECT
            origin_date,

            COUNT(*) AS n,

            ROUND(
                AVG(absolute_error),
                4
            ) AS mae,

            ROUND(
                SQRT(AVG(error * error)),
                4
            ) AS rmse,

            ROUND(
                AVG(error),
                4
            ) AS bias

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        GROUP BY origin_date

        ORDER BY origin_date DESC
    """).fetchdf()

    if len(origins) == 0:
        print("No evaluated forecasts yet.")
    else:
        print(origins.to_string(index=False))

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print_section("MODEL MONITORING STATUS")

    evaluated_count = int(
        overall.iloc[0]["evaluated_forecasts"]
    )

    if evaluated_count == 0:

        print("STATUS: WAITING FOR EVALUATED FORECASTS")

    elif evaluated_count < 36:

        print(
            f"STATUS: COLLECTING PRODUCTION DATA "
            f"({evaluated_count} evaluated forecasts)"
        )

    else:

        mae = float(overall.iloc[0]["mae"])
        within20 = float(overall.iloc[0]["within20"])

        print("STATUS: SUFFICIENT DATA FOR INITIAL REVIEW")

        print(f"Current MAE     : {mae:.4f}")
        print(f"Within ±20 AQI : {within20:.2f}%")

    print_section("MODEL PERFORMANCE MONITOR COMPLETE")

    conn.close()


if __name__ == "__main__":
    main()