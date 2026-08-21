from src.database.connection import get_connection

c = get_connection()

print("=== FORECAST EVALUATION SUMMARY ===")

print(
    c.execute("""
        SELECT
            COUNT(*) AS evaluated_forecasts,
            AVG(absolute_error) AS mae,
            SQRT(AVG(error * error)) AS rmse,
            AVG(error) AS bias,
            AVG(
                CASE
                    WHEN absolute_error <= 20
                    THEN 1.0
                    ELSE 0.0
                END
            ) * 100 AS within20
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
    """).fetchdf().to_string(index=False)
)

print("\n=== BY CITY ===")

print(
    c.execute("""
        SELECT
            city_name,
            COUNT(*) AS n,
            AVG(absolute_error) AS mae,
            AVG(error) AS bias
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
        GROUP BY city_name
        ORDER BY mae
    """).fetchdf().to_string(index=False)
)

print("\n=== BY HORIZON ===")

print(
    c.execute("""
        SELECT
            horizon,
            COUNT(*) AS n,
            AVG(absolute_error) AS mae,
            AVG(error) AS bias
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
        GROUP BY horizon
        ORDER BY horizon
    """).fetchdf().to_string(index=False)
)

c.close()
