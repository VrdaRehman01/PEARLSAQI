from src.database.connection import get_connection

c = get_connection()

print("=== FORECAST HISTORY ===")

print(
    c.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(actual_aqi) AS evaluated,
            COUNT(*) - COUNT(actual_aqi) AS pending
        FROM forecast_predictions
    """).fetchdf().to_string(index=False)
)

print("\n=== EVALUATED FORECASTS ===")

print(
    c.execute("""
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
        ORDER BY origin_date, city_name, horizon
    """).fetchdf().to_string(index=False)
)

print("\n=== EVALUATION BY HORIZON ===")

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
