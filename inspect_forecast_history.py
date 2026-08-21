from src.database.connection import get_connection

c = get_connection()

print("=" * 70)
print("FORECAST HISTORY SUMMARY")
print("=" * 70)

print(
    c.execute("""
        SELECT
            COUNT(*) AS total_forecasts,
            COUNT(actual_aqi) AS evaluated_forecasts,
            COUNT(*) - COUNT(actual_aqi)
                AS pending_forecasts,
            COUNT(DISTINCT city_id)
                AS cities,
            COUNT(DISTINCT origin_date)
                AS origin_dates
        FROM forecast_predictions
    """)
    .fetchdf()
    .to_string(index=False)
)

print()
print("=" * 70)
print("LATEST FORECASTS")
print("=" * 70)

print(
    c.execute("""
        SELECT
            city_name,
            origin_date,
            forecast_date,
            horizon,
            ROUND(predicted_aqi, 2)
                AS predicted_aqi,
            actual_aqi,
            ROUND(error, 2)
                AS error,
            ROUND(absolute_error, 2)
                AS absolute_error
        FROM forecast_predictions
        ORDER BY
            origin_date DESC,
            city_id,
            horizon
        LIMIT 36
    """)
    .fetchdf()
    .to_string(index=False)
)

c.close()