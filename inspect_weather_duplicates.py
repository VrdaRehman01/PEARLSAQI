from src.database.connection import get_connection

c = get_connection()

print("=== WEATHER FORECASTS SCHEMA ===")

print(
    c.execute(
        "DESCRIBE weather_forecasts"
    ).fetchdf().to_string(index=False)
)

print()
print("=== DUPLICATE CITY + DATE CHECK ===")

print(
    c.execute("""
        SELECT
            city_id,
            date,
            COUNT(*) AS rows,
            MIN(forecast_origin) AS oldest_origin,
            MAX(forecast_origin) AS latest_origin
        FROM weather_forecasts
        GROUP BY city_id, date
        HAVING COUNT(*) > 1
        ORDER BY city_id, date
    """).fetchdf().to_string(index=False)
)

c.close()
