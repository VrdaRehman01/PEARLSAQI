from src.database.connection import get_connection

c = get_connection()

print("=== CURRENT AQI ===")

print(
    c.execute("""
        SELECT
            ci.city_name,
            MAX(a.date) AS latest_aqi_date
        FROM aqi a
        JOIN cities ci
            ON a.city_id = ci.city_id
        GROUP BY ci.city_name
        ORDER BY ci.city_name
    """).fetchdf().to_string(index=False)
)

print()
print("=== HISTORICAL WEATHER ===")

print(
    c.execute("""
        SELECT
            ci.city_name,
            MAX(w.date) AS latest_weather_date
        FROM weather w
        JOIN cities ci
            ON w.city_id = ci.city_id
        GROUP BY ci.city_name
        ORDER BY ci.city_name
    """).fetchdf().to_string(index=False)
)

print()
print("=== FUTURE WEATHER ===")

print(
    c.execute("""
        SELECT
            ci.city_name,
            MIN(wf.date) AS first_future_date,
            MAX(wf.date) AS last_future_date,
            COUNT(*) AS rows
        FROM weather_forecasts wf
        JOIN cities ci
            ON wf.city_id = ci.city_id
        GROUP BY ci.city_name
        ORDER BY ci.city_name
    """).fetchdf().to_string(index=False)
)

print()
print("=== FORECAST WEATHER ORIGINS ===")

print(
    c.execute("""
        SELECT
            forecast_origin,
            COUNT(*) AS rows,
            COUNT(DISTINCT city_id) AS cities,
            MIN(date) AS first_date,
            MAX(date) AS last_date
        FROM weather_forecasts
        GROUP BY forecast_origin
        ORDER BY forecast_origin
    """).fetchdf().to_string(index=False)
)

c.close()
