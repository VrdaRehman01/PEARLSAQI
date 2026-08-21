from src.database.connection import get_connection

c = get_connection()

print("=== AQI LATEST DATE ===")
print(
    c.execute("""
        SELECT
            c.city_name,
            MAX(a.date) AS latest_aqi_date
        FROM aqi a
        JOIN cities c
            ON a.city_id = c.city_id
        GROUP BY c.city_name
        ORDER BY c.city_name
    """).fetchdf().to_string(index=False)
)

print()
print("=== HISTORICAL WEATHER LATEST DATE ===")
print(
    c.execute("""
        SELECT
            c.city_name,
            MAX(w.date) AS latest_weather_date
        FROM weather w
        JOIN cities c
            ON w.city_id = c.city_id
        GROUP BY c.city_name
        ORDER BY c.city_name
    """).fetchdf().to_string(index=False)
)

print()
print("=== FUTURE WEATHER ===")
print(
    c.execute("""
        SELECT
            city_name,
            MIN(date) AS first_future_date,
            MAX(date) AS last_future_date,
            COUNT(*) AS rows
        FROM weather_forecasts
        GROUP BY city_name
        ORDER BY city_name
    """).fetchdf().to_string(index=False)
)

c.close()
