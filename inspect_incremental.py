from src.database.connection import get_connection

c = get_connection()

print("=== AQI COVERAGE ===")
print(
    c.execute("""
        SELECT
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            COUNT(*) AS rows
        FROM aqi
    """).fetchdf().to_string(index=False)
)

print("\n=== WEATHER COVERAGE ===")
print(
    c.execute("""
        SELECT
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            COUNT(*) AS rows
        FROM weather
    """).fetchdf().to_string(index=False)
)

print("\n=== AQI DUPLICATES ===")
print(
    c.execute("""
        SELECT
            city_id,
            date,
            COUNT(*) AS n
        FROM aqi
        GROUP BY city_id, date
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT 20
    """).fetchdf().to_string(index=False)
)

print("\n=== WEATHER DUPLICATES ===")
print(
    c.execute("""
        SELECT
            city_id,
            date,
            COUNT(*) AS n
        FROM weather
        GROUP BY city_id, date
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT 20
    """).fetchdf().to_string(index=False)
)

c.close()
