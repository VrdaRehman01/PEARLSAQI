from src.database.connection import get_connection

c = get_connection()

print("=== TABLES ===")
print(c.execute("SHOW TABLES").fetchdf().to_string(index=False))

print("\n=== AQI ===")
print(c.execute("DESCRIBE aqi").fetchdf().to_string(index=False))

print("\n=== WEATHER ===")
print(c.execute("DESCRIBE weather").fetchdf().to_string(index=False))

print("\n=== PREDICTIONS ===")
print(c.execute("DESCRIBE predictions").fetchdf().to_string(index=False))

print("\n=== WEATHER COVERAGE ===")
print(c.execute("""
SELECT MIN(date) AS start_date,
       MAX(date) AS end_date,
       COUNT(*) AS rows
FROM weather
""").fetchdf().to_string(index=False))

print("\n=== AQI DUPLICATES ===")
print(c.execute("""
SELECT city_id, date, COUNT(*) AS n
FROM aqi
GROUP BY city_id, date
HAVING COUNT(*) > 1
ORDER BY n DESC
LIMIT 20
""").fetchdf().to_string(index=False))

c.close()
