from src.database.connection import get_connection

c = get_connection()

print(
    c.execute("SHOW TABLES").fetchdf().to_string(index=False)
)

print("\n=== FUTURE WEATHER ===")

df = c.execute("""
    SELECT
        city_name,
        city_id,
        date,
        temperature,
        humidity,
        precipitation,
        windspeed,
        forecast_origin
    FROM weather_forecasts
    ORDER BY city_id, date
""").fetchdf()

print(df.to_string(index=False))

print("\nRows:", len(df))
print("Cities:", df["city_id"].nunique())
print("Dates:", df["date"].nunique())

c.close()
