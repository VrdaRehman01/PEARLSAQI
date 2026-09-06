from src.database.connection import get_connection

c = get_connection()

print("=== AUG 13 AQI + WEATHER ===")

df = c.execute("""
    SELECT
        c.city_name,
        a.date,
        a.aqi,
        w.date AS weather_date,
        w.temperature,
        w.humidity,
        w.precipitation,
        w.windspeed
    FROM aqi a
    JOIN cities c
        ON a.city_id = c.city_id
    LEFT JOIN weather w
        ON a.city_id = w.city_id
        AND a.date = w.date
    WHERE a.date = '2026-08-13'
    ORDER BY c.city_id
""").fetchdf()

print(df.to_string(index=False))

print()
print("Rows:", len(df))
print("Missing weather rows:", df["weather_date"].isna().sum())

print()
print("=== AUG 13 FEATURE ROWS ===")

features = c.execute("""
    SELECT *
    FROM features
    WHERE date = '2026-08-13'
    ORDER BY city_id
""").fetchdf()

print("Feature rows:", len(features))

weather_cols = [
    "temperature",
    "humidity",
    "precipitation",
    "windspeed"
]

available = [
    x for x in weather_cols
    if x in features.columns
]

print()
print("Weather columns found:", available)

if available:
    print(
        features[
            ["city_id", "city_name", "date"] + available
        ].to_string(index=False)
    )

    print()
    print("Missing values:")
    print(
        features[available]
        .isna()
        .sum()
        .to_string()
    )

c.close()
