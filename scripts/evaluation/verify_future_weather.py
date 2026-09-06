from src.database.connection import get_connection

c = get_connection()

df = c.execute("""
    SELECT
        c.city_name,
        w.city_id,
        w.date,
        w.temperature,
        w.humidity,
        w.precipitation,
        w.windspeed
    FROM weather w
    JOIN cities c
        ON w.city_id = c.city_id
    WHERE w.date BETWEEN '2026-08-14' AND '2026-08-16'
    ORDER BY w.city_id, w.date
""").fetchdf()

print(df.to_string(index=False))

print()
print("Rows:", len(df))
print("Cities:", df["city_id"].nunique())
print("Dates:", df["date"].nunique())

c.close()
