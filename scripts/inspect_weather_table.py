from src.database.connection import get_connection

conn = get_connection()

print("=== WEATHER TABLE SCHEMA ===")
print(
    conn.execute(
        "DESCRIBE weather"
    ).fetchdf().to_string(index=False)
)

print()
print("=== WEATHER SAMPLE ===")
print(
    conn.execute("""
        SELECT *
        FROM weather
        ORDER BY date DESC
        LIMIT 5
    """).fetchdf().to_string(index=False)
)

conn.close()
