from src.database.connection import get_connection

c = get_connection()

print("=== SEQUENCES ===")

try:
    print(
        c.execute(
            "SHOW SEQUENCES"
        ).fetchdf().to_string(index=False)
    )
except Exception as e:
    print("SHOW SEQUENCES failed:")
    print(e)

c.close()
