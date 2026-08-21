from src.database.connection import get_connection

c = get_connection()

print("TABLES:")
print(
    c.execute("SHOW TABLES").fetchdf().to_string(index=False)
)

print("\nWEATHER-RELATED TABLES:")
tables = c.execute("SHOW TABLES").fetchdf()

for col in tables.columns:
    for name in tables[col].dropna().astype(str):
        if "weather" in name.lower():
            print(f"\n--- {name} ---")
            print(
                c.execute(
                    f'DESCRIBE "{name}"'
                ).fetchdf().to_string(index=False)
            )

c.close()
