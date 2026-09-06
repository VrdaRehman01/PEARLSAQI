from src.database.connection import get_connection

c = get_connection()

print(
    c.execute(
        "DESCRIBE forecast_predictions"
    ).fetchdf().to_string(index=False)
)

c.close()
