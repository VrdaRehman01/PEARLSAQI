from src.database.connection import get_connection

c = get_connection()

print(
    c.execute(
        "DESCRIBE model_monitoring_history"
    ).fetchdf().to_string(index=False)
)

c.close()
