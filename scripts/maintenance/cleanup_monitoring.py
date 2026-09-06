from src.database.connection import get_connection

c = get_connection()

c.execute("""
DELETE FROM model_monitoring_history
WHERE run_date = CURRENT_DATE
""")

print("Today's monitoring snapshots removed.")

c.close()
