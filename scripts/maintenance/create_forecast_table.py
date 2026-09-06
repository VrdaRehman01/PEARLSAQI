from src.database.connection import get_connection

c = get_connection()

c.execute("""
CREATE TABLE IF NOT EXISTS forecast_predictions (
    id BIGINT,
    city_id INTEGER,
    city_name VARCHAR,
    origin_date DATE,
    forecast_date DATE,
    horizon INTEGER,
    predicted_aqi DOUBLE,
    actual_aqi DOUBLE,
    error DOUBLE,
    absolute_error DOUBLE,
    model_name VARCHAR,
    created_at TIMESTAMP
)
""")

c.close()

print("forecast_predictions table created successfully.")
