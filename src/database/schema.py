from src.database.connection import get_connection


def create_tables():
    conn = get_connection()

    # Cities table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cities (
        city_id INTEGER PRIMARY KEY,
        city_name VARCHAR,
        province VARCHAR,
        latitude DOUBLE,
        longitude DOUBLE
    );
    """)

    # AQI table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS aqi (
        id BIGINT,
        city_id INTEGER,
        date DATE,
        aqi DOUBLE,
        pm25 DOUBLE,
        pm10 DOUBLE,
        no2 DOUBLE,
        so2 DOUBLE,
        co DOUBLE,
        o3 DOUBLE
    );
    """)

    # Weather table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS aqi (
        id BIGINT,
        city_id INTEGER,
        date DATE,
        aqi DOUBLE,
        pm25 DOUBLE,
        pm10 DOUBLE,
        no2 DOUBLE,
        so2 DOUBLE,
        co DOUBLE,
        o3 DOUBLE,
        aqi_source VARCHAR
    );
    """)

    # Features table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS features (
        id BIGINT,
        city_id INTEGER,
        date DATE,
        feature_name VARCHAR,
        feature_value DOUBLE
    );
    """)

    # Predictions table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id BIGINT,
        city_id INTEGER,
        prediction_date DATE,
        predicted_aqi DOUBLE,
        model_name VARCHAR
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS weather (
        id BIGINT,
        city_id INTEGER,
        date DATE,
        temperature DOUBLE,
        humidity DOUBLE,
        precipitation DOUBLE,
        windspeed DOUBLE
    );
    """)

    conn.close()

    print("Database tables created successfully!")