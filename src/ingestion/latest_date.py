from src.database.connection import get_connection


def main():

    conn = get_connection()

    print("=" * 60)
    print("CURRENT DATASET COVERAGE")
    print("=" * 60)

    aqi = conn.execute("""
        SELECT
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            COUNT(*) AS records
        FROM aqi
    """).fetchdf()

    weather = conn.execute("""
        SELECT
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            COUNT(*) AS records
        FROM weather
    """).fetchdf()

    print("\nAQI:")
    print(aqi.to_string(index=False))

    print("\nWeather:")
    print(weather.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()