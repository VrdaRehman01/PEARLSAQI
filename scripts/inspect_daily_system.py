from pathlib import Path

from src.database.connection import get_connection


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_production_xgboost"
    / "final_xgboost_model.json"
)

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "models"
    / "forecast"
    / "calibration_v9"
    / "calibration_parameters.json"
)

FORECAST_CSV = (
    PROJECT_ROOT
    / "models"
    / "forecast"
    / "v9"
    / "forecast_predictions.csv"
)

FORECAST_PARQUET = (
    PROJECT_ROOT
    / "models"
    / "forecast"
    / "v9"
    / "forecast_predictions.parquet"
)

FORECAST_METADATA = (
    PROJECT_ROOT
    / "models"
    / "forecast"
    / "v9"
    / "forecast_metadata.json"
)


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_database(conn):
    print_section("DATABASE HEALTH")

    tables = conn.execute(
        "SHOW TABLES"
    ).fetchdf()

    table_names = set(tables["name"].tolist())

    required_tables = {
        "aqi",
        "cities",
        "weather",
        "features",
        "predictions",
        "forecast_predictions",
    }

    print("Required tables:")

    for table in sorted(required_tables):
        status = "OK" if table in table_names else "MISSING"
        print(f"{table:<25} {status}")

    # -----------------------------------------------------
    # AQI coverage
    # -----------------------------------------------------

    aqi = conn.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT city_id) AS cities,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM aqi
        """
    ).fetchdf()

    print()
    print("AQI:")
    print(aqi.to_string(index=False))

    # -----------------------------------------------------
    # Weather coverage
    # -----------------------------------------------------

    weather = conn.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT city_id) AS cities,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM weather
        """
    ).fetchdf()

    print()
    print("Weather:")
    print(weather.to_string(index=False))

    # -----------------------------------------------------
    # AQI duplicates
    # -----------------------------------------------------

    aqi_duplicates = conn.execute(
        """
        SELECT
            city_id,
            date,
            COUNT(*) AS n
        FROM aqi
        GROUP BY city_id, date
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        """
    ).fetchdf()

    print()

    if aqi_duplicates.empty:
        print("No AQI duplicates found.")
    else:
        print("WARNING: AQI duplicates found:")
        print(aqi_duplicates.to_string(index=False))

    # -----------------------------------------------------
    # Weather duplicates
    # -----------------------------------------------------

    weather_duplicates = conn.execute(
        """
        SELECT
            city_id,
            date,
            COUNT(*) AS n
        FROM weather
        GROUP BY city_id, date
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        """
    ).fetchdf()

    print()

    if weather_duplicates.empty:
        print("No weather duplicates found.")
    else:
        print("WARNING: Weather duplicates found:")
        print(weather_duplicates.to_string(index=False))

    # -----------------------------------------------------
    # Forecast history
    # -----------------------------------------------------

    forecast_summary = conn.execute(
        """
        SELECT
            COUNT(*) AS total_forecasts,
            COUNT(actual_aqi) AS evaluated_forecasts,
            COUNT(*) - COUNT(actual_aqi) AS pending_forecasts,
            COUNT(DISTINCT city_id) AS cities,
            MIN(origin_date) AS first_origin_date,
            MAX(origin_date) AS latest_origin_date,
            MIN(forecast_date) AS first_forecast_date,
            MAX(forecast_date) AS latest_forecast_date
        FROM forecast_predictions
        """
    ).fetchdf()

    print()
    print("Forecast history:")
    print(forecast_summary.to_string(index=False))

    # -----------------------------------------------------
    # Forecast duplicates
    # -----------------------------------------------------

    forecast_duplicates = conn.execute(
        """
        SELECT
            city_id,
            origin_date,
            forecast_date,
            horizon,
            COUNT(*) AS n
        FROM forecast_predictions
        GROUP BY
            city_id,
            origin_date,
            forecast_date,
            horizon
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        """
    ).fetchdf()

    print()

    if forecast_duplicates.empty:
        print("No forecast duplicates found.")
    else:
        print("WARNING: Forecast duplicates found:")
        print(forecast_duplicates.to_string(index=False))

    # -----------------------------------------------------
    # Latest forecasts
    # -----------------------------------------------------

    latest_forecasts = conn.execute(
        """
        SELECT
            city_name,
            origin_date,
            forecast_date,
            horizon,
            predicted_aqi,
            actual_aqi
        FROM forecast_predictions
        ORDER BY origin_date DESC, city_name, horizon
        LIMIT 36
        """
    ).fetchdf()

    print()
    print("LATEST FORECASTS")

    if latest_forecasts.empty:
        print("No forecasts found.")
    else:
        print(
            latest_forecasts.to_string(
                index=False
            )
        )


def check_files():
    print_section("PRODUCTION FILE HEALTH")

    files = {
        "Production XGBoost": MODEL_FILE,
        "V9 Calibration": CALIBRATION_FILE,
        "Forecast CSV": FORECAST_CSV,
        "Forecast Parquet": FORECAST_PARQUET,
        "Forecast Metadata": FORECAST_METADATA,
    }

    all_ok = True

    for name, path in files.items():

        exists = path.exists()

        status = "OK" if exists else "MISSING"

        print(
            f"{name:<25} {status}"
        )

        if not exists:
            all_ok = False

        if exists:
            print(
                f"  {path}"
            )

    return all_ok


def main():

    print()
    print("=" * 70)
    print("PEARLSAQI DAILY SYSTEM HEALTH CHECK")
    print("=" * 70)

    conn = None

    try:

        conn = get_connection()

        check_database(conn)

        files_ok = check_files()

    except Exception as error:

        print()
        print("=" * 70)
        print("SYSTEM HEALTH CHECK FAILED")
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        raise

    finally:

        if conn is not None:
            conn.close()

    print()
    print("=" * 70)

    if files_ok:
        print("DAILY SYSTEM HEALTH CHECK PASSED")
    else:
        print("DAILY SYSTEM HEALTH CHECK FAILED")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()