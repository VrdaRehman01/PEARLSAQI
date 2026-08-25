from src.database.connection import get_connection


def get_latest_predictions():
    """
    Return the latest V9 H1 predictions for all cities.

    Current AQI comes from the latest actual AQI record.
    Prediction comes from the latest V9 forecast.
    """

    conn = get_connection()

    try:
        query = """
            WITH latest_aqi AS (
                SELECT
                    city_id,
                    aqi,
                    date,
                    ROW_NUMBER() OVER (
                        PARTITION BY city_id
                        ORDER BY date DESC
                    ) AS rn
                FROM aqi
                WHERE aqi IS NOT NULL
            ),

            latest_forecast_origin AS (
                SELECT
                    MAX(origin_date) AS origin_date
                FROM forecast_predictions
            )

            SELECT
                f.city_id,
                f.city_name,
                a.date AS date,
                f.origin_date AS prediction_date,
                a.aqi AS aqi,
                f.predicted_aqi AS prediction
            FROM forecast_predictions f

            INNER JOIN latest_forecast_origin lfo
                ON f.origin_date = lfo.origin_date

            INNER JOIN latest_aqi a
                ON f.city_id = a.city_id
                AND a.rn = 1

            WHERE f.horizon = 1

            ORDER BY f.city_name
        """

        return conn.execute(query).fetchdf()

    finally:
        conn.close()

def get_all_forecasts():
    """
    Return the latest V9 24h / 48h / 72h forecasts
    for all cities.
    """

    conn = get_connection()

    try:
        query = """
            WITH latest_forecast_origin AS (
                SELECT
                    MAX(origin_date) AS origin_date
                FROM forecast_predictions
            )

            SELECT
                f.city_id,
                f.city_name,
                f.origin_date,
                f.forecast_date,
                f.horizon,
                f.predicted_aqi,
                f.model_name
            FROM forecast_predictions f

            INNER JOIN latest_forecast_origin lfo
                ON f.origin_date = lfo.origin_date

            WHERE f.horizon IN (1, 2, 3)

            ORDER BY
                f.city_name,
                f.horizon
        """

        return conn.execute(query).fetchdf()

    finally:
        conn.close()

def get_forecast_performance():
    """
    Return V9 forecast accuracy metrics for 24h / 48h / 72h horizons.

    Only forecasts with an available actual AQI are evaluated.
    """

    conn = get_connection()

    try:
        query = """
            SELECT
                horizon,
                COUNT(actual_aqi) AS evaluated_rows,
                ROUND(AVG(absolute_error), 2) AS mae,
                ROUND(SQRT(AVG(error * error)), 2) AS rmse
            FROM forecast_predictions
            WHERE actual_aqi IS NOT NULL
              AND horizon IN (1, 2, 3)
            GROUP BY horizon
            ORDER BY horizon
        """

        return conn.execute(query).fetchdf()

    finally:
        conn.close()

