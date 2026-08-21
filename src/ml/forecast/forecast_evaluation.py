"""
PEARLSAQI PRODUCTION FORECAST EVALUATION

Production metrics:

Overall:
    MAE
    RMSE
    R2
    Bias
    Within +/-10
    Within +/-20

By horizon:
    24h
    48h
    72h

By city:
    MAE
    RMSE
    R2
    Bias
    Within +/-20

R2:

    R2 = 1 - SSE / SST

SSE:
    sum((actual - predicted)^2)

SST:
    sum((actual - mean_actual)^2)

R2 is NULL when:
    - fewer than 2 observations exist
    - actual AQI has zero variance

This module only evaluates forecasts whose
actual AQI is already available.
"""


from src.database.connection import get_connection


# ============================================================
# R2 HELPER SQL
# ============================================================

def calculate_r2_sql(
    partition_column=None
):
    """
    Returns SQL fragments for safe R2 calculation.

    We intentionally calculate the mean in a separate
    aggregation step.

    This avoids DuckDB's restriction against using
    window functions inside aggregate functions.
    """

    if partition_column is None:

        return """
        WITH evaluated AS (

            SELECT
                predicted_aqi,
                actual_aqi,
                error,
                absolute_error

            FROM forecast_predictions

            WHERE actual_aqi IS NOT NULL
        ),

        statistics AS (

            SELECT

                COUNT(*) AS n,

                AVG(actual_aqi) AS actual_mean,

                SUM(
                    (actual_aqi - predicted_aqi)
                    *
                    (actual_aqi - predicted_aqi)
                ) AS sse

            FROM evaluated
        ),

        variance AS (

            SELECT

                COUNT(*) AS n,

                SUM(
                    (actual_aqi - statistics.actual_mean)
                    *
                    (actual_aqi - statistics.actual_mean)
                ) AS sst

            FROM evaluated

            CROSS JOIN statistics
        )

        SELECT
            statistics.n,

            statistics.sse,

            variance.sst

        FROM statistics

        CROSS JOIN variance
        """

    return f"""
    WITH evaluated AS (

        SELECT
            {partition_column},
            predicted_aqi,
            actual_aqi,
            error,
            absolute_error

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL
    ),

    means AS (

        SELECT

            {partition_column},

            COUNT(*) AS n,

            AVG(actual_aqi) AS actual_mean

        FROM evaluated

        GROUP BY {partition_column}
    ),

    errors AS (

        SELECT

            e.{partition_column},

            SUM(
                (e.actual_aqi - e.predicted_aqi)
                *
                (e.actual_aqi - e.predicted_aqi)
            ) AS sse

        FROM evaluated e

        GROUP BY e.{partition_column}
    ),

    variance AS (

        SELECT

            e.{partition_column},

            SUM(
                (e.actual_aqi - m.actual_mean)
                *
                (e.actual_aqi - m.actual_mean)
            ) AS sst

        FROM evaluated e

        INNER JOIN means m

            ON e.{partition_column}
            = m.{partition_column}

        GROUP BY e.{partition_column}
    )

    SELECT

        m.{partition_column},

        m.n,

        errors.sse,

        variance.sst

    FROM means m

    INNER JOIN errors

        ON m.{partition_column}
        = errors.{partition_column}

    INNER JOIN variance

        ON m.{partition_column}
        = variance.{partition_column}
    """


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():

    print()
    print("=" * 70)
    print("PEARLSAQI FORECAST PRODUCTION EVALUATION")
    print("=" * 70)

    conn = get_connection()

    try:

        # ====================================================
        # OVERALL PERFORMANCE
        # ====================================================

        overall = conn.execute(
            """
            WITH evaluated AS (

                SELECT
                    predicted_aqi,
                    actual_aqi,
                    error,
                    absolute_error

                FROM forecast_predictions

                WHERE actual_aqi IS NOT NULL
            ),

            base_metrics AS (

                SELECT

                    COUNT(*) AS evaluated_forecasts,

                    AVG(absolute_error) AS mae,

                    SQRT(
                        AVG(error * error)
                    ) AS rmse,

                    AVG(error) AS bias,

                    AVG(
                        CASE
                            WHEN absolute_error <= 10
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) * 100 AS within10,

                    AVG(
                        CASE
                            WHEN absolute_error <= 20
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) * 100 AS within20,

                    AVG(actual_aqi) AS actual_mean

                FROM evaluated
            ),

            error_stats AS (

                SELECT

                    SUM(
                        (actual_aqi - predicted_aqi)
                        *
                        (actual_aqi - predicted_aqi)
                    ) AS sse

                FROM evaluated
            ),

            variance_stats AS (

                SELECT

                    SUM(
                        (actual_aqi - base_metrics.actual_mean)
                        *
                        (actual_aqi - base_metrics.actual_mean)
                    ) AS sst

                FROM evaluated

                CROSS JOIN base_metrics
            )

            SELECT

                base_metrics.evaluated_forecasts,

                base_metrics.mae,

                base_metrics.rmse,

                CASE
                    WHEN
                        base_metrics.evaluated_forecasts >= 2
                        AND variance_stats.sst > 0
                    THEN
                        1.0
                        -
                        (
                            error_stats.sse
                            /
                            variance_stats.sst
                        )
                    ELSE NULL
                END AS r2,

                base_metrics.bias,

                base_metrics.within10,

                base_metrics.within20

            FROM base_metrics

            CROSS JOIN error_stats

            CROSS JOIN variance_stats
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("OVERALL PRODUCTION PERFORMANCE")
        print("=" * 70)

        print(
            overall.to_string(
                index=False
            )
        )

                # ====================================================
        # PERFORMANCE BY HORIZON
        # ====================================================

        horizon = conn.execute(
            """
            WITH horizons AS (
    SELECT 1 AS horizon, '24h' AS horizon_label
    UNION ALL
    SELECT 2 AS horizon, '48h' AS horizon_label
    UNION ALL
    SELECT 3 AS horizon, '72h' AS horizon_label
),

            evaluated AS (

                SELECT
                    horizon,
                    predicted_aqi,
                    actual_aqi,
                    error,
                    absolute_error

                FROM forecast_predictions

                WHERE actual_aqi IS NOT NULL
            ),

            base_metrics AS (

                SELECT

                    horizon,

                    COUNT(*) AS n,

                    AVG(absolute_error) AS mae,

                    SQRT(
                        AVG(error * error)
                    ) AS rmse,

                    AVG(error) AS bias,

                    AVG(
                        CASE
                            WHEN absolute_error <= 20
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) * 100 AS within20,

                    AVG(actual_aqi) AS actual_mean

                FROM evaluated

                GROUP BY horizon
            ),

            error_stats AS (

                SELECT

                    horizon,

                    SUM(
                        (actual_aqi - predicted_aqi)
                        *
                        (actual_aqi - predicted_aqi)
                    ) AS sse

                FROM evaluated

                GROUP BY horizon
            ),

            variance_stats AS (

                SELECT

                    e.horizon,

                    SUM(
                        (e.actual_aqi - b.actual_mean)
                        *
                        (e.actual_aqi - b.actual_mean)
                    ) AS sst

                FROM evaluated e

                INNER JOIN base_metrics b

                    ON e.horizon = b.horizon

                GROUP BY e.horizon
            )

            SELECT

    h.horizon,

    h.horizon_label,

    COALESCE(
        b.n,
        0
    ) AS n,

                b.mae,

                b.rmse,

                CASE

                    WHEN
                        b.n >= 2
                        AND v.sst > 0

                    THEN

                        1.0
                        -
                        (
                            e.sse
                            /
                            v.sst
                        )

                    ELSE NULL

                END AS r2,

                b.bias,

                b.within20

            FROM horizons h

            LEFT JOIN base_metrics b

                ON h.horizon = b.horizon

            LEFT JOIN error_stats e

                ON h.horizon = e.horizon

            LEFT JOIN variance_stats v

                ON h.horizon = v.horizon

            ORDER BY h.horizon
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("PERFORMANCE BY HORIZON")
        print("=" * 70)

        print(
            horizon.to_string(
                index=False
            )
        )

        # ====================================================
        # PERFORMANCE BY CITY
        # ====================================================

        city = conn.execute(
            """
            WITH evaluated AS (

                SELECT
                    city_name,
                    predicted_aqi,
                    actual_aqi,
                    error,
                    absolute_error

                FROM forecast_predictions

                WHERE actual_aqi IS NOT NULL
            ),

            base_metrics AS (

                SELECT

                    city_name,

                    COUNT(*) AS n,

                    AVG(absolute_error) AS mae,

                    SQRT(
                        AVG(error * error)
                    ) AS rmse,

                    AVG(error) AS bias,

                    AVG(
                        CASE
                            WHEN absolute_error <= 20
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) * 100 AS within20,

                    AVG(actual_aqi) AS actual_mean

                FROM evaluated

                GROUP BY city_name
            ),

            error_stats AS (

                SELECT

                    city_name,

                    SUM(
                        (actual_aqi - predicted_aqi)
                        *
                        (actual_aqi - predicted_aqi)
                    ) AS sse

                FROM evaluated

                GROUP BY city_name
            ),

            variance_stats AS (

                SELECT

                    e.city_name,

                    SUM(
                        (e.actual_aqi - b.actual_mean)
                        *
                        (e.actual_aqi - b.actual_mean)
                    ) AS sst

                FROM evaluated e

                INNER JOIN base_metrics b

                    ON e.city_name = b.city_name

                GROUP BY e.city_name
            )

            SELECT

                b.city_name,

                b.n,

                b.mae,

                b.rmse,

                CASE
                    WHEN
                        b.n >= 2
                        AND v.sst > 0
                    THEN
                        1.0
                        -
                        (
                            e.sse
                            /
                            v.sst
                        )
                    ELSE NULL
                END AS r2,

                b.bias,

                b.within20

            FROM base_metrics b

            INNER JOIN error_stats e

                ON b.city_name = e.city_name

            INNER JOIN variance_stats v

                ON b.city_name = v.city_name

            ORDER BY b.mae
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("PERFORMANCE BY CITY")
        print("=" * 70)

        print(
            city.to_string(
                index=False
            )
        )

        # ====================================================
        # PENDING FORECASTS
        # ====================================================

        pending = conn.execute(
            """
            SELECT

                COUNT(*) AS pending_forecasts,

                MIN(forecast_date)
                    AS first_pending_date,

                MAX(forecast_date)
                    AS last_pending_date

            FROM forecast_predictions

            WHERE actual_aqi IS NULL
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("PENDING FORECASTS")
        print("=" * 70)

        print(
            pending.to_string(
                index=False
            )
        )

        # ====================================================
        # HORIZON COVERAGE
        # ====================================================

        coverage = conn.execute(
            """
            SELECT

                horizon,

                COUNT(*) AS total_forecasts,

                COUNT(actual_aqi) AS evaluated,

                COUNT(*) -
                COUNT(actual_aqi) AS pending

            FROM forecast_predictions

            GROUP BY horizon

            ORDER BY horizon
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("HORIZON COVERAGE")
        print("=" * 70)

        print(
            coverage.to_string(
                index=False
            )
        )

        # ====================================================
        # LATEST EVALUATED FORECASTS
        # ====================================================

        latest = conn.execute(
            """
            SELECT

                city_name,
                origin_date,
                forecast_date,
                horizon,
                predicted_aqi,
                actual_aqi,
                error,
                absolute_error

            FROM forecast_predictions

            WHERE actual_aqi IS NOT NULL

            ORDER BY
                forecast_date DESC,
                city_name

            LIMIT 20
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("LATEST EVALUATED FORECASTS")
        print("=" * 70)

        print(
            latest.to_string(
                index=False
            )
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        evaluated_count = int(
            overall.iloc[0][
                "evaluated_forecasts"
            ]
        )

        print()
        print("=" * 70)
        print("EVALUATION STATUS")
        print("=" * 70)

        print(
            f"Evaluated forecasts : "
            f"{evaluated_count}"
        )

        if evaluated_count < 36:

            print(
                "Status              : "
                "COLLECTING DATA"
            )

            print(
                "Recommendation      : "
                "Continue production collection"
            )

        else:

            print(
                "Status              : "
                "SUFFICIENT FOR MONITORING"
            )

        print()
        print("=" * 70)
        print("FORECAST EVALUATION COMPLETE")
        print("=" * 70)

    finally:

        conn.close()


if __name__ == "__main__":
    run_evaluation()