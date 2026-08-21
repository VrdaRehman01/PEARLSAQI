from datetime import date, datetime, timedelta

from src.database.connection import get_connection


# ============================================================
# PEARLSAQI PRODUCTION MODEL MONITOR
# ============================================================

MIN_PRODUCTION_FORECASTS = 36

DRIFT_MEAN_THRESHOLD = 20.0
DRIFT_STD_THRESHOLD = 30.0
RECENT_MAE_WARNING = 20.0

MODEL_STATUS_COLLECTING = "COLLECTING DATA"
MODEL_STATUS_STABLE = "STABLE"
MODEL_STATUS_REVIEW = "REVIEW"


# ============================================================
# DISPLAY
# ============================================================

def header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SAFE VALUE
# ============================================================

def safe_float(value):

    if value is None:
        return None

    try:

        value = float(value)

        if value != value:
            return None

        return value

    except (TypeError, ValueError):

        return None


def fmt(value, digits=4):

    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# OVERALL FORECAST PERFORMANCE
# ============================================================

def get_overall_performance(c):

    return c.execute("""
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
            ) * 100 AS within20

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL
    """).fetchdf()


# ============================================================
# PERFORMANCE BY HORIZON
# ============================================================

def get_horizon_performance(c):

    return c.execute("""
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
            ) * 100 AS within20

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        GROUP BY horizon

        ORDER BY horizon
    """).fetchdf()


# ============================================================
# PERFORMANCE BY CITY
# ============================================================

def get_city_performance(c):

    return c.execute("""
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
            ) * 100 AS within20

        FROM forecast_predictions

        WHERE actual_aqi IS NOT NULL

        GROUP BY city_name

        ORDER BY mae DESC
    """).fetchdf()


# ============================================================
# FORECAST STATUS
# ============================================================

def get_forecast_status(c):

    return c.execute("""
        SELECT

            COUNT(*) AS total_forecasts,

            COUNT(actual_aqi) AS evaluated_forecasts,

            COUNT(*) - COUNT(actual_aqi)
                AS pending_forecasts,

            COUNT(DISTINCT city_id)
                AS cities,

            MIN(origin_date)
                AS first_origin_date,

            MAX(origin_date)
                AS latest_origin_date,

            MIN(forecast_date)
                AS first_forecast_date,

            MAX(forecast_date)
                AS latest_forecast_date

        FROM forecast_predictions
    """).fetchdf()


# ============================================================
# NAIVE BASELINE
#
# Baseline = previous day's actual AQI.
# ============================================================

def get_baseline_comparison(c):

    return c.execute("""
        WITH evaluated AS (

            SELECT
                fp.city_id,
                fp.city_name,
                fp.origin_date,
                fp.forecast_date,
                fp.horizon,
                fp.predicted_aqi,
                fp.actual_aqi,

                a_prev.aqi AS baseline_aqi

            FROM forecast_predictions fp

            LEFT JOIN aqi a_prev
                ON a_prev.city_id = fp.city_id
                AND a_prev.date = fp.forecast_date - INTERVAL 1 DAY

            WHERE fp.actual_aqi IS NOT NULL
        ),

        scored AS (

            SELECT

                predicted_aqi,
                actual_aqi,
                baseline_aqi,

                ABS(
                    predicted_aqi - actual_aqi
                ) AS model_error,

                ABS(
                    baseline_aqi - actual_aqi
                ) AS baseline_error

            FROM evaluated

            WHERE baseline_aqi IS NOT NULL
        )

        SELECT

            COUNT(*) AS n,

            AVG(model_error) AS model_mae,

            SQRT(
                AVG(model_error * model_error)
            ) AS model_rmse,

            AVG(baseline_error) AS baseline_mae,

            SQRT(
                AVG(baseline_error * baseline_error)
            ) AS baseline_rmse,

            CASE
                WHEN AVG(baseline_error) = 0
                THEN NULL

                ELSE
                    (
                        (
                            AVG(baseline_error)
                            -
                            AVG(model_error)
                        )
                        /
                            AVG(baseline_error)
                    ) * 100
            END AS improvement_pct

        FROM scored
    """).fetchdf()
# ============================================================
# RECENT AQI DISTRIBUTION
# ============================================================

def get_aqi_distribution(c, days=30):

    latest = c.execute("""
        SELECT MAX(date)
        FROM aqi
    """).fetchone()[0]

    if latest is None:
        return None

    cutoff = latest - timedelta(days=days)

    return c.execute("""
        SELECT

            AVG(aqi) AS mean_aqi,

            STDDEV(aqi) AS std_aqi

        FROM aqi

        WHERE date >= ?
    """, [cutoff]).fetchone()


# ============================================================
# HISTORICAL AQI DISTRIBUTION
# ============================================================

def get_historical_aqi_distribution(c, days=30):

    latest = c.execute("""
        SELECT MAX(date)
        FROM aqi
    """).fetchone()[0]

    if latest is None:
        return None

    cutoff = latest - timedelta(days=days)

    return c.execute("""
        SELECT

            AVG(aqi) AS mean_aqi,

            STDDEV(aqi) AS std_aqi

        FROM aqi

        WHERE date < ?
    """, [cutoff]).fetchone()


# ============================================================
# PERCENT CHANGE
# ============================================================

def percentage_change(old, new):

    old = safe_float(old)
    new = safe_float(new)

    if old is None or new is None:
        return None

    if old == 0:
        return None

    return (
        (new - old)
        / abs(old)
    ) * 100.0


# ============================================================
# HORIZON MAE
# ============================================================

def horizon_mae(horizon_df, horizon):

    rows = horizon_df[
        horizon_df["horizon"] == horizon
    ]

    if rows.empty:
        return None

    return safe_float(
        rows.iloc[0]["mae"]
    )


# ============================================================
# GENERATE STATUS
# ============================================================

def determine_status(
    evaluated,
    warnings
):

    if evaluated < MIN_PRODUCTION_FORECASTS:

        return (
            MODEL_STATUS_COLLECTING,
            (
                f"Only {evaluated} evaluated production "
                f"forecasts. Need at least "
                f"{MIN_PRODUCTION_FORECASTS}."
            ),
            "Continue collecting production data."
        )

    if warnings:

        return (
            MODEL_STATUS_REVIEW,
            (
                "Production monitoring detected "
                "one or more warning signals."
            ),
            (
                "Investigate performance and drift "
                "before considering retraining."
            )
        )

    return (
        MODEL_STATUS_STABLE,
        "No configured production warning thresholds exceeded.",
        "Continue normal production monitoring."
    )


# ============================================================
# SAVE MONITORING SNAPSHOT
# ============================================================

def save_monitoring_snapshot(
    c,
    evaluated,
    overall,
    mae_24h,
    mae_48h,
    mae_72h,
    baseline,
    recent_mean,
    historical_mean,
    mean_change,
    recent_std,
    historical_std,
    std_change,
    drift_status,
    model_status,
    recommendation
):

    next_id = c.execute("""
        SELECT
            COALESCE(MAX(id), 0) + 1
        FROM model_monitoring_history
    """).fetchone()[0]

    c.execute("""
    DELETE FROM model_monitoring_history
    WHERE run_date = CURRENT_DATE
    """)

    c.execute("""
        INSERT INTO model_monitoring_history (

            id,
            run_date,

            evaluated_forecasts,

            overall_mae,
            overall_rmse,
            overall_bias,
            overall_within10,
            overall_within20,

            mae_24h,
            mae_48h,
            mae_72h,

            baseline_mae,
            baseline_rmse,
            baseline_improvement_pct,

            recent_mean_aqi,
            historical_mean_aqi,
            mean_aqi_change_pct,

            recent_std_aqi,
            historical_std_aqi,
            std_aqi_change_pct,

            drift_status,
            model_status,
            recommendation,

            created_at
        )

        VALUES (

            ?,
            ?,

            ?,

            ?,
            ?,
            ?,
            ?,
            ?,

            ?,
            ?,
            ?,

            ?,
            ?,
            ?,

            ?,
            ?,
            ?,

            ?,
            ?,
            ?,

            ?,
            ?,
            ?,

            ?
        )
    """, [

        next_id,
        date.today(),

        evaluated,

        safe_float(overall.iloc[0]["mae"]),
        safe_float(overall.iloc[0]["rmse"]),
        safe_float(overall.iloc[0]["bias"]),
        safe_float(overall.iloc[0]["within10"]),
        safe_float(overall.iloc[0]["within20"]),

        mae_24h,
        mae_48h,
        mae_72h,

        safe_float(baseline.iloc[0]["baseline_mae"]),
        safe_float(baseline.iloc[0]["baseline_rmse"]),
        safe_float(baseline.iloc[0]["improvement_pct"]),

        recent_mean,
        historical_mean,
        mean_change,

        recent_std,
        historical_std,
        std_change,

        drift_status,
        model_status,
        recommendation,

        datetime.now()
    ])

    return next_id


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "PEARLSAQI PRODUCTION MODEL MONITOR"
    )

    c = get_connection()

    try:

        # ====================================================
        # OVERALL PERFORMANCE
        # ====================================================

        header(
            "OVERALL PRODUCTION PERFORMANCE"
        )

        overall = get_overall_performance(c)

        print(
            overall.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        evaluated = int(
            overall.iloc[0]["evaluated_forecasts"]
        )

        # ====================================================
        # HORIZON
        # ====================================================

        header(
            "PERFORMANCE BY HORIZON"
        )

        horizon = get_horizon_performance(c)

        if horizon.empty:

            print("No evaluated forecasts yet.")

        else:

            print(
                horizon.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}"
                )
            )

        mae_24h = horizon_mae(
            horizon,
            1
        )

        mae_48h = horizon_mae(
            horizon,
            2
        )

        mae_72h = horizon_mae(
            horizon,
            3
        )

        # ====================================================
        # CITY
        # ====================================================

        header(
            "PERFORMANCE BY CITY"
        )

        city = get_city_performance(c)

        if city.empty:

            print(
                "No evaluated forecasts yet."
            )

        else:

            print(
                city.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}"
                )
            )

        # ====================================================
        # FORECAST STATUS
        # ====================================================

        header(
            "FORECAST STATUS"
        )

        status_df = get_forecast_status(c)

        print(
            status_df.to_string(
                index=False
            )
        )

        # ====================================================
        # BASELINE
        # ====================================================

        header(
            "MODEL VS NAIVE BASELINE"
        )

        baseline = get_baseline_comparison(c)

        print(
            baseline.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        baseline_improvement = safe_float(
            baseline.iloc[0]["improvement_pct"]
        )

        if baseline_improvement is not None:

            if baseline_improvement > 0:

                print(
                    f"\nML MODEL BEATS BASELINE "
                    f"BY {baseline_improvement:.2f}%"
                )

            else:

                print(
                    f"\nML MODEL IS "
                    f"{abs(baseline_improvement):.2f}% "
                    f"WORSE THAN BASELINE"
                )

        # ====================================================
        # AQI DRIFT
        # ====================================================

        header(
            "AQI DISTRIBUTION DRIFT"
        )

        recent = get_aqi_distribution(
            c,
            days=30
        )

        historical = get_historical_aqi_distribution(
            c,
            days=30
        )

        recent_mean = None
        historical_mean = None

        recent_std = None
        historical_std = None

        mean_change = None
        std_change = None

        if recent is not None:

            recent_mean = safe_float(
                recent[0]
            )

            recent_std = safe_float(
                recent[1]
            )

        if historical is not None:

            historical_mean = safe_float(
                historical[0]
            )

            historical_std = safe_float(
                historical[1]
            )

        mean_change = percentage_change(
            historical_mean,
            recent_mean
        )

        std_change = percentage_change(
            historical_std,
            recent_std
        )

        print(
            f"Recent mean AQI      : {fmt(recent_mean)}"
        )

        print(
            f"Historical mean AQI  : {fmt(historical_mean)}"
        )

        print(
            f"Mean change          : "
            f"{mean_change:+.2f}%"
            if mean_change is not None
            else "Mean change          : N/A"
        )

        print(
            f"Recent std AQI       : {fmt(recent_std)}"
        )

        print(
            f"Historical std AQI   : {fmt(historical_std)}"
        )

        print(
            f"Std change           : "
            f"{std_change:+.2f}%"
            if std_change is not None
            else "Std change           : N/A"
        )

        # ====================================================
        # DRIFT WARNINGS
        # ====================================================

        warnings = []

        if (
            mean_change is not None
            and abs(mean_change) >= DRIFT_MEAN_THRESHOLD
        ):

            warnings.append(
                f"AQI mean changed by "
                f"{mean_change:+.2f}%"
            )

        if (
            std_change is not None
            and abs(std_change) >= DRIFT_STD_THRESHOLD
        ):

            warnings.append(
                f"AQI standard deviation changed by "
                f"{std_change:+.2f}%"
            )

        # ====================================================
        # RECENT PERFORMANCE WARNING
        # ====================================================

        recent_mae = safe_float(
            overall.iloc[0]["mae"]
        )

        if (
            recent_mae is not None
            and recent_mae >= RECENT_MAE_WARNING
            and evaluated >= MIN_PRODUCTION_FORECASTS
        ):

            warnings.append(
                f"Overall production MAE is "
                f"{recent_mae:.2f}"
            )

        # ====================================================
        # STATUS
        # ====================================================

        (
            model_status,
            drift_reason,
            recommendation
        ) = determine_status(
            evaluated,
            warnings
        )

        if warnings:

            drift_status = "WARNING"

        else:

            drift_status = "NORMAL"

        # ====================================================
        # DRIFT STATUS
        # ====================================================

        header(
            "MODEL DRIFT STATUS"
        )

        print(
            f"Drift status : {drift_status}"
        )

        print(
            f"Model status : {model_status}"
        )

        print(
            f"Reason       : {drift_reason}"
        )

        if warnings:

            print("\nWarnings:")

            for warning in warnings:

                print(
                    f"  - {warning}"
                )

        else:

            print(
                "\nNo significant drift warnings."
            )

        # ====================================================
        # RECOMMENDATION
        # ====================================================

        header(
            "RECOMMENDATION"
        )

        print(
            recommendation
        )

        if evaluated < MIN_PRODUCTION_FORECASTS:

            print(
                f"\nCurrent evaluated forecasts : "
                f"{evaluated}"
            )

            print(
                f"Recommended minimum         : "
                f"{MIN_PRODUCTION_FORECASTS}"
            )

            print(
                "\nDo NOT retrain or recalibrate yet."
            )

        elif model_status == MODEL_STATUS_REVIEW:

            print(
                "\nDo NOT automatically replace "
                "the production model."
            )

            print(
                "Investigate the warning signals first."
            )

        else:

            print(
                "\nProduction model remains stable."
            )

        # ====================================================
        # SAVE SNAPSHOT
        # ====================================================

        header(
            "SAVING MONITORING SNAPSHOT"
        )

        snapshot_id = save_monitoring_snapshot(
            c=c,
            evaluated=evaluated,
            overall=overall,
            mae_24h=mae_24h,
            mae_48h=mae_48h,
            mae_72h=mae_72h,
            baseline=baseline,
            recent_mean=recent_mean,
            historical_mean=historical_mean,
            mean_change=mean_change,
            recent_std=recent_std,
            historical_std=historical_std,
            std_change=std_change,
            drift_status=drift_status,
            model_status=model_status,
            recommendation=recommendation
        )

        print(
            f"Monitoring snapshot saved."
        )

        print(
            f"Snapshot ID : {snapshot_id}"
        )

        # ====================================================
        # HISTORY
        # ====================================================

        header(
            "MONITORING HISTORY"
        )

        history = c.execute("""
            SELECT

                run_date,

                evaluated_forecasts,

                overall_mae,

                overall_rmse,

                baseline_improvement_pct,

                mean_aqi_change_pct,

                std_aqi_change_pct,

                drift_status,

                model_status

            FROM model_monitoring_history

            ORDER BY run_date DESC, id DESC

            LIMIT 10
        """).fetchdf()

        print(
            history.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    finally:

        c.close()

    header(
        "PRODUCTION MODEL MONITOR COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()