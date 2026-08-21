from datetime import timedelta

from src.database.connection import get_connection


import pandas as pd
# ============================================================
# PEARLSAQI MODEL DRIFT MONITOR
# ============================================================

MIN_PRODUCTION_FORECASTS = 36

# Warning thresholds
AQI_MEAN_WARNING_PCT = 20.0
AQI_STD_WARNING_PCT = 30.0
ERROR_WARNING_MAE = 20.0


# ============================================================
# DISPLAY HELPERS
# ============================================================

def header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# AQI DATA DISTRIBUTION
# ============================================================

def get_aqi_distribution(c):

    return c.execute("""
        SELECT
            COUNT(*) AS rows,
            AVG(aqi) AS mean_aqi,
            STDDEV(aqi) AS std_aqi,
            MIN(aqi) AS min_aqi,
            MAX(aqi) AS max_aqi
        FROM aqi
    """).fetchdf()


# ============================================================
# AQI DISTRIBUTION BY CITY
# ============================================================

def get_city_distribution(c):

    return c.execute("""
        SELECT
            cities.city_name,
            COUNT(*) AS rows,
            AVG(aqi.aqi) AS mean_aqi,
            STDDEV(aqi.aqi) AS std_aqi
        FROM aqi
        INNER JOIN cities
            ON aqi.city_id = cities.city_id
        GROUP BY cities.city_name
        ORDER BY cities.city_name
    """).fetchdf()


# ============================================================
# RECENT AQI DISTRIBUTION
# ============================================================

def get_recent_aqi_distribution(c, days=30):

    latest = c.execute("""
        SELECT MAX(date)
        FROM aqi
    """).fetchone()[0]

    if latest is None:

        return c.execute("""
            SELECT
                0 AS rows,
                NULL AS mean_aqi,
                NULL AS std_aqi,
                NULL AS min_aqi,
                NULL AS max_aqi
        """).fetchdf()

    cutoff = latest - timedelta(days=days)

    return c.execute("""
        SELECT
            COUNT(*) AS rows,
            AVG(aqi) AS mean_aqi,
            STDDEV(aqi) AS std_aqi,
            MIN(aqi) AS min_aqi,
            MAX(aqi) AS max_aqi
        FROM aqi
        WHERE date >= ?
    """, [cutoff]).fetchdf()


# ============================================================
# HISTORICAL AQI DISTRIBUTION
# ============================================================

def get_historical_aqi_distribution(c, days=30):

    latest = c.execute("""
        SELECT MAX(date)
        FROM aqi
    """).fetchone()[0]

    if latest is None:

        return c.execute("""
            SELECT
                0 AS rows,
                NULL AS mean_aqi,
                NULL AS std_aqi,
                NULL AS min_aqi,
                NULL AS max_aqi
        """).fetchdf()

    cutoff = latest - timedelta(days=days)

    return c.execute("""
        SELECT
            COUNT(*) AS rows,
            AVG(aqi) AS mean_aqi,
            STDDEV(aqi) AS std_aqi,
            MIN(aqi) AS min_aqi,
            MAX(aqi) AS max_aqi
        FROM aqi
        WHERE date < ?
    """, [cutoff]).fetchdf()


# ============================================================
# PRODUCTION FORECAST ERROR
# ============================================================

def get_production_error(c):

    return c.execute("""
        SELECT
            COUNT(*) AS evaluated,
            AVG(absolute_error) AS mae,
            SQRT(AVG(error * error)) AS rmse,
            AVG(error) AS bias
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
    """).fetchdf()


# ============================================================
# RECENT PRODUCTION ERROR
# ============================================================

def get_recent_production_error(c, days=7):

    latest = c.execute("""
        SELECT MAX(forecast_date)
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
    """).fetchone()[0]

    if latest is None:

        return c.execute("""
            SELECT
                0 AS evaluated,
                NULL AS mae,
                NULL AS rmse,
                NULL AS bias
        """).fetchdf()

    cutoff = latest - timedelta(days=days)

    return c.execute("""
        SELECT
            COUNT(*) AS evaluated,
            AVG(absolute_error) AS mae,
            SQRT(AVG(error * error)) AS rmse,
            AVG(error) AS bias
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
          AND forecast_date >= ?
    """, [cutoff]).fetchdf()


# ============================================================
# PERCENTAGE CHANGE
# ============================================================

def percentage_change(old, new):

    if old is None or new is None:
        return None

    if old == 0:
        return None

    return ((new - old) / abs(old)) * 100.0


# ============================================================
# SAFE FLOAT FORMAT
# ============================================================

def format_value(value):

    if value is None:
        return "N/A"

    try:

        if value != value:
            return "N/A"

        return f"{float(value):.4f}"

    except (TypeError, ValueError):

        return "N/A"


# ============================================================
# MAIN
# ============================================================

def main():

    header("PEARLSAQI MODEL DRIFT MONITOR")

    c = get_connection()

    try:

        # ====================================================
        # 1. AQI DATA DISTRIBUTION
        # ====================================================

        header("AQI DATA DISTRIBUTION")

        overall = get_aqi_distribution(c)

        print(
            overall.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        # ====================================================
        # 2. RECENT VS HISTORICAL AQI
        # ====================================================

        header("RECENT VS HISTORICAL AQI")

        recent = get_recent_aqi_distribution(
            c,
            days=30
        )

        historical = get_historical_aqi_distribution(
            c,
            days=30
        )

        recent_row = recent.iloc[0]
        historical_row = historical.iloc[0]

        recent_mean = recent_row["mean_aqi"]
        historical_mean = historical_row["mean_aqi"]

        recent_std = recent_row["std_aqi"]
        historical_std = historical_row["std_aqi"]

        mean_change = percentage_change(
            historical_mean,
            recent_mean
        )

        std_change = percentage_change(
            historical_std,
            recent_std
        )

        print("\nHistorical period:")

        print(
            f"Mean AQI : {format_value(historical_mean)}"
        )

        print(
            f"Std AQI  : {format_value(historical_std)}"
        )

        print("\nRecent 30 days:")

        print(
            f"Mean AQI : {format_value(recent_mean)}"
        )

        print(
            f"Std AQI  : {format_value(recent_std)}"
        )

        print("\nDistribution change:")

        if mean_change is not None:

            print(
                f"Mean change : {mean_change:+.2f}%"
            )

        else:

            print(
                "Mean change : N/A"
            )

        if std_change is not None:

            print(
                f"Std change  : {std_change:+.2f}%"
            )

        else:

            print(
                "Std change  : N/A"
            )

        # ====================================================
        # 3. AQI DISTRIBUTION BY CITY
        # ====================================================

        header("AQI DISTRIBUTION BY CITY")

        city_distribution = get_city_distribution(c)

        if city_distribution.empty:

            print(
                "No city-level AQI data available."
            )

        else:

            print(
                city_distribution.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}"
                )
            )

        # ====================================================
        # 4. PRODUCTION FORECAST ERROR
        # ====================================================

        header("PRODUCTION FORECAST ERROR")

        production = get_production_error(c)

        print(
            production.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        evaluated_value = production.iloc[0]["evaluated"]

        if evaluated_value is None:

            evaluated = 0

        else:

            evaluated = int(evaluated_value)

        # ====================================================
        # 5. RECENT PRODUCTION ERROR
        # ====================================================

        header("RECENT 7-DAY FORECAST ERROR")

        recent_error = get_recent_production_error(
            c,
            days=7
        )

        print(
            recent_error.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        recent_mae = recent_error.iloc[0]["mae"]

        # ====================================================
        # 6. DRIFT SIGNAL DETECTION
        # ====================================================

        header("DRIFT SIGNALS")

        warnings = []

        # ----------------------------------------------------
        # AQI MEAN DRIFT
        # ----------------------------------------------------

        if (
            mean_change is not None
            and abs(mean_change) >= AQI_MEAN_WARNING_PCT
        ):

            warnings.append(
                f"AQI mean changed by "
                f"{mean_change:+.2f}% "
                f"(threshold: ±{AQI_MEAN_WARNING_PCT:.1f}%)"
            )

        # ----------------------------------------------------
        # AQI STD DRIFT
        # ----------------------------------------------------

        if (
            std_change is not None
            and abs(std_change) >= AQI_STD_WARNING_PCT
        ):

            warnings.append(
                f"AQI standard deviation changed by "
                f"{std_change:+.2f}% "
                f"(threshold: ±{AQI_STD_WARNING_PCT:.1f}%)"
            )

        # ----------------------------------------------------
        # FORECAST ERROR DRIFT
        # ----------------------------------------------------

        if (
        recent_mae is not None
        and not pd.isna(recent_mae)
        and float(recent_mae) >= ERROR_WARNING_MAE
    ):

            warnings.append(
                f"Recent production MAE is "
                f"{float(recent_mae):.2f} "
                f"(threshold: {ERROR_WARNING_MAE:.2f})"
            )

        if warnings:

            for warning in warnings:

                print(
                    f"WARNING: {warning}"
                )

        else:

            print(
                "No significant drift signals detected."
            )

        # ====================================================
        # 7. OVERALL DRIFT STATUS
        # ====================================================

        header("MODEL DRIFT STATUS")

        if evaluated < MIN_PRODUCTION_FORECASTS:

            status = "COLLECTING DATA"

            reason = (
                f"Only {evaluated} evaluated production "
                f"forecasts are available. "
                f"At least {MIN_PRODUCTION_FORECASTS} "
                f"are recommended before making a reliable "
                f"production drift decision."
            )

        elif warnings:

            status = "DRIFT WARNING"

            reason = (
                "One or more configured production "
                "drift thresholds were exceeded."
            )

        else:

            status = "STABLE"

            reason = (
                "No configured production drift "
                "thresholds were exceeded."
            )

        print(
            f"STATUS : {status}"
        )

        print(
            f"REASON : {reason}"
        )

        # ====================================================
        # 8. RECOMMENDATION
        # ====================================================

        header("RECOMMENDATION")

        if evaluated < MIN_PRODUCTION_FORECASTS:

            print(
                "Continue collecting real production forecasts."
            )

            print(
                f"Current evaluated forecasts : {evaluated}"
            )

            print(
                f"Recommended minimum         : "
                f"{MIN_PRODUCTION_FORECASTS}"
            )

            print(
                "Do NOT retrain or recalibrate yet."
            )

        elif warnings:

            print(
                "Investigate the drift signals."
            )

            print(
                "Compare recent production performance "
                "against the validation baseline."
            )

            print(
                "Do NOT automatically replace the production model."
            )

        else:

            print(
                "Production data appears stable."
            )

            print(
                "Continue normal daily monitoring."
            )

        # ====================================================
        # 9. MONITORING SUMMARY
        # ====================================================

        header("MONITORING SUMMARY")

        print(
            f"Total evaluated forecasts : {evaluated}"
        )

        print(
            f"Required minimum          : "
            f"{MIN_PRODUCTION_FORECASTS}"
        )

        print(
            f"AQI mean change           : "
            f"{mean_change:+.2f}%"
            if mean_change is not None
            else "AQI mean change           : N/A"
        )

        print(
            f"AQI std change            : "
            f"{std_change:+.2f}%"
            if std_change is not None
            else "AQI std change            : N/A"
        )

        print(
            f"Recent MAE                : "
            f"{format_value(recent_mae)}"
        )

        print(
            f"Warnings                  : "
            f"{len(warnings)}"
        )

    finally:

        c.close()

    header("MODEL DRIFT MONITOR COMPLETE")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
