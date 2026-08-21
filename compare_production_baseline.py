from src.database.connection import get_connection


def main():

    print("\n" + "=" * 70)
    print("PEARLSAQI PRODUCTION MODEL vs BASELINE")
    print("=" * 70)

    conn = get_connection()

    # ---------------------------------------------------------
    # Check evaluated forecasts
    # ---------------------------------------------------------

    evaluated = conn.execute("""
        SELECT
            city_name,
            origin_date,
            forecast_date,
            horizon,
            predicted_aqi,
            actual_aqi,
            absolute_error
        FROM forecast_predictions
        WHERE actual_aqi IS NOT NULL
        ORDER BY origin_date, city_name, horizon
    """).fetchdf()

    if evaluated.empty:

        print("\nNo evaluated forecasts available yet.")
        print("Run this again after actual AQI values arrive.")

        conn.close()
        return

    # ---------------------------------------------------------
    # Build naive baseline
    #
    # Baseline prediction:
    # use the AQI from the forecast origin date
    # ---------------------------------------------------------

    baseline = conn.execute("""
        SELECT
            f.city_name,
            f.origin_date,
            f.forecast_date,
            f.horizon,
            f.predicted_aqi,
            f.actual_aqi,

            a.aqi AS baseline_aqi

        FROM forecast_predictions f

        INNER JOIN aqi a
            ON f.city_id = a.city_id
            AND f.origin_date = a.date

        WHERE f.actual_aqi IS NOT NULL

        ORDER BY
            f.origin_date,
            f.city_name,
            f.horizon
    """).fetchdf()

    if baseline.empty:

        print("\nCould not build baseline.")
        print("Check that origin-date AQI exists in the AQI table.")

        conn.close()
        return

    # ---------------------------------------------------------
    # Calculate baseline errors
    # ---------------------------------------------------------

    baseline["baseline_error"] = (
        baseline["actual_aqi"]
        - baseline["baseline_aqi"]
    )

    baseline["baseline_absolute_error"] = (
        baseline["baseline_error"]
        .abs()
    )

    # ---------------------------------------------------------
    # Model errors
    # ---------------------------------------------------------

    baseline["model_error"] = (
        baseline["actual_aqi"]
        - baseline["predicted_aqi"]
    )

    baseline["model_absolute_error"] = (
        baseline["model_error"]
        .abs()
    )

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    model_mae = baseline["model_absolute_error"].mean()

    baseline_mae = (
        baseline["baseline_absolute_error"].mean()
    )

    model_rmse = (
        (baseline["model_error"] ** 2).mean()
    ) ** 0.5

    baseline_rmse = (
        (baseline["baseline_error"] ** 2).mean()
    ) ** 0.5

    improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
        * 100
        if baseline_mae != 0
        else 0
    )

    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)

    print(
        f"\nEvaluated forecasts : {len(baseline)}"
    )

    print(
        f"\nML MODEL"
    )

    print(
        f"MAE                 : {model_mae:.4f}"
    )

    print(
        f"RMSE                : {model_rmse:.4f}"
    )

    print(
        f"\nNAIVE BASELINE"
    )

    print(
        f"MAE                 : {baseline_mae:.4f}"
    )

    print(
        f"RMSE                : {baseline_rmse:.4f}"
    )

    print(
        f"\nBaseline MAE improvement: "
        f"{improvement:+.2f}%"
    )

    if improvement > 0:

        print(
            "\nRESULT: ML MODEL BEATS BASELINE"
        )

    elif improvement < 0:

        print(
            "\nRESULT: BASELINE CURRENTLY BEATS ML MODEL"
        )

    else:

        print(
            "\nRESULT: ML MODEL AND BASELINE ARE EQUAL"
        )

    # ---------------------------------------------------------
    # By horizon
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON BY HORIZON")
    print("=" * 70)

    horizon_summary = (
        baseline
        .groupby("horizon")
        .agg(
            n=("actual_aqi", "count"),
            model_mae=("model_absolute_error", "mean"),
            baseline_mae=(
                "baseline_absolute_error",
                "mean"
            )
        )
        .reset_index()
    )

    horizon_summary["improvement_pct"] = (
        (
            horizon_summary["baseline_mae"]
            - horizon_summary["model_mae"]
        )
        / horizon_summary["baseline_mae"]
        * 100
    )

    print(
        horizon_summary
        .round(4)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # By city
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON BY CITY")
    print("=" * 70)

    city_summary = (
        baseline
        .groupby("city_name")
        .agg(
            n=("actual_aqi", "count"),
            model_mae=("model_absolute_error", "mean"),
            baseline_mae=(
                "baseline_absolute_error",
                "mean"
            )
        )
        .reset_index()
    )

    city_summary["improvement_pct"] = (
        (
            city_summary["baseline_mae"]
            - city_summary["model_mae"]
        )
        / city_summary["baseline_mae"]
        * 100
    )

    print(
        city_summary
        .sort_values("model_mae")
        .round(4)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    output_file = (
        "models/forecast/v9/"
        "production_baseline_comparison.csv"
    )

    city_summary.to_csv(
        output_file,
        index=False
    )

    print("\nSaved city comparison:")
    print(output_file)

    conn.close()

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()