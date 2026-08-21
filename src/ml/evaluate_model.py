import os
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Configuration
# ==========================================================

MODEL_FILE = "models/xgboost/final_xgboost_model.json"

TEST_FILE = "data/processed/test.parquet"

OUTPUT_DIR = "data/predictions"

OUTPUT_FILE = (
    f"{OUTPUT_DIR}/test_predictions.parquet"
)


# ==========================================================
# Features
# ==========================================================

FEATURE_COLUMNS = [
    "city_id",

    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",

    "temperature",
    "humidity",
    "precipitation",
    "windspeed",

    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_7",

    "aqi_rolling_3",
    "aqi_rolling_7",

    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
]

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Overall metrics
# ==========================================================

def calculate_overall_metrics(
    actual,
    predictions
):

    errors = actual - predictions

    absolute_errors = errors.abs()

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = mean_squared_error(
        actual,
        predictions
    ) ** 0.5

    r2 = r2_score(
        actual,
        predictions
    )

    mean_aqi = actual.mean()

    nmae = mae / mean_aqi

    within_10 = (
        absolute_errors <= 10
    ).mean()

    within_20 = (
        absolute_errors <= 20
    ).mean()

    within_30 = (
        absolute_errors <= 30
    ).mean()

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nmae": nmae,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


# ==========================================================
# Main evaluation
# ==========================================================

def evaluate():

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------
    # Load test data
    # ------------------------------------------------------

    test_df = pd.read_parquet(
        TEST_FILE
    )

    print()
    print(
        f"Test rows: {len(test_df)}"
    )

    print(
        f"Test start: {test_df['date'].min()}"
    )

    print(
        f"Test end: {test_df['date'].max()}"
    )

    # ------------------------------------------------------
    # Load final model
    # ------------------------------------------------------

    model = XGBRegressor()

    model.load_model(
        MODEL_FILE
    )

    print()
    print(
        f"Loaded model: {MODEL_FILE}"
    )

    # ------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ------------------------------------------------------
    # Create results dataframe
    # ------------------------------------------------------

    results = test_df[
        [
            "city_id",
            "city_name",
            "date",
            "aqi",
            "target_aqi"
        ]
    ].copy()

    results["prediction"] = predictions

    results["error"] = (
        results["target_aqi"]
        - results["prediction"]
    )

    results["absolute_error"] = (
        results["error"].abs()
    )

    # ======================================================
    # OVERALL RESULTS
    # ======================================================

    metrics = calculate_overall_metrics(
        results["target_aqi"],
        results["prediction"]
    )

    print()
    print("=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)

    print(
        f"MAE  : {metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.4f}"
    )

    print(
        f"R²   : {metrics['r2']:.4f}"
    )

    print(
        f"NMAE : {metrics['nmae']:.4%}"
    )

    print()
    print("Prediction accuracy:")

    print(
        f"Within ±10 AQI : "
        f"{metrics['within_10']:.2%}"
    )

    print(
        f"Within ±20 AQI : "
        f"{metrics['within_20']:.2%}"
    )

    print(
        f"Within ±30 AQI : "
        f"{metrics['within_30']:.2%}"
    )

    # ======================================================
    # PER CITY RESULTS
    # ======================================================

    print()
    print("=" * 60)
    print("PER-CITY PERFORMANCE")
    print("=" * 60)

    city_results = []

    for city_name, city_df in results.groupby(
        "city_name"
    ):

        city_metrics = calculate_overall_metrics(
            city_df["target_aqi"],
            city_df["prediction"]
        )

        city_results.append({

            "city": city_name,

            "rows": len(city_df),

            "mae": city_metrics["mae"],

            "rmse": city_metrics["rmse"],

            "r2": city_metrics["r2"],

            "nmae": city_metrics["nmae"],

            "within_10": city_metrics["within_10"],

            "within_20": city_metrics["within_20"],

            "within_30": city_metrics["within_30"],

        })

    city_df = pd.DataFrame(
        city_results
    )

    city_df = city_df.sort_values(
        "rmse"
    )

    print()

    print(
        city_df.to_string(
            index=False,
            formatters={
                "mae": "{:.2f}".format,
                "rmse": "{:.2f}".format,
                "r2": "{:.3f}".format,
                "nmae": "{:.2%}".format,
                "within_10": "{:.2%}".format,
                "within_20": "{:.2%}".format,
                "within_30": "{:.2%}".format,
            }
        )
    )

    # ======================================================
    # WORST PREDICTIONS
    # ======================================================

    print()
    print("=" * 60)
    print("10 WORST PREDICTIONS")
    print("=" * 60)

    worst_predictions = results.sort_values(
        "absolute_error",
        ascending=False
    ).head(10)

    print()

    print(
        worst_predictions[
            [
                "city_name",
                "date",
                "aqi",
                "target_aqi",
                "prediction",
                "absolute_error"
            ]
        ].to_string(
            index=False
        )
    )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    city_output = (
        f"{OUTPUT_DIR}/city_metrics.csv"
    )

    city_df.to_csv(
        city_output,
        index=False
    )

    print()
    print("=" * 60)
    print("EVALUATION FILES SAVED")
    print("=" * 60)

    print(
        f"Predictions: {OUTPUT_FILE}"
    )

    print(
        f"City metrics: {city_output}"
    )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    evaluate()