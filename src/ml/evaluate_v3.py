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

TEST_FILE = "data/processed/v3/test.parquet"

MODEL_FILE = (
    "models/xgboost/"
    "v3_xgboost_aqi_model.json"
)


EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Main
# ==========================================================

def evaluate_v3():

    print("=" * 60)
    print("V3 MODEL - 2026 TEST EVALUATION")
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
        f"Test date: "
        f"{test_df['date'].min()} → "
        f"{test_df['date'].max()}"
    )

    # ------------------------------------------------------
    # Build features
    # ------------------------------------------------------

    FEATURE_COLUMNS = [
        column
        for column in test_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    print()
    print("Loading V3 model...")

    model = XGBRegressor()

    model.load_model(
        MODEL_FILE
    )

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("V3 TEST RESULTS")
    print("=" * 60)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    # ------------------------------------------------------
    # Prediction accuracy
    # ------------------------------------------------------

    absolute_error = (
        abs(
            y_test -
            predictions
        )
    )

    within_10 = (
        absolute_error <= 10
    ).mean() * 100

    within_20 = (
        absolute_error <= 20
    ).mean() * 100

    within_30 = (
        absolute_error <= 30
    ).mean() * 100

    print()
    print("=" * 60)
    print("PREDICTION ACCURACY")
    print("=" * 60)

    print(
        f"Within ±10 AQI : "
        f"{within_10:.2f}%"
    )

    print(
        f"Within ±20 AQI : "
        f"{within_20:.2f}%"
    )

    print(
        f"Within ±30 AQI : "
        f"{within_30:.2f}%"
    )

    # ------------------------------------------------------
    # Create results dataframe
    # ------------------------------------------------------

    results = test_df[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi"
        ]
    ].copy()

    results[
        "prediction"
    ] = predictions

    results[
        "absolute_error"
    ] = absolute_error

    results[
        "error"
    ] = (
        predictions -
        y_test
    )

    # ------------------------------------------------------
    # Worst predictions
    # ------------------------------------------------------

    worst = (
        results
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(20)
    )

    print()
    print("=" * 60)
    print("WORST 20 PREDICTIONS")
    print("=" * 60)

    print(
        worst.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # City metrics
    # ------------------------------------------------------

    city_metrics = []

    for city, group in results.groupby(
        "city_name"
    ):

        city_mae = mean_absolute_error(
            group["target_aqi"],
            group["prediction"]
        )

        city_rmse = mean_squared_error(
            group["target_aqi"],
            group["prediction"]
        ) ** 0.5

        city_r2 = r2_score(
            group["target_aqi"],
            group["prediction"]
        )

        city_metrics.append({

            "city": city,

            "rows": len(group),

            "mae": city_mae,

            "rmse": city_rmse,

            "r2": city_r2,

            "max_actual":
                group["target_aqi"].max(),

            "max_prediction":
                group["prediction"].max()
        })

    city_df = pd.DataFrame(
        city_metrics
    )

    city_df = city_df.sort_values(
        "mae",
        ascending=False
    )

    print()
    print("=" * 60)
    print("CITY PERFORMANCE")
    print("=" * 60)

    print(
        city_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_dir = (
        "data/analysis/v3"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results.to_parquet(
        os.path.join(
            output_dir,
            "test_predictions.parquet"
        ),
        index=False
    )

    city_df.to_csv(
        os.path.join(
            output_dir,
            "city_metrics.csv"
        ),
        index=False
    )

    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print(
        "data/analysis/v3/test_predictions.parquet"
    )

    print(
        "data/analysis/v3/city_metrics.csv"
    )


if __name__ == "__main__":

    evaluate_v3()