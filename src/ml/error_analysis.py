import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Configuration
# ==========================================================

TEST_FILE = "data/processed/test.parquet"

MODEL_FILE = "models/xgboost_aqi_model.json"

OUTPUT_DIR = "data/analysis"

TARGET = "target_aqi"


# ==========================================================
# Features
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("AQI MODEL ERROR ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------
    # Load test data
    # ------------------------------------------------------

    print()
    print("Loading test data...")

    df = pd.read_parquet(
        TEST_FILE
    )

    print(
        f"Test rows: {len(df)}"
    )

    # ------------------------------------------------------
    # Create feature list
    # ------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[
        feature_columns
    ]

    y = df[
        TARGET
    ]

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    print()
    print("Loading model...")

    model = XGBRegressor()

    model.load_model(
        MODEL_FILE
    )

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    predictions = model.predict(X)

    df["prediction"] = predictions

    df["absolute_error"] = (
        abs(
            df[TARGET]
            - df["prediction"]
        )
    )

    df["error"] = (
        df["prediction"]
        - df[TARGET]
    )

    # ======================================================
    # OVERALL METRICS
    # ======================================================

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = mean_squared_error(
        y,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions
    )

    print()
    print("=" * 60)
    print("OVERALL TEST PERFORMANCE")
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

    # ======================================================
    # CITY ANALYSIS
    # ======================================================

    print()
    print("=" * 60)
    print("CITY ERROR ANALYSIS")
    print("=" * 60)

    city_results = []

    for city, group in df.groupby(
        "city_name"
    ):

        city_mae = mean_absolute_error(
            group[TARGET],
            group["prediction"]
        )

        city_rmse = (
            mean_squared_error(
                group[TARGET],
                group["prediction"]
            ) ** 0.5
        )

        city_r2 = r2_score(
            group[TARGET],
            group["prediction"]
        )

        city_results.append({

            "city":
                city,

            "rows":
                len(group),

            "mae":
                city_mae,

            "rmse":
                city_rmse,

            "r2":
                city_r2,

            "mean_actual":
                group[TARGET].mean(),

            "mean_prediction":
                group["prediction"].mean(),

            "max_actual":
                group[TARGET].max(),

            "max_prediction":
                group["prediction"].max()

        })

    city_results = pd.DataFrame(
        city_results
    )

    city_results = city_results.sort_values(
        "mae",
        ascending=False
    )

    print(
        city_results.to_string(
            index=False
        )
    )

    # ======================================================
    # AQI RANGE ANALYSIS
    # ======================================================

    print()
    print("=" * 60)
    print("AQI RANGE ERROR ANALYSIS")
    print("=" * 60)

    df["aqi_range"] = pd.cut(

        df[TARGET],

        bins=[
            0,
            50,
            100,
            150,
            200,
            300,
            500,
            np.inf
        ],

        labels=[
            "0-50 Good",
            "51-100 Moderate",
            "101-150 Unhealthy",
            "151-200 Very Unhealthy",
            "201-300 Severe",
            "301-500 Extreme",
            "500+"
        ]
    )

    range_results = (
        df
        .groupby(
            "aqi_range",
            observed=True
        )
        .agg(

            rows=(
                TARGET,
                "count"
            ),

            mae=(
                "absolute_error",
                "mean"
            ),

            rmse=(
                "absolute_error",
                lambda x:
                    np.sqrt(
                        np.mean(
                            x ** 2
                        )
                    )
            ),

            actual_mean=(
                TARGET,
                "mean"
            ),

            prediction_mean=(
                "prediction",
                "mean"
            )

        )
        .reset_index()
    )

    print(
        range_results.to_string(
            index=False
        )
    )

    # ======================================================
    # UNDER / OVER PREDICTION
    # ======================================================

    print()
    print("=" * 60)
    print("BIAS ANALYSIS")
    print("=" * 60)

    mean_error = df["error"].mean()

    print(
        f"Mean error: {mean_error:.4f}"
    )

    if mean_error < 0:

        print(
            "Model tends to OVERPREDICT."
        )

    elif mean_error > 0:

        print(
            "Model tends to UNDERPREDICT."
        )

    else:

        print(
            "Model has approximately zero bias."
        )

    # ======================================================
    # WORST PREDICTIONS
    # ======================================================

    print()
    print("=" * 60)
    print("TOP 20 WORST PREDICTIONS")
    print("=" * 60)

    worst = df.sort_values(
        "absolute_error",
        ascending=False
    ).head(20)

    print(
        worst[
            [
                "city_name",
                "date",
                "aqi",
                TARGET,
                "prediction",
                "absolute_error",
                "error"
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

    df.to_parquet(
        os.path.join(
            OUTPUT_DIR,
            "test_error_analysis.parquet"
        ),
        index=False
    )

    city_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "city_error_analysis.csv"
        ),
        index=False
    )

    range_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "aqi_range_error_analysis.csv"
        ),
        index=False
    )

    print()
    print("=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print(
        "Saved:"
    )

    print(
        "data/analysis/test_error_analysis.parquet"
    )

    print(
        "data/analysis/city_error_analysis.csv"
    )

    print(
        "data/analysis/aqi_range_error_analysis.csv"
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()