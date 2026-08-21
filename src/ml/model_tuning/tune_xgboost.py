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

TRAIN_FILE = "data/processed/train.parquet"
VALIDATION_FILE = "data/processed/validation.parquet"

MODEL_DIR = "models/xgboost"

BEST_MODEL_FILE = (
    f"{MODEL_DIR}/best_xgboost_model.json"
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
# Evaluation
# ==========================================================

def calculate_metrics(model, X, y):

    predictions = model.predict(X)

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

    return mae, rmse, r2


# ==========================================================
# Main tuning function
# ==========================================================

def tune_model():

    print("=" * 60)
    print("XGBOOST MODEL TUNING")
    print("=" * 60)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print()
    print(
        f"Training rows   : {len(train_df)}"
    )

    print(
        f"Validation rows : {len(validation_df)}"
    )

    # ------------------------------------------------------
    # Prepare X / y
    # ------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    # ------------------------------------------------------
    # Candidate models
    # ------------------------------------------------------

    models = [

        {
            "name": "Model_1_Baseline",

            "params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        },

        {
            "name": "Model_2_Less_Overfit",

            "params": {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 4,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        },

        {
            "name": "Model_3_Regularized",

            "params": {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 5,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            }
        },

        {
            "name": "Model_4_Deeper",

            "params": {
                "n_estimators": 600,
                "learning_rate": 0.04,
                "max_depth": 5,
                "min_child_weight": 7,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            }
        },

        {
            "name": "Model_5_Strong_Regularization",

            "params": {
                "n_estimators": 900,
                "learning_rate": 0.025,
                "max_depth": 3,
                "min_child_weight": 8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        }

    ]

    # ------------------------------------------------------
    # Train models
    # ------------------------------------------------------

    results = []

    best_model = None
    best_model_name = None

    best_rmse = float("inf")

    print()
    print("=" * 60)
    print("TESTING MODELS")
    print("=" * 60)

    for experiment in models:

        name = experiment["name"]

        params = experiment["params"]

        print()
        print("-" * 60)
        print(name)
        print("-" * 60)

        model = XGBRegressor(

            objective="reg:squarederror",

            eval_metric="rmse",

            random_state=42,

            n_jobs=-1,

            **params
        )

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (X_validation, y_validation)
            ],

            verbose=False
        )

        mae, rmse, r2 = calculate_metrics(
            model,
            X_validation,
            y_validation
        )

        print(
            f"Validation MAE  : {mae:.4f}"
        )

        print(
            f"Validation RMSE : {rmse:.4f}"
        )

        print(
            f"Validation R²   : {r2:.4f}"
        )

        results.append({

            "model": name,

            "mae": mae,

            "rmse": rmse,

            "r2": r2

        })

        # --------------------------------------------------
        # Select best model
        # --------------------------------------------------

        if rmse < best_rmse:

            best_rmse = rmse

            best_model = model

            best_model_name = name

    # ------------------------------------------------------
    # Results table
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "rmse"
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save best model
    # ------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    best_model.save_model(
        BEST_MODEL_FILE
    )

    print()
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Model : {best_model_name}"
    )

    print(
        f"RMSE  : {best_rmse:.4f}"
    )

    print()
    print(
        f"Saved to: {BEST_MODEL_FILE}"
    )

    # ------------------------------------------------------
    # Save tuning results
    # ------------------------------------------------------

    results_file = (
        f"{MODEL_DIR}/tuning_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print(
        f"Results saved to: {results_file}"
    )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    tune_model()