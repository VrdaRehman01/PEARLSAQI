import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/processed/v4/train.parquet"
VALIDATION_FILE = "data/processed/v4/validation.parquet"
TEST_FILE = "data/processed/v4/test.parquet"

RESULT_DIR = "models/walk_forward_v4"
RESULT_FILE = os.path.join(
    RESULT_DIR,
    "walk_forward_results.csv"
)

TARGET_COLUMN = "target_aqi"

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season"
]


# ==========================================================
# Metrics
# ==========================================================

def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = mean_squared_error(
        y_true,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_true,
        predictions
    )

    within_10 = (
        np.abs(
            y_true - predictions
        ) <= 10
    ).mean() * 100

    within_20 = (
        np.abs(
            y_true - predictions
        ) <= 20
    ).mean() * 100

    within_30 = (
        np.abs(
            y_true - predictions
        ) <= 30
    ).mean() * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30
    }


# ==========================================================
# Models
# ==========================================================

def create_models():

    models = {}

    # ------------------------------------------------------
    # HistGradientBoosting
    # ------------------------------------------------------

    models["HGB"] = HistGradientBoostingRegressor(

        learning_rate=0.03,

        max_iter=500,

        max_leaf_nodes=15,

        min_samples_leaf=30,

        l2_regularization=5.0,

        random_state=42
    )

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    models["XGB"] = XGBRegressor(

        n_estimators=800,

        learning_rate=0.025,

        max_depth=4,

        min_child_weight=8,

        subsample=0.85,

        colsample_bytree=0.80,

        reg_alpha=0.5,

        reg_lambda=5.0,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=42,

        n_jobs=-1
    )

    # ------------------------------------------------------
    # Ridge
    # ------------------------------------------------------

    models["Ridge"] = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "ridge",
            Ridge(alpha=0.01)
        )
    ])

    return models


# ==========================================================
# Ensemble
# ==========================================================

def create_ensemble_predictions(
    predictions
):

    # Best candidate from previous experiments
    #
    # HGB 50%
    # XGB 25%
    # Ridge 25%

    ensemble_prediction = (

        0.50 * predictions["HGB"]

        + 0.25 * predictions["XGB"]

        + 0.25 * predictions["Ridge"]
    )

    return ensemble_prediction


# ==========================================================
# Prepare features
# ==========================================================

def prepare_features(
    train_df,
    validation_df
):

    feature_columns = [

        column

        for column in train_df.columns

        if column not in EXCLUDED_COLUMNS

    ]

    missing_features = [

        column

        for column in feature_columns

        if column not in validation_df.columns

    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    return feature_columns


# ==========================================================
# Run one fold
# ==========================================================

def run_fold(
    train_df,
    validation_df,
    fold_name,
    feature_columns
):

    print()
    print("=" * 70)
    print(f"FOLD: {fold_name}")
    print("=" * 70)

    print(
        f"Training period: "
        f"{train_df['date'].min()} → "
        f"{train_df['date'].max()}"
    )

    print(
        f"Validation period: "
        f"{validation_df['date'].min()} → "
        f"{validation_df['date'].max()}"
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    models = create_models()

    predictions = {}

    results = []

    # ------------------------------------------------------
    # Train individual models
    # ------------------------------------------------------

    for model_name, model in models.items():

        print()
        print(
            f"Training {model_name}..."
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time()
            - start_time
        )

        prediction = model.predict(
            X_validation
        )

        predictions[
            model_name
        ] = prediction

        metrics = calculate_metrics(
            y_validation,
            prediction
        )

        results.append({

            "fold": fold_name,

            "model": model_name,

            "train_rows": len(train_df),

            "validation_rows": len(
                validation_df
            ),

            "mae": metrics["mae"],

            "rmse": metrics["rmse"],

            "r2": metrics["r2"],

            "within_10": metrics[
                "within_10"
            ],

            "within_20": metrics[
                "within_20"
            ],

            "within_30": metrics[
                "within_30"
            ],

            "training_time_seconds":
                training_time
        })

        print(
            f"{model_name}"
        )

        print(
            f"MAE       : "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"RMSE      : "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"R²        : "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"Within ±10: "
            f"{metrics['within_10']:.2f}%"
        )

    # ------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------

    ensemble_prediction = (
        create_ensemble_predictions(
            predictions
        )
    )

    ensemble_metrics = calculate_metrics(
        y_validation,
        ensemble_prediction
    )

    results.append({

        "fold": fold_name,

        "model":
            "HGB_XGB_Ridge_50_25_25",

        "train_rows": len(train_df),

        "validation_rows": len(
            validation_df
        ),

        "mae":
            ensemble_metrics["mae"],

        "rmse":
            ensemble_metrics["rmse"],

        "r2":
            ensemble_metrics["r2"],

        "within_10":
            ensemble_metrics["within_10"],

        "within_20":
            ensemble_metrics["within_20"],

        "within_30":
            ensemble_metrics["within_30"],

        "training_time_seconds":
            0
    })

    print()
    print(
        "HGB + XGB + Ridge "
        "(50/25/25)"
    )

    print(
        f"MAE       : "
        f"{ensemble_metrics['mae']:.4f}"
    )

    print(
        f"RMSE      : "
        f"{ensemble_metrics['rmse']:.4f}"
    )

    print(
        f"R²        : "
        f"{ensemble_metrics['r2']:.4f}"
    )

    print(
        f"Within ±10: "
        f"{ensemble_metrics['within_10']:.2f}%"
    )

    return results


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print(
        "AQI WALK-FORWARD VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "Loading datasets..."
    )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    test_df = pd.read_parquet(
        TEST_FILE
    )

    train_df["date"] = pd.to_datetime(
        train_df["date"]
    )

    validation_df["date"] = pd.to_datetime(
        validation_df["date"]
    )

    test_df["date"] = pd.to_datetime(
        test_df["date"]
    )

    print(
        f"Training rows    : "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows  : "
        f"{len(validation_df)}"
    )

    print(
        f"Test rows        : "
        f"{len(test_df)}"
    )

    # ------------------------------------------------------
    # Feature list
    # ------------------------------------------------------

    feature_columns = prepare_features(
        train_df,
        validation_df
    )

    print()
    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------------
    # Fold 1
    #
    # 2023-2023 -> 2024
    # ------------------------------------------------------

    train_fold_1 = train_df[
        train_df["date"].dt.year == 2023
    ]

    validation_fold_1 = train_df[
        train_df["date"].dt.year == 2024
    ]

    # ------------------------------------------------------
    # Fold 2
    #
    # 2023-2024 -> 2025
    # ------------------------------------------------------

    train_fold_2 = pd.concat([

        train_df,

        validation_df[
            validation_df["date"].dt.year == 2025
        ]

    ], ignore_index=True)

    # The above would incorrectly include 2025
    # because validation_df already contains 2025.
    #
    # Therefore rebuild correctly below.

    train_fold_2 = pd.concat([

        train_df,

        validation_df.iloc[0:0]

    ], ignore_index=True)

    # train_df = 2023-2024
    # validation_df = 2025
    validation_fold_2 = validation_df

    # ------------------------------------------------------
    # Fold 3
    #
    # 2023-2025 -> 2026
    # ------------------------------------------------------

    train_fold_3 = pd.concat([

        train_df,

        validation_df

    ], ignore_index=True)

    validation_fold_3 = test_df

    # ------------------------------------------------------
    # Run folds
    # ------------------------------------------------------

    all_results = []

    # Fold 1
    fold_1_results = run_fold(

        train_fold_1,

        validation_fold_1,

        "2023_train_2024_validate",

        feature_columns
    )

    all_results.extend(
        fold_1_results
    )

    # Fold 2
    fold_2_results = run_fold(

        train_fold_2,

        validation_fold_2,

        "2023_2024_train_2025_validate",

        feature_columns
    )

    all_results.extend(
        fold_2_results
    )

    # Fold 3
    fold_3_results = run_fold(

        train_fold_3,

        validation_fold_3,

        "2023_2025_train_2026_test",

        feature_columns
    )

    all_results.extend(
        fold_3_results
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print(
        "WALK-FORWARD RESULTS"
    )
    print("=" * 70)

    print()

    print(
        results_df[
            [
                "fold",
                "model",
                "mae",
                "rmse",
                "r2",
                "within_10",
                "within_20",
                "within_30"
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Average performance
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AVERAGE PERFORMANCE"
    )
    print("=" * 70)

    average_results = (

        results_df

        .groupby("model")

        [

            [
                "mae",
                "rmse",
                "r2",
                "within_10",
                "within_20",
                "within_30"
            ]

        ]

        .mean()

        .sort_values(
            "mae"
        )
    )

    print(
        average_results.to_string()
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )

    # ------------------------------------------------------
    # Best model
    # ------------------------------------------------------

    best_model = (
        average_results
        .sort_values("mae")
        .index[0]
    )

    print()
    print("=" * 70)
    print(
        "BEST WALK-FORWARD MODEL"
    )
    print("=" * 70)

    print(
        f"Model: {best_model}"
    )

    print(
        f"Average MAE: "
        f"{average_results.loc[best_model, 'mae']:.4f}"
    )

    print(
        f"Average RMSE: "
        f"{average_results.loc[best_model, 'rmse']:.4f}"
    )

    print(
        f"Average R²: "
        f"{average_results.loc[best_model, 'r2']:.4f}"
    )


if __name__ == "__main__":
    main()