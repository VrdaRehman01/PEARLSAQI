import os
import time
import joblib
import numpy as np
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

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"
TEST_FILE = "data/processed/v3/test.parquet"

BASE_MODEL_DIR = "models/extreme_residual"

BASE_MODEL_FILE = os.path.join(
    BASE_MODEL_DIR,
    "base_xgboost_model.json"
)

RESIDUAL_MODEL_FILE = os.path.join(
    BASE_MODEL_DIR,
    "extreme_residual_model.json"
)

RESULT_FILE = os.path.join(
    BASE_MODEL_DIR,
    "extreme_residual_results.csv"
)

PREDICTION_FILE = os.path.join(
    BASE_MODEL_DIR,
    "test_predictions.parquet"
)

TARGET_COLUMN = "target_aqi"

EXTREME_THRESHOLD = 200


# ==========================================================
# Features
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season"
]


# ==========================================================
# Metrics
# ==========================================================

def calculate_metrics(
    y_true,
    predictions
):

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
# Feature preparation
# ==========================================================

def get_features(
    train_df,
    other_df
):

    features = [

        column

        for column in train_df.columns

        if column not in EXCLUDED_COLUMNS

    ]

    missing = [

        column

        for column in features

        if column not in other_df.columns

    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    return features


# ==========================================================
# Create base XGBoost
# ==========================================================

def create_base_model():

    return XGBRegressor(

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


# ==========================================================
# Create residual model
# ==========================================================

def create_residual_model():

    return XGBRegressor(

        n_estimators=300,

        learning_rate=0.03,

        max_depth=3,

        min_child_weight=5,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_alpha=0.5,

        reg_lambda=5.0,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=42,

        n_jobs=-1
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print(
        "EXTREME AQI RESIDUAL CORRECTION MODEL"
    )
    print("=" * 70)

    os.makedirs(
        BASE_MODEL_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------

    print()
    print("Loading training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print()
    print("Loading validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    print()
    print("Loading 2026 test data...")

    test_df = pd.read_parquet(
        TEST_FILE
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    # ------------------------------------------------------
    # Dates
    # ------------------------------------------------------

    for df in [
        train_df,
        validation_df,
        test_df
    ]:

        df["date"] = pd.to_datetime(
            df["date"]
        )

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    feature_columns = get_features(
        train_df,
        validation_df
    )

    print()
    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    # ======================================================
    # STEP 1
    # Train base XGBoost
    # ======================================================

    print()
    print("=" * 70)
    print(
        "STEP 1 - TRAINING BASE XGBOOST"
    )
    print("=" * 70)

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

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    base_model = create_base_model()

    start = time.time()

    base_model.fit(
        X_train,
        y_train
    )

    base_training_time = (
        time.time() - start
    )

    print(
        f"Training time: "
        f"{base_training_time:.2f}s"
    )

    # ------------------------------------------------------
    # Base predictions
    # ------------------------------------------------------

    validation_base = (
        base_model.predict(
            X_validation
        )
    )

    test_base = (
        base_model.predict(
            X_test
        )
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_base
    )

    test_base_metrics = calculate_metrics(
        y_test,
        test_base
    )

    print()
    print("BASE XGBOOST - VALIDATION")

    print(
        f"MAE  : "
        f"{validation_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{validation_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
        f"{validation_metrics['r2']:.4f}"
    )

    print()
    print("BASE XGBOOST - 2026 TEST")

    print(
        f"MAE  : "
        f"{test_base_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{test_base_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
        f"{test_base_metrics['r2']:.4f}"
    )

    # ======================================================
    # STEP 2
    # Create residual training data
    # ======================================================

    print()
    print("=" * 70)
    print(
        "STEP 2 - BUILDING EXTREME RESIDUAL DATA"
    )
    print("=" * 70)

    train_base = (
        base_model.predict(
            X_train
        )
    )

    train_residual = (
        y_train.to_numpy()
        - train_base
    )

    extreme_mask = (
        y_train.to_numpy()
        > EXTREME_THRESHOLD
    )

    extreme_count = (
        extreme_mask.sum()
    )

    print(
        f"Extreme training rows: "
        f"{extreme_count}"
    )

    print(
        f"Extreme threshold: "
        f"AQI > {EXTREME_THRESHOLD}"
    )

    if extreme_count < 100:

        raise ValueError(
            "Too few extreme samples "
            "for residual model."
        )

    X_extreme = X_train.loc[
        extreme_mask
    ]

    y_extreme_residual = (
        train_residual[
            extreme_mask
        ]
    )

    print(
        f"Residual mean: "
        f"{y_extreme_residual.mean():.4f}"
    )

    print(
        f"Residual median: "
        f"{np.median(y_extreme_residual):.4f}"
    )

    print(
        f"Residual std: "
        f"{y_extreme_residual.std():.4f}"
    )

    # ======================================================
    # STEP 3
    # Train residual model
    # ======================================================

    print()
    print("=" * 70)
    print(
        "STEP 3 - TRAINING EXTREME RESIDUAL MODEL"
    )
    print("=" * 70)

    residual_model = (
        create_residual_model()
    )

    start = time.time()

    residual_model.fit(
        X_extreme,
        y_extreme_residual
    )

    residual_training_time = (
        time.time() - start
    )

    print(
        f"Training time: "
        f"{residual_training_time:.2f}s"
    )

    # ======================================================
    # STEP 4
    # Validation correction
    # ======================================================

    print()
    print("=" * 70)
    print(
        "STEP 4 - VALIDATION EXTREME CORRECTION"
    )
    print("=" * 70)

    validation_extreme_residual = (
        residual_model.predict(
            X_validation
        )
    )

    validation_actual_extreme = (
        y_validation.to_numpy()
        > EXTREME_THRESHOLD
    )

    # ------------------------------------------------------
    # Test several correction strengths
    # ------------------------------------------------------

    correction_strengths = [
        0.25,
        0.50,
        0.75,
        1.00
    ]

    validation_results = []

    for strength in correction_strengths:

        corrected = (
            validation_base.copy()
        )

        corrected[
            validation_actual_extreme
        ] += (

            strength
            * validation_extreme_residual[
                validation_actual_extreme
            ]

        )

        metrics = calculate_metrics(
            y_validation,
            corrected
        )

        validation_results.append({

            "strength":
                strength,

            "mae":
                metrics["mae"],

            "rmse":
                metrics["rmse"],

            "r2":
                metrics["r2"],

            "within_10":
                metrics["within_10"],

            "within_20":
                metrics["within_20"],

            "within_30":
                metrics["within_30"]
        })

        print()
        print(
            f"Correction strength: "
            f"{strength:.2f}"
        )

        print(
            f"MAE  : "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"RMSE : "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"R²   : "
            f"{metrics['r2']:.4f}"
        )

    validation_results_df = pd.DataFrame(
        validation_results
    )

    # ------------------------------------------------------
    # Select based on validation MAE
    # ------------------------------------------------------

    best_row = (
        validation_results_df
        .sort_values("mae")
        .iloc[0]
    )

    best_strength = (
        float(
            best_row["strength"]
        )
    )

    print()
    print("=" * 70)
    print(
        "BEST VALIDATION CORRECTION"
    )
    print("=" * 70)

    print(
        f"Strength: "
        f"{best_strength:.2f}"
    )

    print(
        f"MAE: "
        f"{best_row['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{best_row['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{best_row['r2']:.4f}"
    )

    # ======================================================
    # STEP 5
    # Apply correction to 2026
    # ======================================================

    print()
    print("=" * 70)
    print(
        "STEP 5 - 2026 HOLDOUT EVALUATION"
    )
    print("=" * 70)

    test_residual_prediction = (
        residual_model.predict(
            X_test
        )
    )

    test_actual_extreme = (
        y_test.to_numpy()
        > EXTREME_THRESHOLD
    )

    corrected_test = (
        test_base.copy()
    )

    corrected_test[
        test_actual_extreme
    ] += (

        best_strength
        * test_residual_prediction[
            test_actual_extreme
        ]

    )

    corrected_metrics = calculate_metrics(
        y_test,
        corrected_test
    )

    print()
    print("BASE XGBOOST")

    print(
        f"MAE  : "
        f"{test_base_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{test_base_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
        f"{test_base_metrics['r2']:.4f}"
    )

    print()
    print(
        "EXTREME RESIDUAL CORRECTED"
    )

    print(
        f"MAE  : "
        f"{corrected_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{corrected_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
        f"{corrected_metrics['r2']:.4f}"
    )

    print(
        f"Within ±10: "
        f"{corrected_metrics['within_10']:.2f}%"
    )

    print(
        f"Within ±20: "
        f"{corrected_metrics['within_20']:.2f}%"
    )

    print(
        f"Within ±30: "
        f"{corrected_metrics['within_30']:.2f}%"
    )

    # ------------------------------------------------------
    # Improvement
    # ------------------------------------------------------

    mae_change = (
        test_base_metrics["mae"]
        - corrected_metrics["mae"]
    )

    rmse_change = (
        test_base_metrics["rmse"]
        - corrected_metrics["rmse"]
    )

    print()
    print(
        f"MAE improvement: "
        f"{mae_change:+.4f}"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_change:+.4f}"
    )

    # ======================================================
    # STEP 6
    # Extreme-only metrics
    # ======================================================

    print()
    print("=" * 70)
    print(
        "EXTREME-ONLY PERFORMANCE"
    )
    print("=" * 70)

    extreme_test_count = (
        test_actual_extreme.sum()
    )

    print(
        f"Extreme test rows: "
        f"{extreme_test_count}"
    )

    if extreme_test_count > 0:

        base_extreme_metrics = (
            calculate_metrics(

                y_test[
                    test_actual_extreme
                ],

                test_base[
                    test_actual_extreme
                ]

            )
        )

        corrected_extreme_metrics = (
            calculate_metrics(

                y_test[
                    test_actual_extreme
                ],

                corrected_test[
                    test_actual_extreme
                ]

            )
        )

        print()
        print(
            "BASE XGBOOST - EXTREME"
        )

        print(
            f"MAE: "
            f"{base_extreme_metrics['mae']:.4f}"
        )

        print(
            f"RMSE: "
            f"{base_extreme_metrics['rmse']:.4f}"
        )

        print()
        print(
            "CORRECTED - EXTREME"
        )

        print(
            f"MAE: "
            f"{corrected_extreme_metrics['mae']:.4f}"
        )

        print(
            f"RMSE: "
            f"{corrected_extreme_metrics['rmse']:.4f}"
        )

    # ======================================================
    # STEP 7
    # Worst predictions
    # ======================================================

    predictions_df = test_df[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi"
        ]
    ].copy()

    predictions_df[
        "base_prediction"
    ] = test_base

    predictions_df[
        "corrected_prediction"
    ] = corrected_test

    predictions_df[
        "base_absolute_error"
    ] = np.abs(

        predictions_df["target_aqi"]
        - predictions_df[
            "base_prediction"
        ]

    )

    predictions_df[
        "corrected_absolute_error"
    ] = np.abs(

        predictions_df["target_aqi"]
        - predictions_df[
            "corrected_prediction"
        ]

    )

    predictions_df = (
        predictions_df
        .sort_values(
            "corrected_absolute_error",
            ascending=False
        )
    )

    print()
    print(
        predictions_df.head(20).to_string(
            index=False
        )
    )

    # ======================================================
    # Save
    # ======================================================

    base_model.save_model(
        BASE_MODEL_FILE
    )

    residual_model.save_model(
        RESIDUAL_MODEL_FILE
    )

    validation_results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    predictions_df.to_parquet(
        PREDICTION_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Base model: "
        f"{BASE_MODEL_FILE}"
    )

    print(
        f"Residual model: "
        f"{RESIDUAL_MODEL_FILE}"
    )

    print(
        f"Validation results: "
        f"{RESULT_FILE}"
    )

    print(
        f"Predictions: "
        f"{PREDICTION_FILE}"
    )


if __name__ == "__main__":

    main()