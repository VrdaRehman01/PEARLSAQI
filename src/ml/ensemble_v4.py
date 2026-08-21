import os
import time
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

CLASSIFIER_FILE = (
    "models/extreme_classifier/"
    "extreme_aqi_classifier.pkl"
)

OUTPUT_DIR = "models/ensemble_v4"

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "ensemble_v4_validation_results.csv"
)

EXTREME_THRESHOLD = 200
EXTREME_PROBABILITY_THRESHOLD = 0.40


# ==========================================================
# FEATURES
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
]


# ==========================================================
# METRICS
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

    return mae, rmse, r2


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    print("=" * 60)
    print("EXTREME-AWARE AQI ENSEMBLE - V4")
    print("=" * 60)

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

    return train_df, validation_df


# ==========================================================
# FEATURES
# ==========================================================

def build_features(
    train_df,
    validation_df
):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    missing = [
        column
        for column in feature_columns
        if column not in validation_df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing validation features: {missing}"
        )

    if "target_aqi" in feature_columns:

        raise ValueError(
            "target_aqi cannot be used as a feature."
        )

    print()
    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    return feature_columns


# ==========================================================
# TRAIN REGRESSION MODELS
# ==========================================================

def train_models(
    X_train,
    y_train
):

    print()
    print("=" * 60)
    print("TRAINING REGRESSION MODELS")
    print("=" * 60)

    # ------------------------------------------------------
    # HistGradientBoosting
    # ------------------------------------------------------

    print()
    print("Training HistGradientBoosting...")

    start = time.time()

    hgb = HistGradientBoostingRegressor(

        learning_rate=0.03,

        max_iter=500,

        max_leaf_nodes=15,

        min_samples_leaf=30,

        l2_regularization=5.0,

        random_state=42
    )

    hgb.fit(
        X_train,
        y_train
    )

    print(
        f"Training time: "
        f"{time.time() - start:.2f}s"
    )

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    print()
    print("Training XGBoost...")

    start = time.time()

    xgb = XGBRegressor(

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

    xgb.fit(
        X_train,
        y_train,
        verbose=False
    )

    print(
        f"Training time: "
        f"{time.time() - start:.2f}s"
    )

    # ------------------------------------------------------
    # Ridge
    # ------------------------------------------------------

    print()
    print("Training Ridge...")

    start = time.time()

    ridge = make_pipeline(

        StandardScaler(),

        Ridge(
            alpha=0.01
        )
    )

    ridge.fit(
        X_train,
        y_train
    )

    print(
        f"Training time: "
        f"{time.time() - start:.2f}s"
    )

    return hgb, xgb, ridge


# ==========================================================
# EXTREME CLASSIFIER
# ==========================================================

def load_extreme_classifier():

    print()
    print(
        "Loading extreme-event classifier..."
    )

    saved = joblib.load(
        CLASSIFIER_FILE
    )

    classifier = saved["model"]

    classifier_features = saved[
        "features"
    ]

    print(
        f"Classifier features: "
        f"{len(classifier_features)}"
    )

    return classifier, classifier_features


# ==========================================================
# ENSEMBLE TESTING
# ==========================================================

def evaluate_ensemble_candidates(
    y_true,
    hgb_pred,
    xgb_pred,
    ridge_pred,
    extreme_probability
):

    results = []

    # ------------------------------------------------------
    # Basic model predictions
    # ------------------------------------------------------

    predictions = {

        "HGB":
            hgb_pred,

        "XGB":
            xgb_pred,

        "Ridge":
            ridge_pred,
    }

    # ------------------------------------------------------
    # Basic ensembles
    # ------------------------------------------------------

    predictions[
        "HGB_XGB_50_50"
    ] = (
        0.50 * hgb_pred
        +
        0.50 * xgb_pred
    )

    predictions[
        "HGB_XGB_Ridge_40_40_20"
    ] = (
        0.40 * hgb_pred
        +
        0.40 * xgb_pred
        +
        0.20 * ridge_pred
    )

    predictions[
        "HGB_XGB_Ridge_50_25_25"
    ] = (
        0.50 * hgb_pred
        +
        0.25 * xgb_pred
        +
        0.25 * ridge_pred
    )

    # ------------------------------------------------------
    # Extreme-aware candidates
    # ------------------------------------------------------

    base_predictions = {

        "ExtremeAware_40_40_20":
            (
                0.40 * hgb_pred
                +
                0.40 * xgb_pred
                +
                0.20 * ridge_pred
            ),

        "ExtremeAware_50_25_25":
            (
                0.50 * hgb_pred
                +
                0.25 * xgb_pred
                +
                0.25 * ridge_pred
            ),

        "ExtremeAware_HGB":
            hgb_pred,

        "ExtremeAware_XGB":
            xgb_pred,
    }

    # ------------------------------------------------------
    # Test several correction strengths
    # ------------------------------------------------------

    correction_strengths = [
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
    ]

    extreme_signal = np.maximum(
        extreme_probability
        - EXTREME_PROBABILITY_THRESHOLD,
        0
    )

    for base_name, base_prediction in (
        base_predictions.items()
    ):

        for strength in correction_strengths:

            corrected = (
                base_prediction
                +
                strength
                *
                extreme_signal
            )

            predictions[
                f"{base_name}_Correction_{strength}"
            ] = corrected

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    for name, prediction in predictions.items():

        mae, rmse, r2 = calculate_metrics(
            y_true,
            prediction
        )

        results.append({

            "model": name,

            "mae": mae,

            "rmse": rmse,

            "r2": r2
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "mae",
            "rmse"
        ],
        ascending=True
    )

    return results_df


# ==========================================================
# EXTREME EVENT PERFORMANCE
# ==========================================================

def evaluate_extreme_performance(
    y_true,
    prediction
):

    actual_extreme = (
        y_true > EXTREME_THRESHOLD
    )

    normal = ~actual_extreme

    if actual_extreme.sum() > 0:

        extreme_mae = mean_absolute_error(
            y_true[actual_extreme],
            prediction[actual_extreme]
        )

    else:

        extreme_mae = np.nan

    if normal.sum() > 0:

        normal_mae = mean_absolute_error(
            y_true[normal],
            prediction[normal]
        )

    else:

        normal_mae = np.nan

    return (
        extreme_mae,
        normal_mae
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    train_df, validation_df = load_data()

    feature_columns = build_features(
        train_df,
        validation_df
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "target_aqi"
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        "target_aqi"
    ]

    # ------------------------------------------------------
    # Train regression models
    # ------------------------------------------------------

    hgb, xgb, ridge = train_models(
        X_train,
        y_train
    )

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    print()
    print(
        "Generating regression predictions..."
    )

    hgb_pred = hgb.predict(
        X_validation
    )

    xgb_pred = xgb.predict(
        X_validation
    )

    ridge_pred = ridge.predict(
        X_validation
    )

    # ------------------------------------------------------
    # Extreme classifier
    # ------------------------------------------------------

    classifier, classifier_features = (
        load_extreme_classifier()
    )

    X_classifier = validation_df[
        classifier_features
    ]

    extreme_probability = (
        classifier
        .predict_proba(X_classifier)[:, 1]
    )

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    results_df = evaluate_ensemble_candidates(

        y_validation,

        hgb_pred,

        xgb_pred,

        ridge_pred,

        extreme_probability
    )

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Best model
    # ------------------------------------------------------

    best = results_df.iloc[0]

    best_name = best["model"]

    print()
    print("=" * 60)
    print("BEST VALIDATION MODEL")
    print("=" * 60)

    print(
        f"Model : {best_name}"
    )

    print(
        f"MAE   : {best['mae']:.4f}"
    )

    print(
        f"RMSE  : {best['rmse']:.4f}"
    )

    print(
        f"R²    : {best['r2']:.4f}"
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()
    