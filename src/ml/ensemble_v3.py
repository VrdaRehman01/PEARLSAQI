import os
import time
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

OUTPUT_DIR = "models/ensemble"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "ensemble_v3_results.csv"
)


TARGET_COLUMN = "target_aqi"


EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]


# ==========================================================
# Metrics
# ==========================================================

def evaluate(
    name,
    predictions,
    y
):

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
    print(
        f"{name}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("AQI ENSEMBLE MODEL - V3")
    print("=" * 60)

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

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    print()
    print(
        f"Number of features: {len(feature_columns)}"
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

    # ======================================================
    # MODEL 1 — HISTGRADIENTBOOSTING
    # ======================================================

    print()
    print("=" * 60)
    print("TRAINING HISTGRADIENTBOOSTING")
    print("=" * 60)

    hgb = HistGradientBoostingRegressor(

        learning_rate=0.03,

        max_iter=500,

        max_leaf_nodes=15,

        max_depth=None,

        min_samples_leaf=30,

        l2_regularization=5.0,

        random_state=42
    )

    start = time.time()

    hgb.fit(
        X_train,
        y_train
    )

    print(
        f"Training time: {time.time() - start:.2f}s"
    )

    hgb_pred = hgb.predict(
        X_validation
    )

    # ======================================================
    # MODEL 2 — XGBOOST
    # ======================================================

    print()
    print("=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

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

    start = time.time()

    xgb.fit(
        X_train,
        y_train
    )

    print(
        f"Training time: {time.time() - start:.2f}s"
    )

    xgb_pred = xgb.predict(
        X_validation
    )

    # ======================================================
    # MODEL 3 — RIDGE
    # ======================================================

    print()
    print("=" * 60)
    print("TRAINING RIDGE")
    print("=" * 60)

    ridge = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "ridge",
            Ridge(
                alpha=0.01
            )
        )

    ])

    start = time.time()

    ridge.fit(
        X_train,
        y_train
    )

    print(
        f"Training time: {time.time() - start:.2f}s"
    )

    ridge_pred = ridge.predict(
        X_validation
    )

    # ======================================================
    # INDIVIDUAL MODELS
    # ======================================================

    print()
    print("=" * 60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 60)

    results = []

    results.append(
        evaluate(
            "HistGradientBoosting",
            hgb_pred,
            y_validation
        )
    )

    results.append(
        evaluate(
            "XGBoost",
            xgb_pred,
            y_validation
        )
    )

    results.append(
        evaluate(
            "Ridge",
            ridge_pred,
            y_validation
        )
    )

    # ======================================================
    # TWO-MODEL ENSEMBLES
    # ======================================================

    print()
    print("=" * 60)
    print("TWO-MODEL ENSEMBLES")
    print("=" * 60)

    # ------------------------------------------------------
    # HGB + XGBoost
    # ------------------------------------------------------

    weights_hgb_xgb = [
        (0.25, 0.75),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.75, 0.25),
    ]

    for hgb_weight, xgb_weight in weights_hgb_xgb:

        prediction = (
            hgb_weight * hgb_pred
            +
            xgb_weight * xgb_pred
        )

        name = (
            f"HGB_XGB_"
            f"{hgb_weight:.2f}_"
            f"{xgb_weight:.2f}"
        )

        results.append(
            evaluate(
                name,
                prediction,
                y_validation
            )
        )

    # ------------------------------------------------------
    # HGB + Ridge
    # ------------------------------------------------------

    weights_hgb_ridge = [
        (0.25, 0.75),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.75, 0.25),
    ]

    for hgb_weight, ridge_weight in weights_hgb_ridge:

        prediction = (
            hgb_weight * hgb_pred
            +
            ridge_weight * ridge_pred
        )

        name = (
            f"HGB_Ridge_"
            f"{hgb_weight:.2f}_"
            f"{ridge_weight:.2f}"
        )

        results.append(
            evaluate(
                name,
                prediction,
                y_validation
            )
        )

    # ------------------------------------------------------
    # XGBoost + Ridge
    # ------------------------------------------------------

    weights_xgb_ridge = [
        (0.25, 0.75),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.75, 0.25),
    ]

    for xgb_weight, ridge_weight in weights_xgb_ridge:

        prediction = (
            xgb_weight * xgb_pred
            +
            ridge_weight * ridge_pred
        )

        name = (
            f"XGB_Ridge_"
            f"{xgb_weight:.2f}_"
            f"{ridge_weight:.2f}"
        )

        results.append(
            evaluate(
                name,
                prediction,
                y_validation
            )
        )

    # ======================================================
    # THREE-MODEL ENSEMBLES
    # ======================================================

    print()
    print("=" * 60)
    print("THREE-MODEL ENSEMBLES")
    print("=" * 60)

    weights_three = [

        (0.50, 0.30, 0.20),

        (0.50, 0.25, 0.25),

        (0.40, 0.40, 0.20),

        (0.40, 0.30, 0.30),

        (0.60, 0.20, 0.20),

        (0.60, 0.30, 0.10),

        (0.70, 0.20, 0.10),

    ]

    for (
        hgb_weight,
        xgb_weight,
        ridge_weight
    ) in weights_three:

        prediction = (

            hgb_weight * hgb_pred

            +

            xgb_weight * xgb_pred

            +

            ridge_weight * ridge_pred

        )

        name = (

            f"HGB_XGB_Ridge_"

            f"{hgb_weight:.2f}_"

            f"{xgb_weight:.2f}_"

            f"{ridge_weight:.2f}"

        )

        results.append(
            evaluate(
                name,
                prediction,
                y_validation
            )
        )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "mae"
    )

    print()
    print("=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    # ======================================================
    # BEST ENSEMBLE
    # ======================================================

    best = results_df.iloc[0]

    print()
    print("=" * 60)
    print("BEST ENSEMBLE")
    print("=" * 60)

    print(
        f"Model : {best['model']}"
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


if __name__ == "__main__":

    main()