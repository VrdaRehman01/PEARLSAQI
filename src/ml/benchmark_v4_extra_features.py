import os
import time

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TRAIN_FILE = "data/processed/v4/train.parquet"
VALIDATION_FILE = "data/processed/v4/validation.parquet"

TARGET = "target_aqi"

EXCLUDED = [
    "target_aqi",
    "city_name",
    "date",
]

RESULTS_DIR = "models/benchmark/v4"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "extra_features_xgboost_results.csv",
)


# ============================================================
# EXTRA FEATURES
# ============================================================

def add_extra_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Weather rolling features
    # --------------------------------------------------------

    grouped = df.groupby("city_id", group_keys=False)

    for column in [
        "temperature",
        "humidity",
        "windspeed",
    ]:

        df[f"{column}_rolling_3"] = (
            grouped[column]
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{column}_rolling_7"] = (
            grouped[column]
            .shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{column}_std_3"] = (
            grouped[column]
            .shift(1)
            .rolling(3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # --------------------------------------------------------
    # Pollutant volatility
    # --------------------------------------------------------

    pollutants = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
    ]

    for column in pollutants:

        for window in [3, 7]:

            df[f"{column}_std_{window}"] = (
                grouped[column]
                .shift(1)
                .rolling(window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

    # --------------------------------------------------------
    # Additional pollutant ratios
    # --------------------------------------------------------

    epsilon = 1e-6

    df["so2_no2_ratio"] = (
        df["so2"] /
        (df["no2"].abs() + epsilon)
    )

    df["o3_no2_ratio"] = (
        df["o3"] /
        (df["no2"].abs() + epsilon)
    )

    df["pm25_pollution_ratio"] = (
        df["pm25"] /
        (df["pollution_sum"].abs() + epsilon)
    )

    df["pm10_pollution_ratio"] = (
        df["pm10"] /
        (df["pollution_sum"].abs() + epsilon)
    )

    # --------------------------------------------------------
    # Clean numerical problems
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLSAQI V4 EXTRA FEATURE EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train = pd.read_parquet(
        TRAIN_FILE
    )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    print()
    print(
        "Original feature count:",
        len([
            c for c in train.columns
            if c not in EXCLUDED
        ])
    )

    # --------------------------------------------------------
    # Add experimental features
    # --------------------------------------------------------

    train = add_extra_features(
        train
    )

    validation = add_extra_features(
        validation
    )

    feature_columns = [
        c for c in train.columns
        if c not in EXCLUDED
    ]

    print(
        "Experimental feature count:",
        len(feature_columns)
    )

    print(
        "New features:",
        len(feature_columns) - 107
    )

    # --------------------------------------------------------
    # Ensure matching schemas
    # --------------------------------------------------------

    missing = [
        c for c in feature_columns
        if c not in validation.columns
    ]

    if missing:

        raise RuntimeError(
            f"Validation missing features: {missing}"
        )

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET
    ]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET
    ]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = XGBRegressor(

        n_estimators=500,

        learning_rate=0.05,

        max_depth=6,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=42,

        n_jobs=-1,
    )

    print()
    print("=" * 70)
    print("TRAINING XGBOOST WITH EXTRA FEATURES")
    print("=" * 70)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = model.predict(
        X_validation
    )

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    rmse = mean_squared_error(
        y_validation,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_validation,
        predictions
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    production_rmse = 14.379474258005216
    production_mae = 9.02976086063472
    production_r2 = 0.9013127340131636

    print()
    print("=" * 70)
    print("EXTRA FEATURE RESULTS")
    print("=" * 70)

    print(
        f"Features             : {len(feature_columns)}"
    )

    print(
        f"New features         : {len(feature_columns) - 107}"
    )

    print(
        f"RMSE                 : {rmse:.4f}"
    )

    print(
        f"MAE                  : {mae:.4f}"
    )

    print(
        f"R2                   : {r2:.4f}"
    )

    print(
        f"Training time        : {training_time:.2f}s"
    )

    print()
    print("=" * 70)
    print("XGBOOST V007 BASELINE")
    print("=" * 70)

    print(
        f"RMSE                 : {production_rmse:.4f}"
    )

    print(
        f"MAE                  : {production_mae:.4f}"
    )

    print(
        f"R2                   : {production_r2:.4f}"
    )

    print()
    print("=" * 70)
    print("DELTA VS PRODUCTION")
    print("=" * 70)

    print(
        f"RMSE change          : "
        f"{rmse - production_rmse:+.4f}"
    )

    print(
        f"MAE change           : "
        f"{mae - production_mae:+.4f}"
    )

    print(
        f"R2 change            : "
        f"{r2 - production_r2:+.4f}"
    )

    # --------------------------------------------------------
    # Save experiment result
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    pd.DataFrame([
        {
            "model": "XGBoost",
            "experiment": "extra_features",
            "feature_count": len(feature_columns),
            "new_features": len(feature_columns) - 107,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "training_time_seconds": training_time,
        }
    ]).to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()