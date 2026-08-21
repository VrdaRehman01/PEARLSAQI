import os
import time

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TRAIN_FILE = "data/processed/v4/train.parquet"
VALIDATION_FILE = "data/processed/v4/validation.parquet"

TARGET_COLUMN = "target_aqi"

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
]

RESULTS_DIR = "models/benchmark/v4"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "extratrees_v4_results.csv"
)


def main():

    print("=" * 70)
    print("PEARLSAQI V4 EXTRATREES EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------
    # Load V4 data
    # --------------------------------------------------

    print("\nLoading V4 training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print("\nLoading V4 validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # --------------------------------------------------
    # Features
    # --------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if TARGET_COLUMN in feature_columns:
        raise ValueError(
            "target_aqi accidentally included as a feature."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in validation_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing validation features: "
            f"{missing_features}"
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

    print(
        f"\nFeature count: {len(feature_columns)}"
    )

    # --------------------------------------------------
    # ExtraTrees
    # --------------------------------------------------

    model = ExtraTreesRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features=1.0,

        random_state=42,

        n_jobs=-1
    )

    print()
    print("=" * 70)
    print("TRAINING EXTRA TREES")
    print("=" * 70)

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

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

    print()
    print("=" * 70)
    print("V4 EXTRA TREES RESULTS")
    print("=" * 70)

    print(
        f"MAE              : {mae:.4f}"
    )

    print(
        f"RMSE             : {rmse:.4f}"
    )

    print(
        f"R2               : {r2:.4f}"
    )

    print(
        f"Training time    : {training_time:.2f}s"
    )

    print(
        f"Feature count    : {len(feature_columns)}"
    )

    # --------------------------------------------------
    # Compare against production
    # --------------------------------------------------

    production_rmse = 14.379474258005216
    production_mae = 9.02976086063472
    production_r2 = 0.9013127340131636

    print()
    print("=" * 70)
    print("PRODUCTION XGBOOST V007 COMPARISON")
    print("=" * 70)

    print(
        f"XGBoost v007 RMSE : {production_rmse:.4f}"
    )

    print(
        f"ExtraTrees RMSE   : {rmse:.4f}"
    )

    print(
        f"XGBoost v007 MAE  : {production_mae:.4f}"
    )

    print(
        f"ExtraTrees MAE    : {mae:.4f}"
    )

    print(
        f"XGBoost v007 R2   : {production_r2:.4f}"
    )

    print(
        f"ExtraTrees R2     : {r2:.4f}"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results = pd.DataFrame([
        {
            "model": "ExtraTrees",
            "version": "V4 experimental",
            "features": len(feature_columns),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "training_time_seconds":
                training_time,
        }
    ])

    results.to_csv(
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