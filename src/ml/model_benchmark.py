import os
import time
import warnings

import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

RESULTS_DIR = "models/benchmark"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "model_benchmark_results.csv"
)


# ==========================================================
# Columns
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
]

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Metrics
# ==========================================================

def evaluate_model(model, X, y):

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
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("AQI MODEL BENCHMARK")
    print("=" * 70)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    print()
    print("Loading V3 training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print()
    print("Loading V3 validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ------------------------------------------------------
    # Build features
    # ------------------------------------------------------

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
            f"Missing validation features: {missing_features}"
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

    print()
    print(
        f"Number of features: {len(feature_columns)}"
    )

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    models = {

        "XGBoost": XGBRegressor(

            n_estimators=500,

            learning_rate=0.05,

            max_depth=6,

            min_child_weight=3,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="reg:squarederror",

            eval_metric="rmse",

            random_state=42,

            n_jobs=-1
        ),

        "Random Forest": RandomForestRegressor(

            n_estimators=300,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features="sqrt",

            random_state=42,

            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(

            n_estimators=300,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features="sqrt",

            random_state=42,

            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(

            max_iter=300,

            learning_rate=0.05,

            max_leaf_nodes=31,

            l2_regularization=1.0,

            random_state=42
        ),

        "Ridge": Ridge(

            alpha=10.0
        ),
    }

    # ------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------

    results = []

    for model_name, model in models.items():

        print()
        print("=" * 70)
        print(f"TRAINING: {model_name}")
        print("=" * 70)

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )

        mae, rmse, r2 = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        print()
        print(
            f"Validation MAE  : {mae:.4f}"
        )

        print(
            f"Validation RMSE : {rmse:.4f}"
        )

        print(
            f"Validation R²   : {r2:.4f}"
        )

        print(
            f"Training time    : {training_time:.2f}s"
        )

        results.append({

            "model": model_name,

            "mae": mae,

            "rmse": rmse,

            "r2": r2,

            "training_time_seconds":
                training_time
        })

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="mae",
        ascending=True
    )

    print()
    print("=" * 70)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    # ------------------------------------------------------
    # Best model
    # ------------------------------------------------------

    best_model = results_df.iloc[0]

    print()
    print("BEST MODEL")
    print(
        f"Model : {best_model['model']}"
    )

    print(
        f"MAE   : {best_model['mae']:.4f}"
    )

    print(
        f"RMSE  : {best_model['rmse']:.4f}"
    )

    print(
        f"R²    : {best_model['r2']:.4f}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    main()