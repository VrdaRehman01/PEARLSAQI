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

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

MODEL_DIR = "models/xgboost"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "v3_xgboost_aqi_model.json"
)


# ==========================================================
# Columns that must NOT be used as features
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]


TARGET_COLUMN = "target_aqi"


# ==========================================================
# Evaluation
# ==========================================================

def evaluate_model(model, X, y, dataset_name):

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

    print()
    print("=" * 60)
    print(f"{dataset_name} RESULTS")
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ==========================================================
# Training
# ==========================================================

def train_model():

    print("=" * 60)
    print("XGBOOST AQI MODEL TRAINING - V3")
    print("=" * 60)

    # ------------------------------------------------------
    # Load training data
    # ------------------------------------------------------

    print()
    print("Loading V3 training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    # ------------------------------------------------------
    # Load validation data
    # ------------------------------------------------------

    print()
    print("Loading V3 validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ------------------------------------------------------
    # Check columns
    # ------------------------------------------------------

    print()
    print("Building feature list...")

    FEATURE_COLUMNS = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    # ------------------------------------------------------
    # Safety check
    # ------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in validation_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Features missing from validation data: "
            f"{missing_features}"
        )

    if TARGET_COLUMN in FEATURE_COLUMNS:

        raise ValueError(
            "target_aqi cannot be used as a feature."
        )

    # ------------------------------------------------------
    # Check for non-numeric features
    # ------------------------------------------------------

    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(
            train_df[column]
        )
    ]

    if non_numeric:

        raise ValueError(
            "Non-numeric features detected: "
            f"{non_numeric}"
        )

    # ------------------------------------------------------
    # Create X / y
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
    # Feature information
    # ------------------------------------------------------

    print()
    print(
        f"Number of features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print()
    print("Features being used:")

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1
    ):

        print(
            f"{index:02d}. {feature}"
        )

    # ------------------------------------------------------
    # Create model
    # ------------------------------------------------------

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

        n_jobs=-1
    )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING V3 MODEL")
    print("=" * 60)

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (X_train, y_train),
            (X_validation, y_validation)
        ],

        verbose=False
    )

    print()
    print("Training completed.")

    # ------------------------------------------------------
    # Training evaluation
    # ------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "V3 TRAINING"
    )

    # ------------------------------------------------------
    # Validation evaluation
    # ------------------------------------------------------

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "V3 VALIDATION"
    )

    # ------------------------------------------------------
    # Save model
    # ------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save_model(
        MODEL_FILE
    )

    print()
    print("=" * 60)
    print("V3 MODEL SAVED")
    print("=" * 60)

    print(
        f"Model path: {MODEL_FILE}"
    )

    # ------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------

    importance = pd.DataFrame({

        "feature":
            FEATURE_COLUMNS,

        "importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("=" * 60)
    print("V3 FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------

    V2_MAE = 13.1605
    V2_RMSE = 20.1657
    V2_R2 = 0.8227

    print()
    print("=" * 60)
    print("V2 vs V3 BENCHMARK")
    print("=" * 60)

    print(
        f"V2 Validation MAE  : {V2_MAE:.4f}"
    )

    print(
        f"V3 Validation MAE  : "
        f"{validation_metrics['mae']:.4f}"
    )

    print()

    print(
        f"V2 Validation RMSE : {V2_RMSE:.4f}"
    )

    print(
        f"V3 Validation RMSE : "
        f"{validation_metrics['rmse']:.4f}"
    )

    print()

    print(
        f"V2 Validation R²   : {V2_R2:.4f}"
    )

    print(
        f"V3 Validation R²   : "
        f"{validation_metrics['r2']:.4f}"
    )

    # ------------------------------------------------------
    # Improvement
    # ------------------------------------------------------

    mae_improvement = (
        (V2_MAE - validation_metrics["mae"])
        / V2_MAE
        * 100
    )

    rmse_improvement = (
        (V2_RMSE - validation_metrics["rmse"])
        / V2_RMSE
        * 100
    )

    r2_improvement = (
        validation_metrics["r2"]
        - V2_R2
    )

    print()
    print("=" * 60)
    print("V3 IMPROVEMENT")
    print("=" * 60)

    print(
        f"MAE improvement  : "
        f"{mae_improvement:+.2f}%"
    )

    print(
        f"RMSE improvement : "
        f"{rmse_improvement:+.2f}%"
    )

    print(
        f"R² improvement   : "
        f"{r2_improvement:+.4f}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_model()