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

TRAIN_FILE = "data/processed/v4/train.parquet"
VALIDATION_FILE = "data/processed/v4/validation.parquet"

MODEL_DIR = "models/v4"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_aqi_model.json"
)


# ==========================================================
# Metrics
# ==========================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

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
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ==========================================================
# Training
# ==========================================================

def train_model():

    print("=" * 60)
    print("XGBOOST AQI MODEL TRAINING - V4")
    print("=" * 60)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    print()
    print("Loading V4 training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print()
    print("Loading V4 validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------

    EXCLUDED_COLUMNS = [
        "target_aqi",
        "city_name",
        "date",
    ]

    TARGET_COLUMN = "target_aqi"

    FEATURE_COLUMNS = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    # ------------------------------------------------------
    # Safety checks
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
    # X / y
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
    # Model
    # ------------------------------------------------------

    model = XGBRegressor(

        n_estimators=700,

        learning_rate=0.035,

        max_depth=6,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=2.0,

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
    print("TRAINING V4 MODEL")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_train,
                y_train
            ),
            (
                X_validation,
                y_validation
            )
        ],

        verbose=False
    )

    print()
    print("Training completed.")

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "V4 TRAINING"
    )

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "V4 VALIDATION"
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
    print("V4 MODEL SAVED")
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

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print()
    print("=" * 60)
    print("V4 FEATURE IMPORTANCE - TOP 40")
    print("=" * 60)

    print(
        importance
        .head(40)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Compare against V3
    # ------------------------------------------------------

    v3_mae = 13.2705
    v3_rmse = 19.9986
    v3_r2 = 0.8256

    mae_change = (
        (
            v3_mae
            -
            validation_metrics["mae"]
        )
        /
        v3_mae
    ) * 100

    rmse_change = (
        (
            v3_rmse
            -
            validation_metrics["rmse"]
        )
        /
        v3_rmse
    ) * 100

    r2_change = (
        validation_metrics["r2"]
        -
        v3_r2
    )

    print()
    print("=" * 60)
    print("V3 → V4 VALIDATION COMPARISON")
    print("=" * 60)

    print()
    print(
        f"V3 MAE : {v3_mae:.4f}"
    )

    print(
        f"V4 MAE : "
        f"{validation_metrics['mae']:.4f}"
    )

    print(
        f"MAE improvement: "
        f"{mae_change:+.2f}%"
    )

    print()
    print(
        f"V3 RMSE : {v3_rmse:.4f}"
    )

    print(
        f"V4 RMSE : "
        f"{validation_metrics['rmse']:.4f}"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_change:+.2f}%"
    )

    print()
    print(
        f"V3 R² : {v3_r2:.4f}"
    )

    print(
        f"V4 R² : "
        f"{validation_metrics['r2']:.4f}"
    )

    print(
        f"R² improvement: "
        f"{r2_change:+.4f}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_model()