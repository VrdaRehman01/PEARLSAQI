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

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_aqi_model.json"
)


# ==========================================================
# Feature Configuration
# ==========================================================

# These columns should NOT be given to XGBoost.

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]

TARGET_COLUMN = "target_aqi"


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
    print("XGBOOST AQI MODEL TRAINING - V2 FEATURES")
    print("=" * 60)

    # ------------------------------------------------------
    # Load training dataset
    # ------------------------------------------------------

    print()
    print("Loading training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    # ------------------------------------------------------
    # Load validation dataset
    # ------------------------------------------------------

    print()
    print("Loading validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ======================================================
    # CREATE FEATURE LIST
    # ======================================================

    print()
    print("=" * 60)
    print("BUILDING FEATURE LIST")
    print("=" * 60)

    FEATURE_COLUMNS = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    print()
    print(
        f"Number of features: {len(FEATURE_COLUMNS)}"
    )

    # ------------------------------------------------------
    # Check that all features exist in validation
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

    # ------------------------------------------------------
    # Make sure target is NOT a feature
    # ------------------------------------------------------

    if TARGET_COLUMN in FEATURE_COLUMNS:

        raise ValueError(
            "target_aqi cannot be used as a model feature."
        )

    # ------------------------------------------------------
    # Check for non-numeric features
    # ------------------------------------------------------

    non_numeric_features = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(
            train_df[column]
        )
    ]

    if non_numeric_features:

        raise ValueError(
            "Non-numeric features found: "
            f"{non_numeric_features}"
        )

    # ======================================================
    # PRINT FEATURES
    # ======================================================

    print()
    print("Features being used:")

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1
    ):

        print(
            f"{index:02d}. {feature}"
        )

    # ======================================================
    # CREATE X / Y
    # ======================================================

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

    # ======================================================
    # CREATE MODEL
    # ======================================================

    print()
    print("=" * 60)
    print("CREATING XGBOOST MODEL")
    print("=" * 60)

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

    # ======================================================
    # TRAIN
    # ======================================================

    print()
    print("=" * 60)
    print("TRAINING MODEL")
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

    # ======================================================
    # TRAINING EVALUATION
    # ======================================================

    train_metrics = evaluate_model(

        model,

        X_train,

        y_train,

        "TRAINING"
    )

    # ======================================================
    # VALIDATION EVALUATION
    # ======================================================

    validation_metrics = evaluate_model(

        model,

        X_validation,

        y_validation,

        "VALIDATION"
    )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save_model(
        MODEL_FILE
    )

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Model path: {MODEL_FILE}"
    )

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

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
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print()
    print("=" * 60)
    print("V2 TRAINING SUMMARY")
    print("=" * 60)

    print()
    print("Training:")
    print(
        f"  MAE  : {train_metrics['mae']:.4f}"
    )
    print(
        f"  RMSE : {train_metrics['rmse']:.4f}"
    )
    print(
        f"  R²   : {train_metrics['r2']:.4f}"
    )

    print()
    print("Validation:")
    print(
        f"  MAE  : {validation_metrics['mae']:.4f}"
    )
    print(
        f"  RMSE : {validation_metrics['rmse']:.4f}"
    )
    print(
        f"  R²   : {validation_metrics['r2']:.4f}"
    )

    print()
    print("V1 tuned benchmark:")
    print(
        "  MAE  : 13.4753"
    )
    print(
        "  RMSE : 20.2782"
    )
    print(
        "  R²   : 0.8207"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_model()