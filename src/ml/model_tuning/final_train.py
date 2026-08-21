import os
import pandas as pd

from xgboost import XGBRegressor


TRAIN_FILE = "data/processed/train.parquet"
VALIDATION_FILE = "data/processed/validation.parquet"

MODEL_DIR = "models/xgboost"

FINAL_MODEL_FILE = (
    f"{MODEL_DIR}/final_xgboost_model.json"
)


FEATURE_COLUMNS = [
    "city_id",

    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",

    "temperature",
    "humidity",
    "precipitation",
    "windspeed",

    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_7",

    "aqi_rolling_3",
    "aqi_rolling_7",

    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
]

TARGET_COLUMN = "target_aqi"


def final_train():

    print("=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # Load training data
    # --------------------------------------------------

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    # --------------------------------------------------
    # Combine 2023-2025
    # --------------------------------------------------

    full_training_df = pd.concat(
        [
            train_df,
            validation_df
        ],
        ignore_index=True
    )

    print()
    print(
        f"Training rows: {len(full_training_df)}"
    )

    print(
        f"Start date: "
        f"{full_training_df['date'].min()}"
    )

    print(
        f"End date: "
        f"{full_training_df['date'].max()}"
    )

    # --------------------------------------------------
    # Prepare features
    # --------------------------------------------------

    X = full_training_df[
        FEATURE_COLUMNS
    ]

    y = full_training_df[
        TARGET_COLUMN
    ]

    # --------------------------------------------------
    # BEST HYPERPARAMETERS
    # Model 5
    # --------------------------------------------------

    model = XGBRegressor(

        n_estimators=900,

        learning_rate=0.025,

        max_depth=3,

        min_child_weight=8,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=42,

        n_jobs=-1
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print()
    print("Training final model...")

    model.fit(
        X,
        y,
        verbose=False
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save_model(
        FINAL_MODEL_FILE
    )

    print()
    print("=" * 60)
    print("FINAL MODEL SAVED")
    print("=" * 60)

    print(
        f"Model: {FINAL_MODEL_FILE}"
    )


if __name__ == "__main__":

    final_train()