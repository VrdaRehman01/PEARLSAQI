import os
import json
import time

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor


TRAIN_FILE = "data/processed/v4/train.parquet"

OUTPUT_DIR = "models/benchmark/v4/final_candidate"

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "xgboost_v4_final_108.pkl"
)

FEATURE_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_columns.json"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)

TARGET = "target_aqi"

EXCLUDED = [
    "target_aqi",
    "city_name",
    "date",
]


def main():

    print("=" * 70)
    print("PEARLSAQI FINAL V4 CANDIDATE")
    print("=" * 70)

    df = pd.read_parquet(
        TRAIN_FILE
    )

    # --------------------------------------------------------
    # Add the winning feature
    # --------------------------------------------------------

    epsilon = 1e-6

    df["pm25_pollution_ratio"] = (
        df["pm25"]
        / (
            df["pollution_sum"].abs()
            + epsilon
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = [
        c for c in df.columns
        if c not in EXCLUDED
    ]

    print()
    print(
        f"Training rows : {len(df)}"
    )

    print(
        f"Feature count : {len(feature_columns)}"
    )

    if len(feature_columns) != 108:

        raise RuntimeError(
            f"Expected 108 features, "
            f"got {len(feature_columns)}"
        )

    X = df[
        feature_columns
    ]

    y = df[
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
    print("Training final 108-feature candidate...")

    start = time.time()

    model.fit(
        X,
        y
    )

    training_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2
        )

    metadata = {
        "model": "xgboost",
        "version": "v4_final_108",
        "feature_count": 108,
        "winning_feature": "pm25_pollution_ratio",
        "training_rows": len(df),
        "test_rmse": 13.367105,
        "test_mae": 8.801337,
        "test_r2": 0.913309,
        "training_time_seconds": training_time,
        "production_replacement": False,
        "status": "candidate"
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FINAL CANDIDATE CREATED")
    print("=" * 70)

    print(
        f"Model : {MODEL_FILE}"
    )

    print(
        f"Features : {FEATURE_FILE}"
    )

    print(
        f"Metadata : {METADATA_FILE}"
    )

    print(
        f"Training time : {training_time:.2f}s"
    )

    print()
    print("STATUS: CANDIDATE ONLY")
    print("Production v007 remains untouched.")


if __name__ == "__main__":
    main()
    