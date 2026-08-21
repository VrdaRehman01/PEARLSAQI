import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/processed/v4/train.parquet"
VALIDATION_FILE = "data/processed/v4/validation.parquet"

RESULTS_DIR = "models/benchmark/v4"

ALL_IMPORTANCE_FILE = os.path.join(
    RESULTS_DIR,
    "extra_feature_importance_all.csv"
)

NEW_IMPORTANCE_FILE = os.path.join(
    RESULTS_DIR,
    "extra_feature_importance_new.csv"
)

TARGET = "target_aqi"

EXCLUDED = [
    "target_aqi",
    "city_name",
    "date",
]


# ============================================================
# EXTRA FEATURE ENGINEERING
# ============================================================

def add_extra_features(df):

    df = df.copy()

    grouped = df.groupby(
        "city_id",
        group_keys=False
    )

    # --------------------------------------------------------
    # Weather rolling features
    # --------------------------------------------------------

    for column in [
        "temperature",
        "humidity",
        "windspeed",
    ]:

        df[f"{column}_rolling_3"] = (
            grouped[column]
            .shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"{column}_rolling_7"] = (
            grouped[column]
            .shift(1)
            .rolling(
                7,
                min_periods=1
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"{column}_std_3"] = (
            grouped[column]
            .shift(1)
            .rolling(
                3,
                min_periods=2
            )
            .std()
            .reset_index(
                level=0,
                drop=True
            )
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
                .rolling(
                    window,
                    min_periods=2
                )
                .std()
                .reset_index(
                    level=0,
                    drop=True
                )
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
    # Clean numerical values
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
    print("PEARLSAQI EXTRA FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading V4 training data...")

    train = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train)}"
    )

    print()
    print("Loading V4 validation data...")

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    # --------------------------------------------------------
    # Original feature list
    # --------------------------------------------------------

    original_features = [
        column
        for column in train.columns
        if column not in EXCLUDED
    ]

    print()
    print(
        f"Original feature count: "
        f"{len(original_features)}"
    )

    # --------------------------------------------------------
    # Add experimental features
    # --------------------------------------------------------

    print()
    print("Adding experimental features...")

    train = add_extra_features(
        train
    )

    validation = add_extra_features(
        validation
    )

    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in train.columns
        if column not in EXCLUDED
    ]

    new_features = [
        column
        for column in feature_columns
        if column not in original_features
    ]

    print(
        f"Experimental feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"New feature count: "
        f"{len(new_features)}"
    )

    print()
    print("NEW FEATURES")
    print("-" * 70)

    for feature in new_features:
        print(feature)

    # --------------------------------------------------------
    # Validate schemas
    # --------------------------------------------------------

    missing = [
        column
        for column in feature_columns
        if column not in validation.columns
    ]

    if missing:

        raise RuntimeError(
            "Validation dataset is missing "
            f"features: {missing}"
        )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

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
    # Train XGBoost
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING EXPERIMENTAL XGBOOST")
    print("=" * 70)

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

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start
    )

    print(
        f"Training completed in "
        f"{training_time:.2f}s"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CALCULATING FEATURE IMPORTANCE")
    print("=" * 70)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    importance_df.insert(
        0,
        "rank",
        range(
            1,
            len(importance_df) + 1
        )
    )

    # --------------------------------------------------------
    # Mark original vs new
    # --------------------------------------------------------

    importance_df["feature_type"] = (
        importance_df["feature"]
        .apply(
            lambda x:
                "NEW"
                if x in new_features
                else "ORIGINAL"
        )
    )

    # --------------------------------------------------------
    # Save all importance
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    importance_df.to_csv(
        ALL_IMPORTANCE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save only new features
    # --------------------------------------------------------

    new_importance_df = (
        importance_df[
            importance_df["feature_type"] == "NEW"
        ]
        .copy()
    )

    new_importance_df.to_csv(
        NEW_IMPORTANCE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print top 30
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 30 FEATURES")
    print("=" * 70)

    print(
        importance_df[
            [
                "rank",
                "feature",
                "importance",
                "feature_type",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Print new feature rankings
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NEW FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        new_importance_df[
            [
                "rank",
                "feature",
                "importance",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMPORTANCE SUMMARY")
    print("=" * 70)

    new_total_importance = (
        new_importance_df["importance"]
        .sum()
    )

    original_total_importance = (
        importance_df[
            importance_df["feature_type"] == "ORIGINAL"
        ]["importance"]
        .sum()
    )

    print(
        f"Original features total importance: "
        f"{original_total_importance:.6f}"
    )

    print(
        f"New features total importance     : "
        f"{new_total_importance:.6f}"
    )

    print(
        f"New feature importance share       : "
        f"{new_total_importance * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"All features:"
        f"\n{ALL_IMPORTANCE_FILE}"
    )

    print(
        f"\nNew features:"
        f"\n{NEW_IMPORTANCE_FILE}"
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()