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


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/processed/v4/train.parquet"
VALIDATION_FILE = "data/processed/v4/validation.parquet"

RESULTS_DIR = "models/benchmark/v4"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "feature_ablation_results.csv",
)

TARGET = "target_aqi"

EXCLUDED = [
    "target_aqi",
    "city_name",
    "date",
]


# ============================================================
# ORIGINAL FEATURES
# ============================================================

ORIGINAL_FEATURES = [
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
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_7",
    "aqi_lag_14",
    "aqi_change_1d",
    "aqi_change_2d",
    "aqi_change_3d",
    "aqi_change_7d",
    "aqi_acceleration_1d",
    "aqi_acceleration_2d",
    "aqi_rolling_3",
    "aqi_std_3",
    "aqi_min_3",
    "aqi_max_3",
    "aqi_range_3",
    "aqi_rolling_7",
    "aqi_std_7",
    "aqi_min_7",
    "aqi_max_7",
    "aqi_range_7",
    "aqi_rolling_14",
    "aqi_std_14",
    "aqi_min_14",
    "aqi_max_14",
    "aqi_range_14",
    "aqi_trend_3d",
    "aqi_trend_7d",
    "aqi_distance_from_max_7",
    "aqi_distance_from_max_14",
    "aqi_percentile_7",
    "aqi_percentile_14",
    "pm25_change_1d",
    "pm25_change_3d",
    "pm25_rolling_3",
    "pm25_rolling_7",
    "pm10_change_1d",
    "pm10_change_3d",
    "pm10_rolling_3",
    "pm10_rolling_7",
    "no2_change_1d",
    "no2_change_3d",
    "no2_rolling_3",
    "no2_rolling_7",
    "so2_change_1d",
    "so2_change_3d",
    "so2_rolling_3",
    "so2_rolling_7",
    "co_change_1d",
    "co_change_3d",
    "co_rolling_3",
    "co_rolling_7",
    "o3_change_1d",
    "o3_change_3d",
    "o3_rolling_3",
    "o3_rolling_7",
    "pm25_trend_3d",
    "pm25_trend_7d",
    "pm10_trend_3d",
    "pm10_trend_7d",
    "pm25_pm10_ratio",
    "pm25_no2_interaction",
    "pm25_co_interaction",
    "pm25_o3_interaction",
    "pollution_sum",
    "temperature_change_1d",
    "humidity_change_1d",
    "windspeed_change_1d",
    "precipitation_change_1d",
    "precipitation_rolling_3",
    "windspeed_rolling_3",
    "aqi_moderate",
    "aqi_unhealthy",
    "aqi_very_unhealthy",
    "aqi_severe",
    "aqi_extreme",
    "aqi_regime_change",
    "high_aqi_recent",
    "extreme_aqi_recent",
    "city_aqi_mean",
    "city_aqi_std",
    "city_pm25_mean",
    "city_recent_mean",
    "city_recent_max",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# ============================================================
# EXTRA FEATURES
# ============================================================

NEW_FEATURES = [
    "temperature_rolling_3",
    "temperature_rolling_7",
    "temperature_std_3",
    "humidity_rolling_3",
    "humidity_rolling_7",
    "humidity_std_3",
    "windspeed_rolling_7",
    "windspeed_std_3",
    "pm25_std_3",
    "pm25_std_7",
    "pm10_std_3",
    "pm10_std_7",
    "no2_std_3",
    "no2_std_7",
    "so2_std_3",
    "so2_std_7",
    "co_std_3",
    "co_std_7",
    "o3_std_3",
    "o3_std_7",
    "so2_no2_ratio",
    "o3_no2_ratio",
    "pm25_pollution_ratio",
    "pm10_pollution_ratio",
]


# ============================================================
# FEATURE GENERATION
# ============================================================

def add_extra_features(df):

    df = df.copy()

    grouped = df.groupby(
        "city_id",
        group_keys=False,
    )

    # Weather rolling features
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
            .reset_index(
                level=0,
                drop=True,
            )
        )

        df[f"{column}_rolling_7"] = (
            grouped[column]
            .shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        df[f"{column}_std_3"] = (
            grouped[column]
            .shift(1)
            .rolling(3, min_periods=2)
            .std()
            .reset_index(
                level=0,
                drop=True,
            )
            .fillna(0)
        )

    # Pollutant volatility
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
                    min_periods=2,
                )
                .std()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .fillna(0)
            )

    # Ratios
    epsilon = 1e-6

    df["so2_no2_ratio"] = (
        df["so2"]
        / (df["no2"].abs() + epsilon)
    )

    df["o3_no2_ratio"] = (
        df["o3"]
        / (df["no2"].abs() + epsilon)
    )

    df["pm25_pollution_ratio"] = (
        df["pm25"]
        / (df["pollution_sum"].abs() + epsilon)
    )

    df["pm10_pollution_ratio"] = (
        df["pm10"]
        / (df["pollution_sum"].abs() + epsilon)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.fillna(0)

    return df


# ============================================================
# MODEL
# ============================================================

def create_model():

    return XGBRegressor(
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, feature_columns, train, validation):

    print()
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

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

    model = create_model()

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time() - start
    )

    predictions = model.predict(
        X_validation
    )

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    rmse = mean_squared_error(
        y_validation,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_validation,
        predictions,
    )

    print(
        f"Features       : {len(feature_columns)}"
    )

    print(
        f"MAE            : {mae:.4f}"
    )

    print(
        f"RMSE           : {rmse:.4f}"
    )

    print(
        f"R2             : {r2:.4f}"
    )

    print(
        f"Training time  : {training_time:.2f}s"
    )

    return {
        "experiment": name,
        "feature_count": len(feature_columns),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "training_time_seconds": training_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLSAQI V4 FEATURE ABLATION")
    print("=" * 70)

    train = pd.read_parquet(
        TRAIN_FILE
    )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    print()
    print(
        f"Training rows   : {len(train)}"
    )

    print(
        f"Validation rows : {len(validation)}"
    )

    # Add experimental features in memory only
    train = add_extra_features(
        train
    )

    validation = add_extra_features(
        validation
    )

    # Verify all features exist
    for feature in (
        ORIGINAL_FEATURES + NEW_FEATURES
    ):

        if feature not in train.columns:
            raise RuntimeError(
                f"Training missing feature: {feature}"
            )

        if feature not in validation.columns:
            raise RuntimeError(
                f"Validation missing feature: {feature}"
            )

    # --------------------------------------------------------
    # Three controlled experiments
    # --------------------------------------------------------

    results = []

    # A: Current production feature set
    results.append(
        evaluate(
            "A - ORIGINAL 107",
            ORIGINAL_FEATURES,
            train,
            validation,
        )
    )

    # B: Original + dominant feature
    results.append(
        evaluate(
            "B - ORIGINAL + PM25 RATIO",
            ORIGINAL_FEATURES
            + ["pm25_pollution_ratio"],
            train,
            validation,
        )
    )

    # C: Original + all 24 experimental features
    results.append(
        evaluate(
            "C - ALL 24 EXTRA FEATURES",
            ORIGINAL_FEATURES
            + NEW_FEATURES,
            train,
            validation,
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "rmse"
    ).reset_index(
        drop=True
    )

    print()
    print("=" * 70)
    print("FEATURE ABLATION RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compare against production v007
    # --------------------------------------------------------

    production_rmse = 14.379474258005216
    production_mae = 9.02976086063472
    production_r2 = 0.9013127340131636

    print()
    print("=" * 70)
    print("PRODUCTION V007 REFERENCE")
    print("=" * 70)

    print(
        f"v007 RMSE : {production_rmse:.4f}"
    )

    print(
        f"v007 MAE  : {production_mae:.4f}"
    )

    print(
        f"v007 R2   : {production_r2:.4f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    print()
    print("=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()