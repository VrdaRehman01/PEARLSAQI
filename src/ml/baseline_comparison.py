import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"


print("=" * 70)
print("AQI BASELINE COMPARISON")
print("=" * 70)


train_df = pd.read_parquet(TRAIN_FILE)
validation_df = pd.read_parquet(VALIDATION_FILE)


# ==========================================================
# BASELINE 1
# Tomorrow AQI = today's AQI
# ==========================================================

actual = validation_df["target_aqi"].values

persistence_prediction = validation_df["aqi"].values


mae = mean_absolute_error(
    actual,
    persistence_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        persistence_prediction
    )
)

r2 = r2_score(
    actual,
    persistence_prediction
)


print()
print("=" * 70)
print("PERSISTENCE BASELINE")
print("=" * 70)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ==========================================================
# BASELINE 2
# Tomorrow AQI = 7-day rolling average
# ==========================================================

rolling_prediction = validation_df[
    "aqi_rolling_7"
].values


mae_rolling = mean_absolute_error(
    actual,
    rolling_prediction
)

rmse_rolling = np.sqrt(
    mean_squared_error(
        actual,
        rolling_prediction
    )
)

r2_rolling = r2_score(
    actual,
    rolling_prediction
)


print()
print("=" * 70)
print("7-DAY ROLLING BASELINE")
print("=" * 70)

print(
    f"MAE  : {mae_rolling:.4f}"
)

print(
    f"RMSE : {rmse_rolling:.4f}"
)

print(
    f"R²   : {r2_rolling:.4f}"
)


# ==========================================================
# EXTREME EVENT ANALYSIS
# ==========================================================

validation = validation_df.copy()

validation["persistence_error"] = np.abs(
    validation["target_aqi"]
    -
    validation["aqi"]
)


validation["aqi_range"] = pd.cut(
    validation["target_aqi"],
    bins=[
        -np.inf,
        100,
        150,
        200,
        300,
        np.inf
    ],
    labels=[
        "≤100",
        "101-150",
        "151-200",
        "201-300",
        "301+"
    ]
)


range_results = (
    validation
    .groupby(
        "aqi_range",
        observed=True
    )
    .agg(
        rows=("target_aqi", "size"),
        mae=("persistence_error", "mean"),
        actual_mean=("target_aqi", "mean")
    )
    .reset_index()
)


print()
print("=" * 70)
print("PERSISTENCE BASELINE BY AQI RANGE")
print("=" * 70)

print(
    range_results.to_string(
        index=False
    )
)