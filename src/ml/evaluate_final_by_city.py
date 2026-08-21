from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "test_predictions.parquet"
)


print()
print("=" * 80)
print("PEARLSAQI - FINAL MODEL CITY-BY-CITY EVALUATION")
print("=" * 80)
print()

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}"
    )

df = pd.read_parquet(PREDICTIONS_FILE)

print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print()


# ------------------------------------------------------------
# DETECT TARGET / PREDICTION COLUMNS
# ------------------------------------------------------------

required = [
    "city_name",
    "prediction",
]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# Your parquet may call the actual next-day AQI either
# target_aqi or another name.

if "target_aqi" in df.columns:
    actual_column = "target_aqi"

elif "next_day_aqi" in df.columns:
    actual_column = "next_day_aqi"

else:
    raise ValueError(
        "Could not find actual AQI column. "
        "Expected target_aqi or next_day_aqi."
    )


df["actual"] = pd.to_numeric(
    df[actual_column],
    errors="coerce"
)

df["prediction"] = pd.to_numeric(
    df["prediction"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "city_name",
        "actual",
        "prediction",
    ]
).copy()


# ------------------------------------------------------------
# ERRORS
# ------------------------------------------------------------

df["error"] = (
    df["prediction"] - df["actual"]
)

df["absolute_error"] = (
    df["error"].abs()
)

df["squared_error"] = (
    df["error"] ** 2
)


# ------------------------------------------------------------
# CITY METRICS
# ------------------------------------------------------------

rows = []

for city, group in df.groupby("city_name"):

    actual = group["actual"].values
    prediction = group["prediction"].values

    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    r2 = r2_score(
        actual,
        prediction
    )

    bias = (
        prediction - actual
    ).mean()

    within_10 = (
        np.abs(
            prediction - actual
        ) <= 10
    ).mean() * 100

    within_20 = (
        np.abs(
            prediction - actual
        ) <= 20
    ).mean() * 100

    within_30 = (
        np.abs(
            prediction - actual
        ) <= 30
    ).mean() * 100

    rows.append({
        "city": city,
        "rows": len(group),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "actual_mean": actual.mean(),
        "prediction_mean": prediction.mean(),
        "actual_max": actual.max(),
        "prediction_max": prediction.max(),
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    })


metrics = pd.DataFrame(rows)

metrics = metrics.sort_values(
    "mae",
    ascending=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print()
print("=" * 80)
print("CITY PERFORMANCE")
print("=" * 80)
print()

print(
    metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# BIAS ANALYSIS
# ------------------------------------------------------------

print()
print("=" * 80)
print("CITY BIAS")
print("=" * 80)
print()

bias_table = metrics[
    [
        "city",
        "actual_mean",
        "prediction_mean",
        "bias",
    ]
].copy()

print(
    bias_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# HIGH AQI PERFORMANCE
# ------------------------------------------------------------

print()
print("=" * 80)
print("HIGH AQI PERFORMANCE (ACTUAL >= 200)")
print("=" * 80)
print()

high = df[
    df["actual"] >= 200
].copy()

if len(high) == 0:

    print("No actual AQI >= 200 found.")

else:

    high_mae = mean_absolute_error(
        high["actual"],
        high["prediction"]
    )

    high_rmse = np.sqrt(
        mean_squared_error(
            high["actual"],
            high["prediction"]
        )
    )

    print(
        f"High-AQI rows : {len(high)}"
    )

    print(
        f"MAE           : {high_mae:.2f}"
    )

    print(
        f"RMSE          : {high_rmse:.2f}"
    )

    print()

    high_city = (
        high
        .groupby("city_name")
        .agg(
            rows=("actual", "size"),
            actual_mean=("actual", "mean"),
            prediction_mean=("prediction", "mean"),
            actual_max=("actual", "max"),
            prediction_max=("prediction", "max"),
            mae=("absolute_error", "mean"),
        )
        .sort_values(
            "mae",
            ascending=False
        )
    )

    print(
        high_city.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


# ------------------------------------------------------------
# WORST 20
# ------------------------------------------------------------

print()
print("=" * 80)
print("20 WORST PREDICTIONS")
print("=" * 80)
print()

display_columns = [
    column
    for column in [
        "city_name",
        "date",
        "aqi",
        "target_aqi",
        "actual",
        "prediction",
        "error",
        "absolute_error",
    ]
    if column in df.columns
]

worst = (
    df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)

print(
    worst[display_columns].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "models"
    / "final_production_xgboost"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    OUTPUT_DIR
    / "city_metrics.csv"
)

metrics.to_csv(
    output_file,
    index=False
)

print()
print("Saved:")
print(output_file)

print()
print("=" * 80)
print("CITY EVALUATION COMPLETE")
print("=" * 80)