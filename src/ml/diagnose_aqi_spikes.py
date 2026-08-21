from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "test_predictions.parquet"
)


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("PEARLSAQI - AQI SPIKE DIAGNOSTICS")
print("=" * 80)
print()

df = pd.read_parquet(PREDICTIONS_FILE)

print(f"Rows loaded: {len(df)}")
print()


# ============================================================
# BASIC CLEANUP
# ============================================================

df["aqi"] = pd.to_numeric(
    df["aqi"],
    errors="coerce"
)

df["target_aqi"] = pd.to_numeric(
    df["target_aqi"],
    errors="coerce"
)

df["prediction"] = pd.to_numeric(
    df["prediction"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "aqi",
        "target_aqi",
        "prediction",
        "city_name",
    ]
).copy()


# ============================================================
# CHANGES
# ============================================================

df["actual_change"] = (
    df["target_aqi"] - df["aqi"]
)

df["predicted_change"] = (
    df["prediction"] - df["aqi"]
)

df["change_error"] = (
    df["predicted_change"]
    - df["actual_change"]
)

df["absolute_error"] = (
    df["prediction"]
    - df["target_aqi"]
).abs()


# ============================================================
# OVERALL
# ============================================================

print("=" * 80)
print("OVERALL AQI MOVEMENT")
print("=" * 80)
print()

print(
    f"Average current AQI: "
    f"{df['aqi'].mean():.2f}"
)

print(
    f"Average actual next-day AQI: "
    f"{df['target_aqi'].mean():.2f}"
)

print(
    f"Average predicted next-day AQI: "
    f"{df['prediction'].mean():.2f}"
)

print(
    f"Average actual change: "
    f"{df['actual_change'].mean():.2f}"
)

print(
    f"Average predicted change: "
    f"{df['predicted_change'].mean():.2f}"
)

print()


# ============================================================
# CURRENT AQI BINS
# ============================================================

print("=" * 80)
print("ERROR BY CURRENT AQI")
print("=" * 80)
print()

bins = [
    -np.inf,
    50,
    100,
    150,
    200,
    250,
    300,
    np.inf,
]

labels = [
    "0-50",
    "51-100",
    "101-150",
    "151-200",
    "201-250",
    "251-300",
    "301+",
]

df["current_aqi_range"] = pd.cut(
    df["aqi"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)


current_summary = (
    df
    .groupby(
        "current_aqi_range",
        observed=False
    )
    .agg(
        rows=("aqi", "size"),
        current_mean=("aqi", "mean"),
        actual_mean=("target_aqi", "mean"),
        prediction_mean=("prediction", "mean"),
        actual_change=("actual_change", "mean"),
        predicted_change=("predicted_change", "mean"),
        mae=("absolute_error", "mean"),
    )
)

print(
    current_summary.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ============================================================
# ACTUAL NEXT-DAY AQI BINS
# ============================================================

print("=" * 80)
print("ERROR BY ACTUAL NEXT-DAY AQI")
print("=" * 80)
print()

target_bins = [
    -np.inf,
    50,
    100,
    150,
    200,
    250,
    300,
    np.inf,
]

target_labels = [
    "0-50",
    "51-100",
    "101-150",
    "151-200",
    "201-250",
    "251-300",
    "301+",
]

df["target_aqi_range"] = pd.cut(
    df["target_aqi"],
    bins=target_bins,
    labels=target_labels,
    include_lowest=True,
)


target_summary = (
    df
    .groupby(
        "target_aqi_range",
        observed=False
    )
    .agg(
        rows=("target_aqi", "size"),
        actual_mean=("target_aqi", "mean"),
        prediction_mean=("prediction", "mean"),
        mae=("absolute_error", "mean"),
    )
)

print(
    target_summary.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ============================================================
# BIGGEST POSITIVE SPIKES
# ============================================================

print("=" * 80)
print("BIGGEST ACTUAL AQI INCREASES")
print("=" * 80)
print()

spikes_up = (
    df
    .sort_values(
        "actual_change",
        ascending=False
    )
    .head(25)
)

print(
    spikes_up[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "prediction",
            "actual_change",
            "predicted_change",
            "change_error",
            "absolute_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ============================================================
# BIGGEST NEGATIVE CHANGES
# ============================================================

print("=" * 80)
print("BIGGEST ACTUAL AQI DECREASES")
print("=" * 80)
print()

spikes_down = (
    df
    .sort_values(
        "actual_change",
        ascending=True
    )
    .head(15)
)

print(
    spikes_down[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "prediction",
            "actual_change",
            "predicted_change",
            "change_error",
            "absolute_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ============================================================
# CITY SPIKE PERFORMANCE
# ============================================================

print("=" * 80)
print("CITY PERFORMANCE DURING AQI INCREASES")
print("=" * 80)
print()

city_spikes = (
    df[df["actual_change"] >= 30]
    .groupby("city_name")
    .agg(
        spike_rows=("actual_change", "size"),
        actual_change=("actual_change", "mean"),
        predicted_change=("predicted_change", "mean"),
        change_error=("change_error", "mean"),
        mae=("absolute_error", "mean"),
    )
    .sort_values(
        "mae",
        ascending=False
    )
)

print(
    city_spikes.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ============================================================
# EXTREME EVENTS
# ============================================================

print("=" * 80)
print("EXTREME EVENTS: ACTUAL AQI >= 200")
print("=" * 80)
print()

extreme = df[
    df["target_aqi"] >= 200
].copy()

print(
    f"Extreme rows: {len(extreme)}"
)

if len(extreme) > 0:

    extreme["underprediction"] = (
        extreme["target_aqi"]
        - extreme["prediction"]
    )

    print(
        f"Average actual AQI: "
        f"{extreme['target_aqi'].mean():.2f}"
    )

    print(
        f"Average prediction: "
        f"{extreme['prediction'].mean():.2f}"
    )

    print(
        f"Average underprediction: "
        f"{extreme['underprediction'].mean():.2f}"
    )

    print(
        f"Median underprediction: "
        f"{extreme['underprediction'].median():.2f}"
    )

    print()


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "aqi_spike_diagnostics.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print()

print(
    f"Saved detailed diagnostics to:\n"
    f"{OUTPUT_FILE}"
)

print()