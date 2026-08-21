"""
PEARLSAQI V8 - CITY REGIME & SPIKE DIAGNOSTICS

Purpose:
    Deeply analyze why AQI predictions fail differently by city,
    AQI regime, and sudden AQI movements.

This is a DIAGNOSTIC stage.
It does NOT replace the production model.

Baseline:
    MAE  = 13.1605
    RMSE = 18.1385
    R2   = 0.8546
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    ROOT
    / "data"
    / "analysis"
    / "test_error_analysis.parquet"
)

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "v8"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

SPIKE_THRESHOLD = 40
EXTREME_THRESHOLD = 200

BASELINE_MAE = 13.1605
BASELINE_RMSE = 18.1385
BASELINE_R2 = 0.8546


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def aqi_regime(value):

    if value <= 50:
        return "Good"

    if value <= 100:
        return "Moderate"

    if value <= 150:
        return "USG"

    if value <= 200:
        return "Unhealthy"

    if value <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def safe_rmse(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan

    return float(np.sqrt(np.mean(values ** 2)))


def safe_r2(actual, prediction):

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if len(actual) < 2:
        return np.nan

    denominator = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if denominator == 0:
        return np.nan

    numerator = np.sum(
        (actual - prediction) ** 2
    )

    return float(
        1 - numerator / denominator
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print_section("PEARLSAQI V8 - CITY REGIME & SPIKE DIAGNOSTICS")

print("\nLoading test predictions...")

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_PATH}"
    )

test = pd.read_parquet(TEST_PATH)

print(f"Test rows : {len(test)}")

required_columns = {
    "city_name",
    "date",
    "aqi",
    "target_aqi",
    "prediction",
}

missing = required_columns - set(test.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

test = test.copy()

test["date"] = pd.to_datetime(
    test["date"]
)

test["aqi"] = pd.to_numeric(
    test["aqi"],
    errors="coerce",
)

test["target_aqi"] = pd.to_numeric(
    test["target_aqi"],
    errors="coerce",
)

test["prediction"] = pd.to_numeric(
    test["prediction"],
    errors="coerce",
)

test = test.dropna(
    subset=[
        "aqi",
        "target_aqi",
        "prediction",
    ]
)

test = test.sort_values(
    ["city_name", "date"]
).reset_index(drop=True)


# ============================================================
# CREATE DIAGNOSTIC VARIABLES
# ============================================================

test["error"] = (
    test["prediction"]
    - test["target_aqi"]
)

test["absolute_error"] = (
    test["error"].abs()
)

test["actual_change"] = (
    test["target_aqi"]
    - test["aqi"]
)

test["predicted_change"] = (
    test["prediction"]
    - test["aqi"]
)

test["change_error"] = (
    test["predicted_change"]
    - test["actual_change"]
)

test["actual_regime"] = (
    test["target_aqi"]
    .apply(aqi_regime)
)

test["current_regime"] = (
    test["aqi"]
    .apply(aqi_regime)
)

test["prediction_regime"] = (
    test["prediction"]
    .apply(aqi_regime)
)

test["is_spike"] = (
    test["actual_change"].abs()
    >= SPIKE_THRESHOLD
)

test["is_upward_spike"] = (
    test["actual_change"]
    >= SPIKE_THRESHOLD
)

test["is_downward_spike"] = (
    test["actual_change"]
    <= -SPIKE_THRESHOLD
)

test["is_extreme"] = (
    test["target_aqi"]
    >= EXTREME_THRESHOLD
)

test["is_underprediction"] = (
    test["prediction"]
    < test["target_aqi"]
)


# ============================================================
# DATA OVERVIEW
# ============================================================

print_section("DATA OVERVIEW")

print(
    f"Date range : "
    f"{test['date'].min().date()} "
    f"→ "
    f"{test['date'].max().date()}"
)

print(
    f"Cities     : "
    f"{test['city_name'].nunique()}"
)

print(
    f"Rows       : "
    f"{len(test)}"
)

print(
    f"Baseline MAE  : {BASELINE_MAE:.4f}"
)

print(
    f"Baseline RMSE : {BASELINE_RMSE:.4f}"
)

print(
    f"Baseline R²   : {BASELINE_R2:.4f}"
)


# ============================================================
# CITY PERFORMANCE
# ============================================================

print_section("CITY PERFORMANCE")

city_rows = []

for city, group in test.groupby(
    "city_name"
):

    actual = group["target_aqi"].values
    prediction = group["prediction"].values

    errors = (
        prediction - actual
    )

    city_rows.append({

        "city": city,

        "rows": len(group),

        "mae": np.mean(
            np.abs(errors)
        ),

        "rmse": safe_rmse(errors),

        "r2": safe_r2(
            actual,
            prediction,
        ),

        "bias": np.mean(errors),

        "actual_mean": np.mean(actual),

        "prediction_mean": np.mean(
            prediction
        ),

        "actual_max": np.max(actual),

        "prediction_max": np.max(
            prediction
        ),

        "extreme_rows": int(
            group["is_extreme"].sum()
        ),

        "spike_rows": int(
            group["is_spike"].sum()
        ),

        "upward_spikes": int(
            group["is_upward_spike"].sum()
        ),

        "downward_spikes": int(
            group["is_downward_spike"].sum()
        ),

        "underprediction_rate": (
            group["is_underprediction"]
            .mean()
            * 100
        ),

    })


city_metrics = pd.DataFrame(
    city_rows
).sort_values(
    "mae",
    ascending=False,
)

print(
    city_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

city_metrics.to_csv(
    OUTPUT_DIR / "city_performance.csv",
    index=False,
)


# ============================================================
# AQI REGIME PERFORMANCE
# ============================================================

print_section("AQI REGIME PERFORMANCE")

regime_rows = []

for regime, group in test.groupby(
    "actual_regime"
):

    errors = (
        group["prediction"]
        - group["target_aqi"]
    )

    regime_rows.append({

        "regime": regime,

        "rows": len(group),

        "mae": np.mean(
            np.abs(errors)
        ),

        "rmse": safe_rmse(errors),

        "bias": np.mean(errors),

        "actual_mean": group[
            "target_aqi"
        ].mean(),

        "prediction_mean": group[
            "prediction"
        ].mean(),

        "actual_max": group[
            "target_aqi"
        ].max(),

        "prediction_max": group[
            "prediction"
        ].max(),

        "underprediction_rate": (
            group["is_underprediction"]
            .mean()
            * 100
        ),

    })


regime_metrics = pd.DataFrame(
    regime_rows
)

print(
    regime_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

regime_metrics.to_csv(
    OUTPUT_DIR / "regime_performance.csv",
    index=False,
)


# ============================================================
# CURRENT AQI RANGE PERFORMANCE
# ============================================================

print_section("CURRENT AQI RANGE PERFORMANCE")

def current_range(value):

    if value <= 50:
        return "0-50"

    if value <= 100:
        return "51-100"

    if value <= 150:
        return "101-150"

    if value <= 200:
        return "151-200"

    if value <= 250:
        return "201-250"

    if value <= 300:
        return "251-300"

    return "301+"


test["current_range"] = (
    test["aqi"]
    .apply(current_range)
)

range_rows = []

for current_range_name, group in test.groupby(
    "current_range"
):

    errors = (
        group["prediction"]
        - group["target_aqi"]
    )

    range_rows.append({

        "current_range": current_range_name,

        "rows": len(group),

        "current_mean": group[
            "aqi"
        ].mean(),

        "actual_mean": group[
            "target_aqi"
        ].mean(),

        "prediction_mean": group[
            "prediction"
        ].mean(),

        "actual_change": group[
            "actual_change"
        ].mean(),

        "predicted_change": group[
            "predicted_change"
        ].mean(),

        "mae": np.mean(
            np.abs(errors)
        ),

    })


range_metrics = pd.DataFrame(
    range_rows
)

print(
    range_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

range_metrics.to_csv(
    OUTPUT_DIR / "current_aqi_ranges.csv",
    index=False,
)


# ============================================================
# SPIKE PERFORMANCE BY CITY
# ============================================================

print_section("SPIKE PERFORMANCE BY CITY")

spike_rows = []

for city, group in test.groupby(
    "city_name"
):

    spikes = group[
        group["is_spike"]
    ]

    if len(spikes) == 0:
        continue

    spike_errors = (
        spikes["prediction"]
        - spikes["target_aqi"]
    )

    spike_rows.append({

        "city": city,

        "spike_rows": len(spikes),

        "actual_change": spikes[
            "actual_change"
        ].mean(),

        "predicted_change": spikes[
            "predicted_change"
        ].mean(),

        "change_error": spikes[
            "change_error"
        ].mean(),

        "mae": np.mean(
            np.abs(spike_errors)
        ),

        "upward_spikes": int(
            spikes["is_upward_spike"].sum()
        ),

        "downward_spikes": int(
            spikes["is_downward_spike"].sum()
        ),

    })


spike_metrics = pd.DataFrame(
    spike_rows
).sort_values(
    "mae",
    ascending=False,
)

print(
    spike_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

spike_metrics.to_csv(
    OUTPUT_DIR / "city_spike_performance.csv",
    index=False,
)


# ============================================================
# UPWARD SPIKES
# ============================================================

print_section("UPWARD SPIKES")

upward = test[
    test["is_upward_spike"]
].copy()

print(
    f"Upward spike rows : {len(upward)}"
)

if len(upward) > 0:

    print(
        f"Average actual change : "
        f"{upward['actual_change'].mean():.2f}"
    )

    print(
        f"Average predicted change : "
        f"{upward['predicted_change'].mean():.2f}"
    )

    print(
        f"Average change error : "
        f"{upward['change_error'].mean():.2f}"
    )

    print(
        f"Spike MAE : "
        f"{upward['absolute_error'].mean():.2f}"
    )


# ============================================================
# DOWNWARD SPIKES
# ============================================================

print_section("DOWNWARD SPIKES")

downward = test[
    test["is_downward_spike"]
].copy()

print(
    f"Downward spike rows : {len(downward)}"
)

if len(downward) > 0:

    print(
        f"Average actual change : "
        f"{downward['actual_change'].mean():.2f}"
    )

    print(
        f"Average predicted change : "
        f"{downward['predicted_change'].mean():.2f}"
    )

    print(
        f"Average change error : "
        f"{downward['change_error'].mean():.2f}"
    )

    print(
        f"Spike MAE : "
        f"{downward['absolute_error'].mean():.2f}"
    )


# ============================================================
# EXTREME AQI PERFORMANCE
# ============================================================

print_section("EXTREME AQI PERFORMANCE")

extreme = test[
    test["is_extreme"]
].copy()

print(
    f"Extreme rows : {len(extreme)}"
)

if len(extreme) > 0:

    extreme_errors = (
        extreme["prediction"]
        - extreme["target_aqi"]
    )

    print(
        f"Actual mean     : "
        f"{extreme['target_aqi'].mean():.2f}"
    )

    print(
        f"Prediction mean : "
        f"{extreme['prediction'].mean():.2f}"
    )

    print(
        f"Underprediction : "
        f"{(
            extreme['target_aqi']
            - extreme['prediction']
        ).mean():.2f}"
    )

    print(
        f"Extreme MAE     : "
        f"{np.abs(extreme_errors).mean():.2f}"
    )

    print(
        f"Extreme RMSE    : "
        f"{safe_rmse(extreme_errors):.2f}"
    )


# ============================================================
# WORST PREDICTIONS
# ============================================================

print_section("TOP 30 WORST PREDICTIONS")

worst = test.sort_values(
    "absolute_error",
    ascending=False,
).head(30)

print(
    worst[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "prediction",
            "actual_change",
            "predicted_change",
            "absolute_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

worst.to_csv(
    OUTPUT_DIR / "worst_predictions.csv",
    index=False,
)


# ============================================================
# BIGGEST UNDERPREDICTIONS
# ============================================================

print_section("TOP 30 UNDERPREDICTIONS")

under = test.sort_values(
    "error",
    ascending=True,
).head(30)

print(
    under[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "prediction",
            "error",
            "actual_change",
            "predicted_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

under.to_csv(
    OUTPUT_DIR / "underpredictions.csv",
    index=False,
)


# ============================================================
# BIGGEST OVERPREDICTIONS
# ============================================================

print_section("TOP 30 OVERPREDICTIONS")

over = test.sort_values(
    "error",
    ascending=False,
).head(30)

print(
    over[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "prediction",
            "error",
            "actual_change",
            "predicted_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

over.to_csv(
    OUTPUT_DIR / "overpredictions.csv",
    index=False,
)


# ============================================================
# CITY × REGIME
# ============================================================

print_section("CITY × AQI REGIME ANALYSIS")

city_regime_rows = []

for (
    city,
    regime,
), group in test.groupby(
    [
        "city_name",
        "actual_regime",
    ]
):

    errors = (
        group["prediction"]
        - group["target_aqi"]
    )

    city_regime_rows.append({

        "city": city,

        "regime": regime,

        "rows": len(group),

        "mae": np.mean(
            np.abs(errors)
        ),

        "bias": np.mean(errors),

        "actual_mean": group[
            "target_aqi"
        ].mean(),

        "prediction_mean": group[
            "prediction"
        ].mean(),

        "underprediction_rate": (
            group["is_underprediction"]
            .mean()
            * 100
        ),

    })


city_regime = pd.DataFrame(
    city_regime_rows
).sort_values(
    "mae",
    ascending=False,
)

print(
    city_regime.head(50).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

city_regime.to_csv(
    OUTPUT_DIR / "city_regime_analysis.csv",
    index=False,
)


# ============================================================
# SAVE FULL DIAGNOSTIC DATA
# ============================================================

diagnostic_columns = [
    "city_name",
    "date",
    "aqi",
    "target_aqi",
    "prediction",
    "error",
    "absolute_error",
    "actual_change",
    "predicted_change",
    "change_error",
    "current_regime",
    "actual_regime",
    "prediction_regime",
    "is_spike",
    "is_upward_spike",
    "is_downward_spike",
    "is_extreme",
    "is_underprediction",
]

test[
    diagnostic_columns
].to_parquet(
    OUTPUT_DIR / "full_v8_diagnostics.parquet",
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section("V8 DIAGNOSTIC SUMMARY")

print(
    f"Cities analyzed        : "
    f"{test['city_name'].nunique()}"
)

print(
    f"Total test rows        : "
    f"{len(test)}"
)

print(
    f"Spike rows             : "
    f"{test['is_spike'].sum()}"
)

print(
    f"Upward spikes          : "
    f"{test['is_upward_spike'].sum()}"
)

print(
    f"Downward spikes        : "
    f"{test['is_downward_spike'].sum()}"
)

print(
    f"Extreme rows           : "
    f"{test['is_extreme'].sum()}"
)

print(
    "\nDiagnostic files saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nPEARLSAQI V8 diagnostics completed successfully."
)