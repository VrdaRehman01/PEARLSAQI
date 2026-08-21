"""
AQI EXTREME RESIDUAL ERROR ANALYSIS

Analyzes the 2026 holdout predictions produced by the
extreme residual correction pipeline.

IMPORTANT:
- This script does NOT retrain anything.
- This script does NOT modify the model.
- It only analyzes existing test predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = Path(
    "models/extreme_residual/test_predictions.parquet"
)

OUTPUT_DIR = Path(
    "models/extreme_residual/analysis"
)

EXTREME_THRESHOLD = 200
SPIKE_THRESHOLD = 40


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(title, y_true, y_pred):
    m = metrics(y_true, y_pred)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"MAE  : {m['mae']:.4f}")
    print(f"RMSE : {m['rmse']:.4f}")
    print(f"R²   : {m['r2']:.4f}")

    return m


def classify_aqi(aqi):
    if aqi <= 100:
        return "≤100 Moderate"
    elif aqi <= 150:
        return "101-150 Unhealthy"
    elif aqi <= 200:
        return "151-200 Very Unhealthy"
    elif aqi <= 300:
        return "201-300 Severe"
    else:
        return "301+ Extreme"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXTREME RESIDUAL ERROR ANALYSIS")
print("=" * 70)

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"\nPrediction file not found:\n{PREDICTIONS_PATH}\n\n"
        "Run the extreme residual model first."
    )

df = pd.read_parquet(PREDICTIONS_PATH)

print(f"\nLoaded predictions: {len(df)} rows")
print(f"File: {PREDICTIONS_PATH}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "target_aqi",
    "base_prediction",
    "corrected_prediction",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# PREPARE ANALYSIS COLUMNS
# ============================================================

df["base_error"] = (
    df["base_prediction"] - df["target_aqi"]
)

df["corrected_error"] = (
    df["corrected_prediction"] - df["target_aqi"]
)

df["base_absolute_error"] = (
    df["base_error"].abs()
)

df["corrected_absolute_error"] = (
    df["corrected_error"].abs()
)

df["correction"] = (
    df["corrected_prediction"]
    - df["base_prediction"]
)

df["correction_magnitude"] = (
    df["correction"].abs()
)

# Positive actual residual means actual AQI was higher
# than the base prediction.
df["base_residual"] = (
    df["target_aqi"]
    - df["base_prediction"]
)

df["corrected_residual"] = (
    df["target_aqi"]
    - df["corrected_prediction"]
)

# Improvement is positive when corrected prediction
# has smaller absolute error.
df["improvement"] = (
    df["base_absolute_error"]
    - df["corrected_absolute_error"]
)

df["correction_helped"] = (
    df["improvement"] > 0
)

df["correction_hurt"] = (
    df["improvement"] < 0
)

df["correction_equal"] = (
    df["improvement"] == 0
)

# Actual AQI category
df["aqi_range"] = df["target_aqi"].apply(classify_aqi)

# Actual extreme event
df["is_extreme"] = (
    df["target_aqi"] > EXTREME_THRESHOLD
)

# Direction of base error
df["error_direction"] = np.where(
    df["base_residual"] > 0,
    "Underprediction",
    np.where(
        df["base_residual"] < 0,
        "Overprediction",
        "Exact"
    )
)

# Magnitude of actual movement if current AQI exists
if "aqi" in df.columns:

    df["actual_change"] = (
        df["target_aqi"] - df["aqi"]
    )

    df["spike_direction"] = np.where(
        df["actual_change"] >= SPIKE_THRESHOLD,
        "Upward spike",
        np.where(
            df["actual_change"] <= -SPIKE_THRESHOLD,
            "Downward spike",
            "Normal"
        )
    )

else:

    df["actual_change"] = np.nan
    df["spike_direction"] = "Unavailable"


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("1. OVERALL PERFORMANCE")
print("=" * 70)

base_metrics = print_metrics(
    "BASE XGBOOST",
    df["target_aqi"],
    df["base_prediction"]
)

corrected_metrics = print_metrics(
    "EXTREME RESIDUAL CORRECTED",
    df["target_aqi"],
    df["corrected_prediction"]
)

print("\nImprovement")
print("-" * 20)

print(
    f"MAE improvement  : "
    f"{base_metrics['mae'] - corrected_metrics['mae']:+.4f}"
)

print(
    f"RMSE improvement : "
    f"{base_metrics['rmse'] - corrected_metrics['rmse']:+.4f}"
)

print(
    f"R² improvement   : "
    f"{corrected_metrics['r2'] - base_metrics['r2']:+.4f}"
)


# ============================================================
# CORRECTION BEHAVIOR
# ============================================================

print("\n" + "=" * 70)
print("2. CORRECTION BEHAVIOR")
print("=" * 70)

print(
    f"Mean correction        : {df['correction'].mean():.4f}"
)

print(
    f"Median correction      : {df['correction'].median():.4f}"
)

print(
    f"Mean correction |abs|  : "
    f"{df['correction_magnitude'].mean():.4f}"
)

print(
    f"Max correction         : {df['correction'].max():.4f}"
)

print(
    f"Min correction         : {df['correction'].min():.4f}"
)

helped = df["correction_helped"].sum()
hurt = df["correction_hurt"].sum()
equal = df["correction_equal"].sum()

total = len(df)

print(
    f"\nCorrection helped     : {helped:,} "
    f"({helped / total * 100:.2f}%)"
)

print(
    f"Correction hurt       : {hurt:,} "
    f"({hurt / total * 100:.2f}%)"
)

print(
    f"Correction unchanged  : {equal:,} "
    f"({equal / total * 100:.2f}%)"
)


# ============================================================
# AQI RANGE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. AQI RANGE ANALYSIS")
print("=" * 70)

range_rows = []

range_order = [
    "≤100 Moderate",
    "101-150 Unhealthy",
    "151-200 Very Unhealthy",
    "201-300 Severe",
    "301+ Extreme",
]

for category in range_order:

    subset = df[df["aqi_range"] == category]

    if len(subset) == 0:
        continue

    base = metrics(
        subset["target_aqi"],
        subset["base_prediction"]
    )

    corrected = metrics(
        subset["target_aqi"],
        subset["corrected_prediction"]
    )

    range_rows.append({
        "aqi_range": category,
        "rows": len(subset),

        "base_mae": base["mae"],
        "corrected_mae": corrected["mae"],
        "mae_improvement": (
            base["mae"] - corrected["mae"]
        ),

        "base_rmse": base["rmse"],
        "corrected_rmse": corrected["rmse"],

        "actual_mean": subset["target_aqi"].mean(),
        "base_prediction_mean": subset["base_prediction"].mean(),
        "corrected_prediction_mean": subset["corrected_prediction"].mean(),

        "mean_correction": subset["correction"].mean(),

        "help_rate": (
            subset["correction_helped"].mean() * 100
        ),
        "hurt_rate": (
            subset["correction_hurt"].mean() * 100
        ),
    })

range_df = pd.DataFrame(range_rows)

print(
    range_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. CITY ANALYSIS")
print("=" * 70)

city_rows = []

if "city_name" in df.columns:

    for city, subset in df.groupby("city_name"):

        base = metrics(
            subset["target_aqi"],
            subset["base_prediction"]
        )

        corrected = metrics(
            subset["target_aqi"],
            subset["corrected_prediction"]
        )

        city_rows.append({
            "city_name": city,
            "rows": len(subset),

            "base_mae": base["mae"],
            "corrected_mae": corrected["mae"],

            "mae_improvement": (
                base["mae"] - corrected["mae"]
            ),

            "base_rmse": base["rmse"],
            "corrected_rmse": corrected["rmse"],

            "base_r2": base["r2"],
            "corrected_r2": corrected["r2"],

            "actual_mean": subset["target_aqi"].mean(),

            "base_prediction_mean": (
                subset["base_prediction"].mean()
            ),

            "corrected_prediction_mean": (
                subset["corrected_prediction"].mean()
            ),

            "mean_correction": (
                subset["correction"].mean()
            ),

            "help_rate": (
                subset["correction_helped"].mean() * 100
            ),

            "hurt_rate": (
                subset["correction_hurt"].mean() * 100
            ),
        })

city_df = pd.DataFrame(city_rows)

if not city_df.empty:

    city_df = city_df.sort_values(
        "mae_improvement",
        ascending=False
    )

    print(
        city_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# EXTREME EVENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. EXTREME EVENT ANALYSIS")
print("=" * 70)

extreme_df = df[
    df["target_aqi"] > EXTREME_THRESHOLD
]

normal_df = df[
    df["target_aqi"] <= EXTREME_THRESHOLD
]

print(
    f"Extreme rows (> {EXTREME_THRESHOLD}): "
    f"{len(extreme_df)}"
)

print(
    f"Normal rows: {len(normal_df)}"
)

if len(extreme_df) > 0:

    extreme_base = print_metrics(
        "BASE XGBOOST - EXTREME",
        extreme_df["target_aqi"],
        extreme_df["base_prediction"]
    )

    extreme_corrected = print_metrics(
        "CORRECTED - EXTREME",
        extreme_df["target_aqi"],
        extreme_df["corrected_prediction"]
    )

    print(
        f"\nExtreme MAE improvement : "
        f"{extreme_base['mae'] - extreme_corrected['mae']:+.4f}"
    )

    print(
        f"Extreme RMSE improvement: "
        f"{extreme_base['rmse'] - extreme_corrected['rmse']:+.4f}"
    )

    print(
        f"Extreme correction help rate: "
        f"{extreme_df['correction_helped'].mean() * 100:.2f}%"
    )

    print(
        f"Extreme correction hurt rate: "
        f"{extreme_df['correction_hurt'].mean() * 100:.2f}%"
    )


# ============================================================
# UNDERPREDICTION / OVERPREDICTION
# ============================================================

print("\n" + "=" * 70)
print("6. BASE MODEL ERROR DIRECTION")
print("=" * 70)

direction_rows = []

for direction, subset in df.groupby("error_direction"):

    if len(subset) == 0:
        continue

    direction_rows.append({
        "direction": direction,
        "rows": len(subset),

        "base_mae": subset["base_absolute_error"].mean(),
        "corrected_mae": subset["corrected_absolute_error"].mean(),

        "mae_improvement": (
            subset["base_absolute_error"].mean()
            - subset["corrected_absolute_error"].mean()
        ),

        "mean_base_error": subset["base_error"].mean(),
        "mean_corrected_error": subset["corrected_error"].mean(),

        "mean_correction": subset["correction"].mean(),

        "help_rate": (
            subset["correction_helped"].mean() * 100
        ),
    })

direction_df = pd.DataFrame(direction_rows)

print(
    direction_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SPIKE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. UPWARD / DOWNWARD SPIKE ANALYSIS")
print("=" * 70)

if "aqi" in df.columns:

    spike_df = df[
        df["spike_direction"] != "Normal"
    ]

    print(
        f"Spike rows: {len(spike_df)}"
    )

    if len(spike_df) > 0:

        spike_rows = []

        for direction, subset in spike_df.groupby(
            "spike_direction"
        ):

            spike_rows.append({
                "spike_direction": direction,
                "rows": len(subset),

                "mean_actual_change": (
                    subset["actual_change"].mean()
                ),

                "base_mae": (
                    subset["base_absolute_error"].mean()
                ),

                "corrected_mae": (
                    subset["corrected_absolute_error"].mean()
                ),

                "mae_improvement": (
                    subset["base_absolute_error"].mean()
                    - subset["corrected_absolute_error"].mean()
                ),

                "help_rate": (
                    subset["correction_helped"].mean() * 100
                ),

                "hurt_rate": (
                    subset["correction_hurt"].mean() * 100
                ),
            })

        spike_analysis_df = pd.DataFrame(spike_rows)

        print(
            spike_analysis_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    else:
        spike_analysis_df = pd.DataFrame()

else:

    print(
        "Current AQI column not available; "
        "spike direction cannot be calculated."
    )

    spike_analysis_df = pd.DataFrame()


# ============================================================
# TOP REMAINING ERRORS
# ============================================================

print("\n" + "=" * 70)
print("8. TOP REMAINING ERRORS")
print("=" * 70)

worst = df.sort_values(
    "corrected_absolute_error",
    ascending=False
).head(30)

display_columns = [
    col for col in [
        "city_name",
        "date",
        "aqi",
        "target_aqi",
        "base_prediction",
        "corrected_prediction",
        "base_absolute_error",
        "corrected_absolute_error",
        "correction",
        "aqi_range",
        "is_extreme",
    ]
    if col in worst.columns
]

print(
    worst[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BIGGEST IMPROVEMENTS
# ============================================================

print("\n" + "=" * 70)
print("9. BIGGEST CORRECTION WINS")
print("=" * 70)

best = df.sort_values(
    "improvement",
    ascending=False
).head(20)

print(
    best[display_columns + ["improvement"]].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BIGGEST CORRECTION FAILURES
# ============================================================

print("\n" + "=" * 70)
print("10. BIGGEST CORRECTION FAILURES")
print("=" * 70)

worst_correction = df.sort_values(
    "improvement",
    ascending=True
).head(20)

print(
    worst_correction[
        display_columns + ["improvement"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Full row-level analysis
full_path = OUTPUT_DIR / "residual_analysis.csv"
df.to_csv(full_path, index=False)

# City analysis
city_path = OUTPUT_DIR / "city_analysis.csv"
city_df.to_csv(city_path, index=False)

# AQI range analysis
range_path = OUTPUT_DIR / "range_analysis.csv"
range_df.to_csv(range_path, index=False)

# Worst predictions
worst_path = OUTPUT_DIR / "worst_predictions.csv"
worst.to_csv(worst_path, index=False)

# Best corrections
best_path = OUTPUT_DIR / "best_corrections.csv"
best.to_csv(best_path, index=False)

# Worst corrections
worst_correction_path = (
    OUTPUT_DIR / "worst_corrections.csv"
)
worst_correction.to_csv(
    worst_correction_path,
    index=False
)

# Spike analysis
if not spike_analysis_df.empty:

    spike_path = (
        OUTPUT_DIR / "spike_analysis.csv"
    )

    spike_analysis_df.to_csv(
        spike_path,
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"\nBase MAE      : {base_metrics['mae']:.4f}"
)

print(
    f"Corrected MAE : {corrected_metrics['mae']:.4f}"
)

print(
    f"MAE change    : "
    f"{base_metrics['mae'] - corrected_metrics['mae']:+.4f}"
)

print(
    f"\nBase RMSE      : {base_metrics['rmse']:.4f}"
)

print(
    f"Corrected RMSE : {corrected_metrics['rmse']:.4f}"
)

print(
    f"RMSE change    : "
    f"{base_metrics['rmse'] - corrected_metrics['rmse']:+.4f}"
)

print(
    f"\nBase R²      : {base_metrics['r2']:.4f}"
)

print(
    f"Corrected R² : {corrected_metrics['r2']:.4f}"
)

print(
    f"R² change    : "
    f"{corrected_metrics['r2'] - base_metrics['r2']:+.4f}"
)

print(
    f"\nCorrection helped: "
    f"{helped / total * 100:.2f}%"
)

print(
    f"Correction hurt  : "
    f"{hurt / total * 100:.2f}%"
)

print("\nSaved analysis files:")

print(f"  {full_path}")
print(f"  {city_path}")
print(f"  {range_path}")
print(f"  {worst_path}")
print(f"  {best_path}")
print(f"  {worst_correction_path}")

if not spike_analysis_df.empty:
    print(f"  {spike_path}")

print("\nAnalysis completed successfully.")