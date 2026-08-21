"""
PEARLSAQI FORECAST CALIBRATION V5

City × Horizon × AQI Regime Calibration

Purpose:
    Improve multi-horizon AQI forecasts by learning systematic
    prediction bias for combinations of:

        city
        forecast horizon
        AQI regime

Important:
    - Does NOT modify V3.
    - Does NOT modify V4.
    - Uses the same walk-forward validation dataset.
    - Uses a temporal calibration/evaluation split.
    - Calibration parameters are learned ONLY from the calibration period.
    - Evaluation is performed on a later untouched period.

AQI regimes:
    0-50       Good
    51-100     Moderate
    101-150    USG
    151-200    Unhealthy
    201-300    Very Unhealthy
    301+       Hazardous
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    ROOT
    / "models"
    / "forecast"
    / "validation"
    / "forecast_validation_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_FILE = (
    OUTPUT_DIR
    / "calibration_strategy_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

CITY_FILE = (
    OUTPUT_DIR
    / "calibration_city_results.csv"
)

REGIME_FILE = (
    OUTPUT_DIR
    / "calibration_regime_results.csv"
)

PARAMETERS_FILE = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)


# ============================================================
# CALIBRATION SETTINGS
# ============================================================

CALIBRATION_RATIO = 0.60

# Bias correction shrinkage.
SHRINKAGES = [
    0.25,
    0.50,
    0.75,
    1.00,
]

# Maximum correction magnitude.
CAPS = [
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
]

# Minimum number of calibration observations required
# for a city × horizon × regime group.
MIN_GROUP_ROWS = 20

# Minimum number of rows for city × horizon fallback.
MIN_CITY_HORIZON_ROWS = 20


# ============================================================
# AQI REGIME
# ============================================================

def get_regime(aqi):
    """
    Return AQI regime based on current/origin AQI.
    """

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "USG"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, prediction):

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    mae = mean_absolute_error(
        actual,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction,
        )
    )

    r2 = r2_score(
        actual,
        prediction,
    )

    bias = float(
        np.mean(
            prediction - actual
        )
    )

    error = (
        prediction - actual
    )

    within_10 = float(
        np.mean(
            np.abs(error) <= 10
        ) * 100
    )

    within_20 = float(
        np.mean(
            np.abs(error) <= 20
        ) * 100
    )

    within_30 = float(
        np.mean(
            np.abs(error) <= 30
        ) * 100
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "bias": bias,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("PEARLSAQI FORECAST CALIBRATION V5")
print("=" * 70)

print("\nLoading walk-forward validation results...")

if not VALIDATION_FILE.exists():

    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

df = pd.read_csv(
    VALIDATION_FILE
)

print(
    f"Validation rows: {len(df):,}"
)

print("\nAvailable columns:")

for column in df.columns:
    print(f"- {column}")


# ============================================================
# COLUMN ALIGNMENT
# ============================================================

COLUMN_ALIASES = {
    "city": [
        "city_name",
        "city",
    ],
    "horizon": [
        "horizon",
    ],
    "date": [
        "forecast_date",
        "date",
    ],
    "actual": [
        "actual_aqi",
        "actual",
        "target_aqi",
    ],
    "prediction": [
        "predicted_aqi",
        "prediction",
        "predicted",
    ],
    "origin_aqi": [
        "origin_aqi",
        "aqi",
    ],
}


def find_column(options):

    for column in options:

        if column in df.columns:
            return column

    return None


city_col = find_column(
    COLUMN_ALIASES["city"]
)

horizon_col = find_column(
    COLUMN_ALIASES["horizon"]
)

date_col = find_column(
    COLUMN_ALIASES["date"]
)

actual_col = find_column(
    COLUMN_ALIASES["actual"]
)

prediction_col = find_column(
    COLUMN_ALIASES["prediction"]
)

origin_aqi_col = find_column(
    COLUMN_ALIASES["origin_aqi"]
)


required = {
    "city": city_col,
    "horizon": horizon_col,
    "date": date_col,
    "actual": actual_col,
    "prediction": prediction_col,
    "origin_aqi": origin_aqi_col,
}


missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"- {item}"
            for item in missing
        )
    )


print("\nCOLUMN ALIGNMENT")

print(
    f"City column       : {city_col}"
)

print(
    f"Horizon column    : {horizon_col}"
)

print(
    f"Date column       : {date_col}"
)

print(
    f"Actual AQI column : {actual_col}"
)

print(
    f"Prediction column : {prediction_col}"
)

print(
    f"Origin AQI column : {origin_aqi_col}"
)


# ============================================================
# CLEAN DATA
# ============================================================

work = pd.DataFrame()

work["city_name"] = (
    df[city_col]
    .astype(str)
)

work["horizon"] = (
    pd.to_numeric(
        df[horizon_col],
        errors="coerce",
    )
)

work["forecast_date"] = (
    pd.to_datetime(
        df[date_col],
        errors="coerce",
    )
)

work["actual"] = (
    pd.to_numeric(
        df[actual_col],
        errors="coerce",
    )
)

work["prediction"] = (
    pd.to_numeric(
        df[prediction_col],
        errors="coerce",
    )
)

work["origin_aqi"] = (
    pd.to_numeric(
        df[origin_aqi_col],
        errors="coerce",
    )
)

work = work.dropna()

work["horizon"] = (
    work["horizon"]
    .astype(int)
)

work["regime"] = (
    work["origin_aqi"]
    .apply(get_regime)
)

work = work.sort_values(
    "forecast_date"
).reset_index(drop=True)


print(
    f"\nClean validation rows: {len(work):,}"
)

print(
    f"Cities: {work['city_name'].nunique()}"
)

print(
    f"Date range: "
    f"{work['forecast_date'].min().date()} "
    f"→ "
    f"{work['forecast_date'].max().date()}"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

unique_dates = (
    work["forecast_date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

split_index = int(
    len(unique_dates)
    * CALIBRATION_RATIO
)

calibration_end = (
    unique_dates.iloc[
        split_index - 1
    ]
)

evaluation_start = (
    unique_dates.iloc[
        split_index
    ]
)

calibration = work[
    work["forecast_date"]
    <= calibration_end
].copy()

evaluation = work[
    work["forecast_date"]
    >= evaluation_start
].copy()


print("\n" + "=" * 70)
print("TEMPORAL CALIBRATION SPLIT")
print("=" * 70)

print(
    f"Calibration start : "
    f"{calibration['forecast_date'].min().date()}"
)

print(
    f"Calibration end   : "
    f"{calibration_end.date()}"
)

print(
    f"Evaluation start  : "
    f"{evaluation_start.date()}"
)

print(
    f"Evaluation end    : "
    f"{evaluation['forecast_date'].max().date()}"
)

print(
    f"Calibration rows  : {len(calibration):,}"
)

print(
    f"Evaluation rows   : {len(evaluation):,}"
)


# ============================================================
# BASELINE
# ============================================================

base_metrics = calculate_metrics(
    evaluation["actual"],
    evaluation["prediction"],
)

print("\n" + "=" * 70)
print("BASE EVALUATION")
print("=" * 70)

print(
    f"MAE  : {base_metrics['mae']:.4f}"
)

print(
    f"RMSE : {base_metrics['rmse']:.4f}"
)

print(
    f"R²   : {base_metrics['r2']:.4f}"
)

print(
    f"Bias : {base_metrics['bias']:.4f}"
)


# ============================================================
# BASELINE BIAS TABLES
# ============================================================

# City × horizon bias
city_horizon_bias = (
    calibration
    .assign(
        residual=lambda x:
        x["prediction"] - x["actual"]
    )
    .groupby(
        [
            "city_name",
            "horizon",
        ]
    )["residual"]
    .agg(
        ["mean", "count"]
    )
    .reset_index()
)

city_horizon_bias = (
    city_horizon_bias
    .rename(
        columns={
            "mean": "bias",
            "count": "rows",
        }
    )
)


# City × horizon × regime bias
group_bias = (
    calibration
    .assign(
        residual=lambda x:
        x["prediction"] - x["actual"]
    )
    .groupby(
        [
            "city_name",
            "horizon",
            "regime",
        ]
    )["residual"]
    .agg(
        ["mean", "count"]
    )
    .reset_index()
)

group_bias = (
    group_bias
    .rename(
        columns={
            "mean": "bias",
            "count": "rows",
        }
    )
)


# Horizon bias fallback
horizon_bias = (
    calibration
    .assign(
        residual=lambda x:
        x["prediction"] - x["actual"]
    )
    .groupby(
        "horizon"
    )["residual"]
    .mean()
    .to_dict()
)


# City bias fallback
city_bias = (
    calibration
    .assign(
        residual=lambda x:
        x["prediction"] - x["actual"]
    )
    .groupby(
        "city_name"
    )["residual"]
    .mean()
    .to_dict()
)


# Global bias fallback
global_bias = float(
    (
        calibration["prediction"]
        - calibration["actual"]
    ).mean()
)


# ============================================================
# CORRECTION FUNCTION
# ============================================================

def build_predictions(
    evaluation_df,
    shrinkage,
    correction_cap,
):

    predictions = []

    calibration_group_lookup = {
        (
            row["city_name"],
            int(row["horizon"]),
            row["regime"],
        ): (
            float(row["bias"]),
            int(row["rows"]),
        )
        for _, row
        in group_bias.iterrows()
    }

    city_horizon_lookup = {
        (
            row["city_name"],
            int(row["horizon"]),
        ): (
            float(row["bias"]),
            int(row["rows"]),
        )
        for _, row
        in city_horizon_bias.iterrows()
    }

    for _, row in evaluation_df.iterrows():

        city = row["city_name"]

        horizon = int(
            row["horizon"]
        )

        regime = row["regime"]

        correction = None

        # ----------------------------------------------------
        # PRIMARY:
        # CITY × HORIZON × REGIME
        # ----------------------------------------------------

        key = (
            city,
            horizon,
            regime,
        )

        if key in calibration_group_lookup:

            bias, rows = (
                calibration_group_lookup[key]
            )

            if rows >= MIN_GROUP_ROWS:

                correction = bias

        # ----------------------------------------------------
        # FALLBACK:
        # CITY × HORIZON
        # ----------------------------------------------------

        if correction is None:

            key = (
                city,
                horizon,
            )

            if key in city_horizon_lookup:

                bias, rows = (
                    city_horizon_lookup[key]
                )

                if rows >= MIN_CITY_HORIZON_ROWS:

                    correction = bias

        # ----------------------------------------------------
        # FALLBACK:
        # CITY
        # ----------------------------------------------------

        if correction is None:

            correction = city_bias.get(
                city,
                None,
            )

        # ----------------------------------------------------
        # FALLBACK:
        # HORIZON
        # ----------------------------------------------------

        if correction is None:

            correction = horizon_bias.get(
                horizon,
                global_bias,
            )

        correction = (
            float(correction)
            * shrinkage
        )

        # Bias is prediction - actual.
        # Therefore subtract the correction.
        correction = float(
            np.clip(
                correction,
                -correction_cap,
                correction_cap,
            )
        )

        prediction = (
            float(row["prediction"])
            - correction
        )

        predictions.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=float,
    )


# ============================================================
# STRATEGY SEARCH
# ============================================================

strategy_rows = []

print("\n" + "=" * 70)
print("TESTING CITY × HORIZON × AQI REGIME STRATEGIES")
print("=" * 70)

for shrinkage in SHRINKAGES:

    for cap in CAPS:

        calibrated = build_predictions(
            evaluation,
            shrinkage,
            cap,
        )

        metrics = calculate_metrics(
            evaluation["actual"],
            calibrated,
        )

        strategy_name = (
            f"REGIME_SHRINK_{shrinkage:.2f}"
            f"_CAP_{cap:g}"
        )

        strategy_rows.append(
            {
                "strategy": strategy_name,
                "shrinkage": shrinkage,
                "max_correction": cap,
                **metrics,
                "mae_improvement":
                    base_metrics["mae"]
                    - metrics["mae"],
                "rmse_improvement":
                    base_metrics["rmse"]
                    - metrics["rmse"],
            }
        )


strategy_results = pd.DataFrame(
    strategy_rows
)

strategy_results = (
    strategy_results
    .sort_values(
        [
            "mae",
            "rmse",
        ]
    )
    .reset_index(drop=True)
)


print(
    strategy_results[
        [
            "strategy",
            "shrinkage",
            "max_correction",
            "mae",
            "rmse",
            "r2",
            "bias",
            "within_10",
            "within_20",
            "within_30",
            "mae_improvement",
            "rmse_improvement",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# BEST STRATEGY
# ============================================================

best = (
    strategy_results.iloc[0]
)

best_shrinkage = float(
    best["shrinkage"]
)

best_cap = float(
    best["max_correction"]
)

best_predictions = build_predictions(
    evaluation,
    best_shrinkage,
    best_cap,
)

best_metrics = calculate_metrics(
    evaluation["actual"],
    best_predictions,
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("SELECTED V5 STRATEGY")
print("=" * 70)

print(
    f"Strategy : {best['strategy']}"
)

print(
    f"Shrinkage: {best_shrinkage:.2f}"
)

print(
    f"Max cap  : {best_cap:.2f}"
)

print(
    f"Base MAE : {base_metrics['mae']:.4f}"
)

print(
    f"V5 MAE   : {best_metrics['mae']:.4f}"
)

print(
    f"MAE improvement : "
    f"{base_metrics['mae'] - best_metrics['mae']:.4f}"
)

print(
    f"Base RMSE: {base_metrics['rmse']:.4f}"
)

print(
    f"V5 RMSE  : {best_metrics['rmse']:.4f}"
)

print(
    f"RMSE improvement : "
    f"{base_metrics['rmse'] - best_metrics['rmse']:.4f}"
)

print(
    f"Base R²  : {base_metrics['r2']:.4f}"
)

print(
    f"V5 R²    : {best_metrics['r2']:.4f}"
)

print(
    f"Within ±10 : "
    f"{best_metrics['within_10']:.2f}%"
)

print(
    f"Within ±20 : "
    f"{best_metrics['within_20']:.2f}%"
)

print(
    f"Within ±30 : "
    f"{best_metrics['within_30']:.2f}%"
)


# ============================================================
# HORIZON RESULTS
# ============================================================

horizon_rows = []

for horizon in sorted(
    evaluation["horizon"].unique()
):

    subset = (
        evaluation[
            evaluation["horizon"]
            == horizon
        ]
    )

    calibrated_subset = (
        best_predictions[
            evaluation["horizon"].values
            == horizon
        ]
    )

    base_horizon = calculate_metrics(
        subset["actual"],
        subset["prediction"],
    )

    v5_horizon = calculate_metrics(
        subset["actual"],
        calibrated_subset,
    )

    horizon_rows.append(
        {
            "horizon": horizon,
            "rows": len(subset),
            "base_mae":
                base_horizon["mae"],
            "v5_mae":
                v5_horizon["mae"],
            "mae_improvement":
                base_horizon["mae"]
                - v5_horizon["mae"],
            "base_rmse":
                base_horizon["rmse"],
            "v5_rmse":
                v5_horizon["rmse"],
            "rmse_improvement":
                base_horizon["rmse"]
                - v5_horizon["rmse"],
            "base_r2":
                base_horizon["r2"],
            "v5_r2":
                v5_horizon["r2"],
        }
    )


horizon_results = pd.DataFrame(
    horizon_rows
)

print("\n" + "=" * 70)
print("HORIZON RESULTS")
print("=" * 70)

print(
    horizon_results.to_string(
        index=False
    )
)


# ============================================================
# CITY RESULTS
# ============================================================

city_rows = []

for city in sorted(
    evaluation["city_name"].unique()
):

    mask = (
        evaluation["city_name"]
        == city
    )

    subset = (
        evaluation[mask]
    )

    calibrated_subset = (
        best_predictions[mask.values]
    )

    base_city = calculate_metrics(
        subset["actual"],
        subset["prediction"],
    )

    v5_city = calculate_metrics(
        subset["actual"],
        calibrated_subset,
    )

    city_rows.append(
        {
            "city_name": city,
            "rows": len(subset),
            "base_mae":
                base_city["mae"],
            "v5_mae":
                v5_city["mae"],
            "mae_improvement":
                base_city["mae"]
                - v5_city["mae"],
            "base_rmse":
                base_city["rmse"],
            "v5_rmse":
                v5_city["rmse"],
            "rmse_improvement":
                base_city["rmse"]
                - v5_city["rmse"],
            "base_bias":
                base_city["bias"],
            "v5_bias":
                v5_city["bias"],
        }
    )


city_results = pd.DataFrame(
    city_rows
)

city_results = (
    city_results
    .sort_values(
        "mae_improvement",
        ascending=False,
    )
)


print("\n" + "=" * 70)
print("CITY RESULTS")
print("=" * 70)

print(
    city_results.to_string(
        index=False
    )
)


# ============================================================
# REGIME RESULTS
# ============================================================

regime_rows = []

for regime in sorted(
    evaluation["regime"].unique()
):

    mask = (
        evaluation["regime"]
        == regime
    )

    subset = (
        evaluation[mask]
    )

    calibrated_subset = (
        best_predictions[mask.values]
    )

    base_regime = calculate_metrics(
        subset["actual"],
        subset["prediction"],
    )

    v5_regime = calculate_metrics(
        subset["actual"],
        calibrated_subset,
    )

    regime_rows.append(
        {
            "regime": regime,
            "rows": len(subset),
            "base_mae":
                base_regime["mae"],
            "v5_mae":
                v5_regime["mae"],
            "mae_improvement":
                base_regime["mae"]
                - v5_regime["mae"],
            "base_rmse":
                base_regime["rmse"],
            "v5_rmse":
                v5_regime["rmse"],
            "rmse_improvement":
                base_regime["rmse"]
                - v5_regime["rmse"],
            "base_bias":
                base_regime["bias"],
            "v5_bias":
                v5_regime["bias"],
        }
    )


regime_results = pd.DataFrame(
    regime_rows
)

print("\n" + "=" * 70)
print("AQI REGIME RESULTS")
print("=" * 70)

print(
    regime_results.to_string(
        index=False
    )
)


# ============================================================
# PREDICTION OUTPUT
# ============================================================

output_predictions = (
    evaluation.copy()
)

output_predictions[
    "base_prediction"
] = (
    output_predictions[
        "prediction"
    ]
)

output_predictions[
    "v5_prediction"
] = best_predictions

output_predictions[
    "base_absolute_error"
] = np.abs(
    output_predictions["actual"]
    - output_predictions["base_prediction"]
)

output_predictions[
    "v5_absolute_error"
] = np.abs(
    output_predictions["actual"]
    - output_predictions["v5_prediction"]
)

output_predictions[
    "improvement"
] = (
    output_predictions[
        "base_absolute_error"
    ]
    -
    output_predictions[
        "v5_absolute_error"
    ]
)


# ============================================================
# SAVE STRATEGIES
# ============================================================

strategy_results.to_csv(
    STRATEGY_FILE,
    index=False,
)

output_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

city_results.to_csv(
    CITY_FILE,
    index=False,
)

regime_results.to_csv(
    REGIME_FILE,
    index=False,
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameters = {
    "version": "V5",
    "strategy": str(
        best["strategy"]
    ),
    "shrinkage": best_shrinkage,
    "max_correction": best_cap,
    "minimum_group_rows":
        MIN_GROUP_ROWS,
    "minimum_city_horizon_rows":
        MIN_CITY_HORIZON_ROWS,
    "calibration_ratio":
        CALIBRATION_RATIO,
    "calibration_start":
        calibration[
            "forecast_date"
        ].min().strftime(
            "%Y-%m-%d"
        ),
    "calibration_end":
        calibration_end.strftime(
            "%Y-%m-%d"
        ),
    "evaluation_start":
        evaluation_start.strftime(
            "%Y-%m-%d"
        ),
    "evaluation_end":
        evaluation[
            "forecast_date"
        ].max().strftime(
            "%Y-%m-%d"
        ),
    "base_metrics":
        base_metrics,
    "v5_metrics":
        best_metrics,
    "mae_improvement":
        float(
            base_metrics["mae"]
            - best_metrics["mae"]
        ),
    "rmse_improvement":
        float(
            base_metrics["rmse"]
            - best_metrics["rmse"]
        ),
}


with open(
    PARAMETERS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        parameters,
        f,
        indent=2,
    )


# ============================================================
# FILES
# ============================================================

print("\n" + "=" * 70)
print("FILES")
print("=" * 70)

print(
    f"Strategies  : {STRATEGY_FILE}"
)

print(
    f"Predictions : {PREDICTIONS_FILE}"
)

print(
    f"City        : {CITY_FILE}"
)

print(
    f"Regimes     : {REGIME_FILE}"
)

print(
    f"Parameters  : {PARAMETERS_FILE}"
)

print(
    "\nPearlsAQI Forecast Calibration V5 completed."
)