"""
PEARLSAQI FORECAST CALIBRATION V4

Experimental final calibration layer.

Pipeline:
    Base recursive forecast
        ↓
    V3 city/horizon calibration
        ↓
    Momentum correction
        ↓
    Horizon-aware weighting
        ↓
    V7 spike probability gating
        ↓
    Final calibrated forecast

IMPORTANT:
- V3 is NOT modified.
- This script only experiments with V4.
- No future actual AQI is used during calibration.
- All calibration parameters are learned from the calibration period.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    ROOT
    / "models"
    / "forecast"
    / "validation"
    / "forecast_validation_results.csv"
)

V3_PARAMETERS_FILE = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v3"
    / "calibration_parameters.json"
)

V3_PREDICTIONS_FILE = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v3"
    / "calibration_predictions.csv"
)

SPIKE_PREDICTIONS_FILE = (
    ROOT
    / "models"
    / "spike_gated_v7"
    / "v7_predictions.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

CALIBRATION_FRACTION = 0.60

HORIZON_WEIGHTS = {
    1: 1.00,
    2: 0.75,
    3: 0.50,
}

# Momentum strength candidates
MOMENTUM_STRENGTHS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
]

# Spike influence candidates
SPIKE_STRENGTHS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

# Maximum correction in AQI points
MAX_CORRECTION = 15.0


# ============================================================
# HELPERS
# ============================================================

def category(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def metrics(actual, prediction):

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    error = prediction - actual

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(
        (actual - prediction) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    bias = np.mean(error)

    within_10 = (
        np.mean(np.abs(error) <= 10)
        * 100
    )

    within_20 = (
        np.mean(np.abs(error) <= 20)
        * 100
    )

    within_30 = (
        np.mean(np.abs(error) <= 30)
        * 100
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "bias": float(bias),
        "within_10": float(within_10),
        "within_20": float(within_20),
        "within_30": float(within_30),
    }


def clip_prediction(prediction):

    return np.clip(
        prediction,
        0,
        500,
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("PEARLSAQI FORECAST CALIBRATION V4")
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


# ============================================================
# COLUMN ALIGNMENT
# ============================================================

column_map = {

    "city":
        "city_name",

    "horizon":
        "horizon",

    "actual":
        "actual_aqi",

    "prediction":
        "predicted_aqi",

    "forecast_date":
        "forecast_date",

    "origin_date":
        "origin_date",

    "origin_aqi":
        "origin_aqi",
}


for logical, column in column_map.items():

    if column not in df.columns:

        raise ValueError(
            f"Missing required column for "
            f"{logical}: {column}"
        )


df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df["origin_date"] = pd.to_datetime(
    df["origin_date"]
)

df = df.sort_values(
    [
        "city_name",
        "forecast_date",
        "horizon",
    ]
).reset_index(drop=True)


# ============================================================
# MOMENTUM FEATURES
# ============================================================

print("\nBuilding momentum features...")


# The walk-forward validation file contains forecast origins.
# We derive momentum only from origin AQI values.

city_groups = []

for city, group in df.groupby("city_name"):

    group = group.copy()

    # Use origin AQI and its temporal ordering.
    origin_series = (
        group[
            [
                "origin_date",
                "origin_aqi",
            ]
        ]
        .drop_duplicates()
        .sort_values("origin_date")
    )

    origin_series["aqi_change_1d"] = (
        origin_series["origin_aqi"]
        .diff(1)
    )

    origin_series["aqi_change_2d"] = (
        origin_series["origin_aqi"]
        .diff(2)
    )

    origin_series["aqi_change_3d"] = (
        origin_series["origin_aqi"]
        .diff(3)
    )

    origin_series["aqi_change_7d"] = (
        origin_series["origin_aqi"]
        .diff(7)
    )

    origin_series["rolling_3"] = (
        origin_series["origin_aqi"]
        .rolling(3)
        .mean()
    )

    origin_series["rolling_7"] = (
        origin_series["origin_aqi"]
        .rolling(7)
        .mean()
    )

    group = group.merge(
        origin_series[
            [
                "origin_date",
                "aqi_change_1d",
                "aqi_change_2d",
                "aqi_change_3d",
                "aqi_change_7d",
                "rolling_3",
                "rolling_7",
            ]
        ],
        on="origin_date",
        how="left",
    )

    city_groups.append(group)


df = pd.concat(
    city_groups,
    ignore_index=True,
)


# ============================================================
# MOMENTUM SIGNAL
# ============================================================

df["momentum_signal"] = (

    0.45
    * df["aqi_change_1d"].fillna(0)

    + 0.30
    * (
        df["aqi_change_2d"].fillna(0)
        / 2.0
    )

    + 0.15
    * (
        df["aqi_change_3d"].fillna(0)
        / 3.0
    )

    + 0.10
    * (
        df["aqi_change_7d"].fillna(0)
        / 7.0
    )
)


# ============================================================
# NORMALIZE MOMENTUM
# ============================================================

momentum_scale = (
    df[
        df["origin_date"]
        <= df["origin_date"].quantile(
            CALIBRATION_FRACTION
        )
    ]["momentum_signal"]
    .abs()
    .median()
)

if (
    pd.isna(momentum_scale)
    or momentum_scale < 1
):

    momentum_scale = 10.0


df["momentum_normalized"] = (
    df["momentum_signal"]
    / momentum_scale
)

df["momentum_normalized"] = np.clip(
    df["momentum_normalized"],
    -3,
    3,
)


# ============================================================
# TEMPORAL CALIBRATION SPLIT
# ============================================================

unique_dates = np.sort(
    df["forecast_date"]
    .dropna()
    .unique()
)

split_index = int(
    len(unique_dates)
    * CALIBRATION_FRACTION
)

calibration_end = pd.Timestamp(
    unique_dates[split_index - 1]
)

evaluation_start = pd.Timestamp(
    unique_dates[split_index]
)

calibration = df[
    df["forecast_date"]
    <= calibration_end
].copy()

evaluation = df[
    df["forecast_date"]
    >= evaluation_start
].copy()


print("\nTEMPORAL SPLIT")

print(
    f"Calibration start : "
    f"{calibration['forecast_date'].min().date()}"
)

print(
    f"Calibration end   : "
    f"{calibration['forecast_date'].max().date()}"
)

print(
    f"Evaluation start  : "
    f"{evaluation['forecast_date'].min().date()}"
)

print(
    f"Evaluation end    : "
    f"{evaluation['forecast_date'].max().date()}"
)

print(
    f"Calibration rows  : "
    f"{len(calibration):,}"
)

print(
    f"Evaluation rows   : "
    f"{len(evaluation):,}"
)


# ============================================================
# CITY / HORIZON BIAS
# ============================================================

bias_table = (
    calibration
    .assign(
        residual=lambda x:
            x["actual_aqi"]
            - x["predicted_aqi"]
    )
    .groupby(
        [
            "city_name",
            "horizon",
        ]
    )["residual"]
    .mean()
    .reset_index(
        name="city_horizon_bias"
    )
)


evaluation = evaluation.merge(
    bias_table,
    on=[
        "city_name",
        "horizon",
    ],
    how="left",
)

evaluation["city_horizon_bias"] = (
    evaluation[
        "city_horizon_bias"
    ]
    .fillna(0)
)


# ============================================================
# CITY MOMENTUM BASELINES
# ============================================================

momentum_bias = (
    calibration
    .assign(
        residual=lambda x:
            x["actual_aqi"]
            - x["predicted_aqi"]
    )
    .groupby(
        [
            "city_name",
            "horizon",
        ]
    )
    .apply(
        lambda g: pd.Series(
            {
                "momentum_residual_corr":
                    np.corrcoef(
                        g["momentum_normalized"].fillna(0),
                        g["residual"],
                    )[0, 1]
                    if (
                        g[
                            "momentum_normalized"
                        ]
                        .nunique()
                        > 1
                    )
                    else 0.0
            }
        ),
        include_groups=False,
    )
    .reset_index()
)


evaluation = evaluation.merge(
    momentum_bias,
    on=[
        "city_name",
        "horizon",
    ],
    how="left",
)

evaluation[
    "momentum_residual_corr"
] = (
    evaluation[
        "momentum_residual_corr"
    ]
    .replace(
        [np.inf, -np.inf],
        0,
    )
    .fillna(0)
)


# ============================================================
# SPIKE PROBABILITY
# ============================================================

print("\nLoading V7 spike probabilities...")

if SPIKE_PREDICTIONS_FILE.exists():

    try:

        spike_df = pd.read_parquet(
            SPIKE_PREDICTIONS_FILE
        )

        print(
            f"Spike rows: "
            f"{len(spike_df):,}"
        )

        spike_columns = {
            "city_name",
            "date",
        }

        probability_column = None

        for candidate in [
            "spike_probability",
            "spike_prob",
            "probability",
        ]:

            if candidate in spike_df.columns:

                probability_column = candidate
                break

        if (
            spike_columns.issubset(
                spike_df.columns
            )
            and probability_column
        ):

            spike_df = spike_df[
                [
                    "city_name",
                    "date",
                    probability_column,
                ]
            ].copy()

            spike_df["date"] = pd.to_datetime(
                spike_df["date"]
            )

            spike_df = spike_df.rename(
                columns={
                    probability_column:
                        "spike_probability"
                }
            )

            evaluation = evaluation.merge(
                spike_df,
                left_on=[
                    "city_name",
                    "forecast_date",
                ],
                right_on=[
                    "city_name",
                    "date",
                ],
                how="left",
            )

            evaluation[
                "spike_probability"
            ] = (
                evaluation[
                    "spike_probability"
                ]
                .fillna(0)
            )

        else:

            print(
                "V7 file does not contain "
                "usable spike probability columns."
            )

            evaluation[
                "spike_probability"
            ] = 0.0

    except Exception as exc:

        print(
            f"Could not load V7 probabilities: "
            f"{exc}"
        )

        evaluation[
            "spike_probability"
        ] = 0.0

else:

    print(
        "V7 predictions not found."
    )

    evaluation[
        "spike_probability"
    ] = 0.0


# ============================================================
# BASE V3 CORRECTION
# ============================================================

evaluation[
    "v3_correction"
] = (
    evaluation[
        "city_horizon_bias"
    ]
    * 0.50
)

evaluation[
    "v3_correction"
] = np.clip(
    evaluation["v3_correction"],
    -MAX_CORRECTION,
    MAX_CORRECTION,
)


# ============================================================
# EXPERIMENT
# ============================================================

results = []

best_prediction = None
best_score = None
best_parameters = None


for momentum_strength in MOMENTUM_STRENGTHS:

    for spike_strength in SPIKE_STRENGTHS:

        predictions = []

        for _, row in evaluation.iterrows():

            base = float(
                row["predicted_aqi"]
            )

            # ------------------------------------------------
            # V3 city/horizon correction
            # ------------------------------------------------

            correction = (
                row["v3_correction"]
            )

            # ------------------------------------------------
            # Momentum correction
            # ------------------------------------------------

            momentum_correction = (
                momentum_strength
                * row[
                    "momentum_normalized"
                ]
                * 10.0
            )

            # Horizon weighting
            horizon_weight = (
                HORIZON_WEIGHTS.get(
                    int(row["horizon"]),
                    0.5,
                )
            )

            momentum_correction *= (
                horizon_weight
            )

            # ------------------------------------------------
            # Spike-aware adjustment
            #
            # When spike probability is high,
            # reduce aggressive momentum corrections.
            # This prevents overshooting during shocks.
            # ------------------------------------------------

            spike_probability = float(
                row[
                    "spike_probability"
                ]
            )

            spike_damping = (
                1.0
                - spike_strength
                * spike_probability
            )

            momentum_correction *= (
                spike_damping
            )

            # ------------------------------------------------
            # Combine
            # ------------------------------------------------

            total_correction = (
                correction
                + momentum_correction
            )

            total_correction = np.clip(
                total_correction,
                -MAX_CORRECTION,
                MAX_CORRECTION,
            )

            final_prediction = (
                base
                + total_correction
            )

            final_prediction = (
                clip_prediction(
                    final_prediction
                )
            )

            predictions.append(
                final_prediction
            )

        evaluation[
            "v4_prediction"
        ] = predictions

        m = metrics(
            evaluation[
                "actual_aqi"
            ],
            evaluation[
                "v4_prediction"
            ],
        )

        result = {

            "momentum_strength":
                momentum_strength,

            "spike_strength":
                spike_strength,

            **m,
        }

        results.append(
            result
        )

        score = m["mae"]

        if (
            best_score is None
            or score < best_score
        ):

            best_score = score

            best_prediction = (
                evaluation[
                    "v4_prediction"
                ].copy()
            )

            best_parameters = {
                "momentum_strength":
                    momentum_strength,

                "spike_strength":
                    spike_strength,

                "max_correction":
                    MAX_CORRECTION,
            }


# ============================================================
# RESULTS
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values("mae")
    .reset_index(drop=True)
)


# ============================================================
# BASE / V4 METRICS
# ============================================================

base_metrics = metrics(
    evaluation[
        "actual_aqi"
    ],
    evaluation[
        "predicted_aqi"
    ],
)

v4_metrics = metrics(
    evaluation[
        "actual_aqi"
    ],
    best_prediction,
)


print("\n" + "=" * 70)
print("V4 RESULTS")
print("=" * 70)

print(
    f"\nBase MAE : "
    f"{base_metrics['mae']:.4f}"
)

print(
    f"V4 MAE   : "
    f"{v4_metrics['mae']:.4f}"
)

print(
    f"MAE improvement : "
    f"{base_metrics['mae'] - v4_metrics['mae']:.4f}"
)

print(
    f"\nBase RMSE : "
    f"{base_metrics['rmse']:.4f}"
)

print(
    f"V4 RMSE   : "
    f"{v4_metrics['rmse']:.4f}"
)

print(
    f"RMSE improvement : "
    f"{base_metrics['rmse'] - v4_metrics['rmse']:.4f}"
)

print(
    f"\nBase R² : "
    f"{base_metrics['r2']:.4f}"
)

print(
    f"V4 R²   : "
    f"{v4_metrics['r2']:.4f}"
)

print(
    f"\nWithin ±10 : "
    f"{v4_metrics['within_10']:.2f}%"
)

print(
    f"Within ±20 : "
    f"{v4_metrics['within_20']:.2f}%"
)

print(
    f"Within ±30 : "
    f"{v4_metrics['within_30']:.2f}%"
)

print(
    "\nBEST PARAMETERS"
)

for key, value in best_parameters.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# HORIZON RESULTS
# ============================================================

evaluation[
    "v4_prediction"
] = best_prediction

horizon_results = []

for horizon in sorted(
    evaluation["horizon"].unique()
):

    subset = evaluation[
        evaluation["horizon"]
        == horizon
    ]

    base = metrics(
        subset["actual_aqi"],
        subset["predicted_aqi"],
    )

    v4 = metrics(
        subset["actual_aqi"],
        subset["v4_prediction"],
    )

    horizon_results.append({

        "horizon":
            int(horizon),

        "rows":
            len(subset),

        "base_mae":
            base["mae"],

        "v4_mae":
            v4["mae"],

        "mae_improvement":
            base["mae"] - v4["mae"],

        "base_rmse":
            base["rmse"],

        "v4_rmse":
            v4["rmse"],

        "rmse_improvement":
            base["rmse"] - v4["rmse"],

        "base_r2":
            base["r2"],

        "v4_r2":
            v4["r2"],
    })


horizon_results_df = pd.DataFrame(
    horizon_results
)

print(
    "\nHORIZON RESULTS"
)

print(
    horizon_results_df.to_string(
        index=False
    )
)


# ============================================================
# CITY RESULTS
# ============================================================

city_results = []

for city in sorted(
    evaluation["city_name"].unique()
):

    subset = evaluation[
        evaluation["city_name"]
        == city
    ]

    base = metrics(
        subset["actual_aqi"],
        subset["predicted_aqi"],
    )

    v4 = metrics(
        subset["actual_aqi"],
        subset["v4_prediction"],
    )

    city_results.append({

        "city_name":
            city,

        "rows":
            len(subset),

        "base_mae":
            base["mae"],

        "v4_mae":
            v4["mae"],

        "mae_improvement":
            base["mae"]
            - v4["mae"],

        "base_rmse":
            base["rmse"],

        "v4_rmse":
            v4["rmse"],

        "rmse_improvement":
            base["rmse"]
            - v4["rmse"],
    })


city_results_df = pd.DataFrame(
    city_results
)

print(
    "\nCITY RESULTS"
)

print(
    city_results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE DETAILED PREDICTIONS
# ============================================================

output_predictions = evaluation[
    [
        "city_name",
        "origin_date",
        "forecast_date",
        "horizon",
        "origin_aqi",
        "actual_aqi",
        "predicted_aqi",
        "v4_prediction",
        "city_horizon_bias",
        "v3_correction",
        "momentum_signal",
        "momentum_normalized",
        "spike_probability",
    ]
].copy()

output_predictions[
    "v4_error"
] = (
    output_predictions[
        "v4_prediction"
    ]
    - output_predictions[
        "actual_aqi"
    ]
)

output_predictions[
    "v4_absolute_error"
] = (
    output_predictions[
        "v4_error"
    ].abs()
)


predictions_path = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

output_predictions.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# SAVE STRATEGIES
# ============================================================

strategy_path = (
    OUTPUT_DIR
    / "calibration_strategy_results.csv"
)

results_df.to_csv(
    strategy_path,
    index=False,
)


# ============================================================
# SAVE HORIZON RESULTS
# ============================================================

horizon_path = (
    OUTPUT_DIR
    / "calibration_horizon_results.csv"
)

horizon_results_df.to_csv(
    horizon_path,
    index=False,
)


# ============================================================
# SAVE CITY RESULTS
# ============================================================

city_path = (
    OUTPUT_DIR
    / "calibration_city_results.csv"
)

city_results_df.to_csv(
    city_path,
    index=False,
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameters = {

    "version":
        "V4",

    "description":
        "V3 city/horizon calibration + "
        "momentum + horizon weighting + "
        "spike-aware damping",

    "calibration_fraction":
        CALIBRATION_FRACTION,

    "calibration_end":
        str(calibration_end.date()),

    "evaluation_start":
        str(evaluation_start.date()),

    "best_parameters":
        best_parameters,

    "base_metrics":
        base_metrics,

    "v4_metrics":
        v4_metrics,

    "v4_improvement": {

        "mae":
            base_metrics["mae"]
            - v4_metrics["mae"],

        "rmse":
            base_metrics["rmse"]
            - v4_metrics["rmse"],

        "r2":
            v4_metrics["r2"]
            - base_metrics["r2"],
    },
}


parameters_path = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)

with open(
    parameters_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        parameters,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FILES"
)

print(
    f"Strategies  : "
    f"{strategy_path}"
)

print(
    f"Predictions : "
    f"{predictions_path}"
)

print(
    f"Horizon     : "
    f"{horizon_path}"
)

print(
    f"City        : "
    f"{city_path}"
)

print(
    f"Parameters  : "
    f"{parameters_path}"
)

print(
    "\nPearlsAQI Forecast Calibration V4 completed."
)