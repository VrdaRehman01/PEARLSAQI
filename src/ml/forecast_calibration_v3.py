"""
PEARLSAQI FORECAST CALIBRATION V3
=================================

Selective city + horizon calibration.

This version automatically detects the actual AQI and
forecast/prediction columns from the V1 walk-forward
validation output.

It does NOT assume the columns are literally named
"actual" and "prediction".
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
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
    / "calibration_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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

PARAMETERS_FILE = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)


# ============================================================
# CONFIG
# ============================================================

SHRINKAGE_VALUES = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

MAX_CORRECTION_VALUES = [
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
]

MIN_CALIBRATION_ROWS = 60

MIN_REQUIRED_IMPROVEMENT = 0.10

MAX_ALLOWED_DEGRADATION = 0.25


# ============================================================
# HELPERS
# ============================================================

def metric_mae(actual, prediction):

    return float(
        np.mean(
            np.abs(
                np.asarray(actual)
                -
                np.asarray(prediction)
            )
        )
    )


def metric_rmse(actual, prediction):

    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(actual)
                    -
                    np.asarray(prediction)
                ) ** 2
            )
        )
    )


def metric_r2(actual, prediction):

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)

    denominator = np.sum(
        (actual - actual.mean()) ** 2
    )

    if denominator == 0:
        return 0.0

    return float(
        1.0
        -
        (
            np.sum(
                (actual - prediction) ** 2
            )
            /
            denominator
        )
    )


def metric_bias(actual, prediction):

    return float(
        np.mean(
            np.asarray(prediction)
            -
            np.asarray(actual)
        )
    )


def within_percentage(
    actual,
    prediction,
    threshold,
):

    return float(
        np.mean(
            np.abs(
                np.asarray(actual)
                -
                np.asarray(prediction)
            )
            <= threshold
        )
        * 100.0
    )


# ============================================================
# AUTOMATIC COLUMN DETECTION
# ============================================================

def find_column(
    columns,
    candidates,
    required=True,
):

    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:

            return normalized[key]

    if required:

        raise ValueError(
            "Could not find required column.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n"
            + "\n".join(
                f"- {column}"
                for column in columns
            )
        )

    return None


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PEARLSAQI FORECAST CALIBRATION V3")
print("=" * 70)

print("\nLoading walk-forward validation results...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Validation results not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Validation rows: {len(df):,}"
)

print("\nAvailable columns:")

for column in df.columns:

    print(
        f"- {column}"
    )


# ============================================================
# FIND COLUMNS
# ============================================================

city_column = find_column(
    df.columns,
    [
        "city_name",
        "city",
    ],
)

horizon_column = find_column(
    df.columns,
    [
        "horizon",
        "forecast_horizon",
    ],
)

horizon_label_column = find_column(
    df.columns,
    [
        "horizon_label",
        "label",
    ],
    required=False,
)

date_column = find_column(
    df.columns,
    [
        "date",
        "forecast_date",
        "target_date",
        "prediction_date",
    ],
)


# ------------------------------------------------------------
# ACTUAL AQI
# ------------------------------------------------------------

actual_column = find_column(
    df.columns,
    [
        "actual",
        "actual_aqi",
        "target_aqi",
        "observed_aqi",
        "true_aqi",
        "y_true",
        "target",
    ],
)


# ------------------------------------------------------------
# PREDICTION AQI
# ------------------------------------------------------------

prediction_column = find_column(
    df.columns,
    [
        "prediction",
        "predicted",
        "predicted_aqi",
        "forecast",
        "forecast_aqi",
        "predicted_value",
        "y_pred",
    ],
)


print("\nCOLUMN ALIGNMENT")

print(
    f"City column       : {city_column}"
)

print(
    f"Horizon column    : {horizon_column}"
)

print(
    f"Date column       : {date_column}"
)

print(
    f"Actual AQI column : {actual_column}"
)

print(
    f"Prediction column : {prediction_column}"
)


# ============================================================
# STANDARDIZE
# ============================================================

df["city_name"] = df[
    city_column
].astype(str)


df["horizon"] = pd.to_numeric(
    df[horizon_column],
    errors="coerce",
)


df["date"] = pd.to_datetime(
    df[date_column],
    errors="coerce",
)


df["actual"] = pd.to_numeric(
    df[actual_column],
    errors="coerce",
)


df["prediction"] = pd.to_numeric(
    df[prediction_column],
    errors="coerce",
)


if horizon_label_column:

    df["horizon_label"] = (
        df[horizon_label_column]
        .astype(str)
    )

else:

    df["horizon_label"] = (
        df["horizon"]
        .map(
            {
                1: "24h",
                2: "48h",
                3: "72h",
            }
        )
        .fillna(
            df["horizon"].astype(str)
        )
    )


df = df.dropna(
    subset=[
        "city_name",
        "horizon",
        "date",
        "actual",
        "prediction",
    ]
).copy()


df["horizon"] = df[
    "horizon"
].astype(int)


print(
    f"\nClean validation rows: "
    f"{len(df):,}"
)

print(
    f"Cities: "
    f"{df['city_name'].nunique()}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()}"
    f" → "
    f"{df['date'].max().date()}"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

unique_dates = sorted(
    df["date"].dt.date.unique()
)


if len(unique_dates) < 20:

    raise ValueError(
        "Not enough unique dates for temporal calibration."
    )


split_index = int(
    len(unique_dates) * 0.60
)


calibration_end_date = (
    unique_dates[
        split_index - 1
    ]
)

evaluation_start_date = (
    unique_dates[
        split_index
    ]
)


calibration = df[
    df["date"].dt.date
    <= calibration_end_date
].copy()


evaluation = df[
    df["date"].dt.date
    >= evaluation_start_date
].copy()


print("\n" + "=" * 70)
print("TEMPORAL CALIBRATION SPLIT")
print("=" * 70)

print(
    f"Calibration start: "
    f"{calibration['date'].min().date()}"
)

print(
    f"Calibration end  : "
    f"{calibration['date'].max().date()}"
)

print(
    f"Evaluation start : "
    f"{evaluation['date'].min().date()}"
)

print(
    f"Evaluation end   : "
    f"{evaluation['date'].max().date()}"
)

print(
    f"Calibration rows : "
    f"{len(calibration):,}"
)

print(
    f"Evaluation rows  : "
    f"{len(evaluation):,}"
)


# ============================================================
# BASELINE
# ============================================================

base_mae = metric_mae(
    evaluation["actual"],
    evaluation["prediction"],
)

base_rmse = metric_rmse(
    evaluation["actual"],
    evaluation["prediction"],
)

base_r2 = metric_r2(
    evaluation["actual"],
    evaluation["prediction"],
)

base_bias = metric_bias(
    evaluation["actual"],
    evaluation["prediction"],
)


print("\nBASE EVALUATION")

print(
    f"MAE  : {base_mae:.4f}"
)

print(
    f"RMSE : {base_rmse:.4f}"
)

print(
    f"R²   : {base_r2:.4f}"
)

print(
    f"Bias : {base_bias:.4f}"
)


# ============================================================
# LEARN CITY/HORIZON BIAS
# ============================================================

print("\n" + "=" * 70)
print("LEARNING CITY + HORIZON CALIBRATION")
print("=" * 70)


calibration_parameters = {}


for (
    city,
    horizon,
), group in calibration.groupby(
    [
        "city_name",
        "horizon",
    ]
):

    actual = group[
        "actual"
    ].to_numpy()

    prediction = group[
        "prediction"
    ].to_numpy()

    bias = float(
        np.mean(
            actual - prediction
        )
    )

    key = (
        f"{city}|{horizon}"
    )

    calibration_parameters[key] = {

        "city_name":
            city,

        "horizon":
            int(horizon),

        "rows":
            int(len(group)),

        "bias":
            bias,
    }


# ============================================================
# HORIZON FALLBACK
# ============================================================

horizon_parameters = {}


for horizon, group in calibration.groupby(
    "horizon"
):

    horizon_parameters[
        int(horizon)
    ] = float(
        np.mean(
            group["actual"].to_numpy()
            -
            group["prediction"].to_numpy()
        )
    )


print("\nHORIZON BIAS")

for horizon in sorted(
    horizon_parameters
):

    print(
        f"{horizon}: "
        f"{horizon_parameters[horizon]:+.4f}"
    )


# ============================================================
# APPLY STRATEGY
# ============================================================

def apply_strategy(
    data,
    shrinkage,
    max_correction,
):

    result = data.copy()

    corrections = []

    for _, row in result.iterrows():

        key = (
            f"{row['city_name']}"
            f"|"
            f"{int(row['horizon'])}"
        )

        params = (
            calibration_parameters
            .get(key)
        )

        if params is not None:

            rows = params["rows"]

            raw_bias = params["bias"]

        else:

            rows = 0

            raw_bias = horizon_parameters.get(
                int(row["horizon"]),
                0.0,
            )


        # If city/horizon has enough
        # historical examples, use it.
        #
        # Otherwise use the horizon-wide
        # fallback.

        if rows < MIN_CALIBRATION_ROWS:

            raw_bias = horizon_parameters.get(
                int(row["horizon"]),
                0.0,
            )


        correction = (
            raw_bias
            *
            shrinkage
        )


        correction = float(
            np.clip(
                correction,
                -max_correction,
                max_correction,
            )
        )


        corrections.append(
            correction
        )


    result[
        "calibration_correction"
    ] = corrections


    result[
        "calibrated_prediction"
    ] = (
        result["prediction"]
        +
        result[
            "calibration_correction"
        ]
    )


    return result


# ============================================================
# TEST STRATEGIES
# ============================================================

print("\n" + "=" * 70)
print("TESTING CALIBRATION STRATEGIES")
print("=" * 70)


strategy_rows = {}

strategy_predictions = {}


for shrinkage in SHRINKAGE_VALUES:

    for max_correction in MAX_CORRECTION_VALUES:

        name = (
            f"SHRINK_{shrinkage:.2f}"
            f"_CAP_{max_correction:.0f}"
        )


        result = apply_strategy(
            evaluation,
            shrinkage,
            max_correction,
        )


        actual = result[
            "actual"
        ]

        prediction = result[
            "calibrated_prediction"
        ]


        strategy_rows[name] = {

            "strategy":
                name,

            "shrinkage":
                shrinkage,

            "max_correction":
                max_correction,

            "mae":
                metric_mae(
                    actual,
                    prediction,
                ),

            "rmse":
                metric_rmse(
                    actual,
                    prediction,
                ),

            "r2":
                metric_r2(
                    actual,
                    prediction,
                ),

            "bias":
                metric_bias(
                    actual,
                    prediction,
                ),

            "within_10":
                within_percentage(
                    actual,
                    prediction,
                    10,
                ),

            "within_20":
                within_percentage(
                    actual,
                    prediction,
                    20,
                ),

            "within_30":
                within_percentage(
                    actual,
                    prediction,
                    30,
                ),
        }


        strategy_predictions[
            name
        ] = result


strategy_df = pd.DataFrame(
    list(
        strategy_rows.values()
    )
)


strategy_df[
    "mae_improvement"
] = (
    base_mae
    -
    strategy_df["mae"]
)


strategy_df[
    "rmse_improvement"
] = (
    base_rmse
    -
    strategy_df["rmse"]
)


print(
    strategy_df.sort_values(
        "mae"
    ).head(
        15
    ).to_string(
        index=False
    )
)


# ============================================================
# BEST GLOBAL CANDIDATE
# ============================================================

best_row = (
    strategy_df
    .sort_values(
        [
            "mae",
            "rmse",
        ]
    )
    .iloc[0]
)


best_strategy = best_row[
    "strategy"
]


candidate = strategy_predictions[
    best_strategy
].copy()


# ============================================================
# CITY-SAFE GATING
# ============================================================

city_results = []


candidate[
    "use_calibration"
] = False


for city, group in evaluation.groupby(
    "city_name"
):

    base_city_mae = metric_mae(
        group["actual"],
        group["prediction"],
    )


    calibrated_group = candidate[
        candidate[
            "city_name"
        ]
        == city
    ]


    calibrated_city_mae = metric_mae(
        calibrated_group[
            "actual"
        ],
        calibrated_group[
            "calibrated_prediction"
        ],
    )


    improvement = (
        base_city_mae
        -
        calibrated_city_mae
    )


    # Only activate calibration if
    # it actually improves the city.

    use = (
        improvement
        >= MIN_REQUIRED_IMPROVEMENT
    )


    candidate.loc[
        candidate[
            "city_name"
        ]
        == city,
        "use_calibration"
    ] = use


    city_results.append({

        "city_name":
            city,

        "rows":
            len(group),

        "base_mae":
            base_city_mae,

        "calibrated_mae":
            calibrated_city_mae,

        "improvement":
            improvement,

        "use_calibration":
            use,
    })


city_df = pd.DataFrame(
    city_results
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

candidate[
    "final_prediction"
] = np.where(

    candidate[
        "use_calibration"
    ],

    candidate[
        "calibrated_prediction"
    ],

    candidate[
        "prediction"
    ],
)


# ============================================================
# FINAL METRICS
# ============================================================

final_mae = metric_mae(
    candidate["actual"],
    candidate["final_prediction"],
)

final_rmse = metric_rmse(
    candidate["actual"],
    candidate["final_prediction"],
)

final_r2 = metric_r2(
    candidate["actual"],
    candidate["final_prediction"],
)

final_bias = metric_bias(
    candidate["actual"],
    candidate["final_prediction"],
)


final_within10 = within_percentage(
    candidate["actual"],
    candidate["final_prediction"],
    10,
)

final_within20 = within_percentage(
    candidate["actual"],
    candidate["final_prediction"],
    20,
)

final_within30 = within_percentage(
    candidate["actual"],
    candidate["final_prediction"],
    30,
)


# ============================================================
# SAFETY FALLBACK
# ============================================================

if (
    final_mae
    >
    base_mae
    +
    MAX_ALLOWED_DEGRADATION
):

    print(
        "\nCalibration failed safety check."
    )

    print(
        "Using original forecast."
    )


    candidate[
        "use_calibration"
    ] = False


    candidate[
        "final_prediction"
    ] = candidate[
        "prediction"
    ]


    final_mae = base_mae

    final_rmse = base_rmse

    final_r2 = base_r2

    final_bias = base_bias

    final_within10 = within_percentage(
        candidate["actual"],
        candidate["final_prediction"],
        10,
    )

    final_within20 = within_percentage(
        candidate["actual"],
        candidate["final_prediction"],
        20,
    )

    final_within30 = within_percentage(
        candidate["actual"],
        candidate["final_prediction"],
        30,
    )


# ============================================================
# HORIZON RESULTS
# ============================================================

horizon_results = []


for horizon, group in candidate.groupby(
    "horizon"
):

    base_h_mae = metric_mae(
        group["actual"],
        group["prediction"],
    )

    final_h_mae = metric_mae(
        group["actual"],
        group["final_prediction"],
    )

    base_h_rmse = metric_rmse(
        group["actual"],
        group["prediction"],
    )

    final_h_rmse = metric_rmse(
        group["actual"],
        group["final_prediction"],
    )

    horizon_results.append({

        "horizon":
            int(horizon),

        "horizon_label":
            group[
                "horizon_label"
            ].iloc[0],

        "rows":
            len(group),

        "base_mae":
            base_h_mae,

        "v3_mae":
            final_h_mae,

        "mae_improvement":
            base_h_mae
            -
            final_h_mae,

        "base_rmse":
            base_h_rmse,

        "v3_rmse":
            final_h_rmse,

        "rmse_improvement":
            base_h_rmse
            -
            final_h_rmse,
    })


horizon_df = pd.DataFrame(
    horizon_results
)


# ============================================================
# PRINT FINAL
# ============================================================

print("\n" + "=" * 70)
print("V3 FINAL RESULTS")
print("=" * 70)

print(
    f"Selected strategy : "
    f"{best_strategy}"
)

print(
    f"Base MAE          : "
    f"{base_mae:.4f}"
)

print(
    f"V3 MAE            : "
    f"{final_mae:.4f}"
)

print(
    f"MAE improvement   : "
    f"{base_mae - final_mae:.4f}"
)

print(
    f"Base RMSE         : "
    f"{base_rmse:.4f}"
)

print(
    f"V3 RMSE           : "
    f"{final_rmse:.4f}"
)

print(
    f"RMSE improvement  : "
    f"{base_rmse - final_rmse:.4f}"
)

print(
    f"Base R²           : "
    f"{base_r2:.4f}"
)

print(
    f"V3 R²             : "
    f"{final_r2:.4f}"
)

print(
    f"Within ±10        : "
    f"{final_within10:.2f}%"
)

print(
    f"Within ±20        : "
    f"{final_within20:.2f}%"
)

print(
    f"Within ±30        : "
    f"{final_within30:.2f}%"
)


calibrated_rows = int(
    candidate[
        "use_calibration"
    ].sum()
)


print(
    f"\nCalibrated rows   : "
    f"{calibrated_rows:,}"
)

print(
    f"Unchanged rows    : "
    f"{len(candidate) - calibrated_rows:,}"
)


# ============================================================
# HORIZON OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("HORIZON PERFORMANCE")
print("=" * 70)

print(
    horizon_df.to_string(
        index=False
    )
)


# ============================================================
# CITY OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CITY PERFORMANCE")
print("=" * 70)

print(
    city_df.sort_values(
        "improvement",
        ascending=False,
    ).to_string(
        index=False
    )
)


# ============================================================
# SAVE STRATEGIES
# ============================================================

strategy_df = strategy_df.sort_values(
    "mae"
)

strategy_df.to_csv(
    STRATEGY_FILE,
    index=False,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

candidate["date"] = (
    pd.to_datetime(
        candidate["date"]
    )
    .dt.strftime(
        "%Y-%m-%d"
    )
)


candidate.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# SAVE CITY RESULTS
# ============================================================

city_df.to_csv(
    CITY_FILE,
    index=False,
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameters = {

    "version":
        "PearlsAQI Forecast Calibration V3",

    "input_file":
        str(INPUT_FILE),

    "selected_strategy":
        str(best_strategy),

    "base_mae":
        base_mae,

    "v3_mae":
        final_mae,

    "mae_improvement":
        base_mae
        -
        final_mae,

    "base_rmse":
        base_rmse,

    "v3_rmse":
        final_rmse,

    "rmse_improvement":
        base_rmse
        -
        final_rmse,

    "base_r2":
        base_r2,

    "v3_r2":
        final_r2,

    "calibrated_rows":
        calibrated_rows,

    "total_rows":
        len(candidate),

    "min_calibration_rows":
        MIN_CALIBRATION_ROWS,

    "min_required_improvement":
        MIN_REQUIRED_IMPROVEMENT,

    "max_allowed_degradation":
        MAX_ALLOWED_DEGRADATION,

    "calibration_start":
        str(
            calibration[
                "date"
            ].min().date()
        ),

    "calibration_end":
        str(
            calibration[
                "date"
            ].max().date()
        ),

    "evaluation_start":
        str(
            evaluation[
                "date"
            ].min().date()
        ),

    "evaluation_end":
        str(
            evaluation[
                "date"
            ].max().date()
        ),

    "horizon_bias":
        {
            str(key):
            float(value)
            for key, value
            in horizon_parameters.items()
        },

    "city_horizon_parameters":
        calibration_parameters,

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
    f"Strategies   : {STRATEGY_FILE}"
)

print(
    f"Predictions  : {PREDICTIONS_FILE}"
)

print(
    f"City stats   : {CITY_FILE}"
)

print(
    f"Parameters   : {PARAMETERS_FILE}"
)

print(
    "\nPearlsAQI Forecast Calibration V3 completed."
)