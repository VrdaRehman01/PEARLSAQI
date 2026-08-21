"""
PEARLSAQI FORECAST CALIBRATION V7
==================================

Shock-Aware Adaptive Calibration

V7 is an experimental calibration layer.

IMPORTANT:
- V5 remains untouched.
- V6 remains untouched.
- Uses the same walk-forward validation dataset.
- Uses rolling temporal folds for robustness.
- Separately handles:
    1. Normal AQI
    2. Upward movement
    3. Downward movement
    4. Upward spikes
    5. Downward spikes
    6. Extreme AQI

Primary goal:
Improve the weaknesses identified in V6:

    - upward spikes became worse
    - extreme AQI became worse
    - 24h calibration was slightly worse

V7 therefore does NOT blindly apply the same regime correction
to every observation.
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
    / "calibration_v7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STRATEGY_FILE = (
    OUTPUT_DIR
    / "calibration_strategy_results.csv"
)

FOLD_FILE = (
    OUTPUT_DIR
    / "calibration_fold_results.csv"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

HORIZON_FILE = (
    OUTPUT_DIR
    / "calibration_horizon_results.csv"
)

CITY_FILE = (
    OUTPUT_DIR
    / "calibration_city_results.csv"
)

STRESS_FILE = (
    OUTPUT_DIR
    / "calibration_stress_results.csv"
)

PARAMETER_FILE = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)


# ============================================================
# PARAMETERS
# ============================================================

SPIKE_THRESHOLD = 40.0

EXTREME_AQI_THRESHOLD = 300.0

HIGH_AQI_THRESHOLD = 200.0

UPWARD_MOVEMENT_THRESHOLD = 10.0

DOWNWARD_MOVEMENT_THRESHOLD = -10.0

# Rolling temporal validation
CALIBRATION_MONTHS = 18
EVALUATION_MONTHS = 6
STEP_MONTHS = 3

# Candidate parameters
SHRINKAGES = [
    0.25,
    0.50,
    0.75,
    1.00,
]

MAX_CORRECTIONS = [
    5.0,
    10.0,
    15.0,
]

# V7-specific shock strengths
UPWARD_SPIKE_STRENGTHS = [
    0.00,
    0.15,
    0.30,
    0.50,
]

DOWNWARD_SPIKE_STRENGTHS = [
    0.00,
    0.10,
    0.20,
]

EXTREME_STRENGTHS = [
    0.00,
    0.15,
    0.30,
]


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("PEARLSAQI FORECAST CALIBRATION V7")
print("=" * 70)

print()
print("Shock-Aware Adaptive Calibration")
print()
print("V5 remains untouched.")
print("V6 remains untouched.")
print()
print("Targets:")
print("- upward spikes")
print("- downward spikes")
print("- extreme AQI")
print("- horizon-specific drift")
print("- regime bias")
print()


# ============================================================
# LOAD DATA
# ============================================================

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

print("Loading walk-forward validation results...")

df = pd.read_csv(
    VALIDATION_FILE
)

print(
    f"Validation rows: {len(df):,}"
)

print()
print("Available columns:")

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


def find_column(
    aliases,
    required=True,
):

    for column in aliases:

        if column in df.columns:
            return column

    if required:
        raise ValueError(
            "Missing required column. "
            f"Expected one of: {aliases}"
        )

    return None


CITY_COLUMN = find_column(
    COLUMN_ALIASES["city"]
)

HORIZON_COLUMN = find_column(
    COLUMN_ALIASES["horizon"]
)

DATE_COLUMN = find_column(
    COLUMN_ALIASES["date"]
)

ACTUAL_COLUMN = find_column(
    COLUMN_ALIASES["actual"]
)

PREDICTION_COLUMN = find_column(
    COLUMN_ALIASES["prediction"]
)

ORIGIN_COLUMN = find_column(
    COLUMN_ALIASES["origin_aqi"]
)


print()
print("COLUMN ALIGNMENT")

print(
    f"City column       : {CITY_COLUMN}"
)

print(
    f"Horizon column    : {HORIZON_COLUMN}"
)

print(
    f"Date column       : {DATE_COLUMN}"
)

print(
    f"Actual AQI column : {ACTUAL_COLUMN}"
)

print(
    f"Prediction column : {PREDICTION_COLUMN}"
)

print(
    f"Origin AQI column : {ORIGIN_COLUMN}"
)


# ============================================================
# CLEAN DATA
# ============================================================

work = df.copy()

work[DATE_COLUMN] = pd.to_datetime(
    work[DATE_COLUMN],
    errors="coerce",
)

work[ACTUAL_COLUMN] = pd.to_numeric(
    work[ACTUAL_COLUMN],
    errors="coerce",
)

work[PREDICTION_COLUMN] = pd.to_numeric(
    work[PREDICTION_COLUMN],
    errors="coerce",
)

work[ORIGIN_COLUMN] = pd.to_numeric(
    work[ORIGIN_COLUMN],
    errors="coerce",
)

work[HORIZON_COLUMN] = pd.to_numeric(
    work[HORIZON_COLUMN],
    errors="coerce",
)

work = work.dropna(
    subset=[
        CITY_COLUMN,
        HORIZON_COLUMN,
        DATE_COLUMN,
        ACTUAL_COLUMN,
        PREDICTION_COLUMN,
        ORIGIN_COLUMN,
    ]
).copy()

work = work.sort_values(
    [
        CITY_COLUMN,
        DATE_COLUMN,
        HORIZON_COLUMN,
    ]
).reset_index(
    drop=True
)

print()
print(
    f"Clean validation rows: {len(work):,}"
)

print(
    f"Cities: {work[CITY_COLUMN].nunique()}"
)

print(
    "Date range: "
    f"{work[DATE_COLUMN].min().date()} "
    "→ "
    f"{work[DATE_COLUMN].max().date()}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

work["actual"] = work[ACTUAL_COLUMN]

work["prediction"] = work[PREDICTION_COLUMN]

work["origin_aqi_value"] = work[ORIGIN_COLUMN]

work["residual"] = (
    work["actual"]
    - work["prediction"]
)

work["absolute_error"] = (
    work["residual"].abs()
)

work["movement"] = (
    work["actual"]
    - work["origin_aqi_value"]
)

work["prediction_movement"] = (
    work["prediction"]
    - work["origin_aqi_value"]
)

# Prediction error direction
work["predicted_change_error"] = (
    work["actual"]
    - work["prediction"]
)


# ============================================================
# REGIMES
# ============================================================

def get_regime(aqi):

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


work["regime"] = work[
    "origin_aqi_value"
].apply(
    get_regime
)


# ============================================================
# SHOCK CLASSIFICATION
# ============================================================

work["upward_spike"] = (
    work["movement"]
    >= SPIKE_THRESHOLD
)

work["downward_spike"] = (
    work["movement"]
    <= -SPIKE_THRESHOLD
)

work["upward_movement"] = (
    work["movement"]
    >= UPWARD_MOVEMENT_THRESHOLD
) & (
    ~work["upward_spike"]
)

work["downward_movement"] = (
    work["movement"]
    <= DOWNWARD_MOVEMENT_THRESHOLD
) & (
    ~work["downward_spike"]
)

work["extreme_aqi"] = (
    work["actual"]
    >= EXTREME_AQI_THRESHOLD
)

work["high_aqi"] = (
    work["actual"]
    >= HIGH_AQI_THRESHOLD
)


# ============================================================
# TEMPORAL FOLDS
# ============================================================

def build_folds(
    data
):

    minimum_date = (
        data[DATE_COLUMN].min()
    )

    maximum_date = (
        data[DATE_COLUMN].max()
    )

    folds = []

    calibration_start = (
        minimum_date
        + pd.DateOffset(
            months=1
        )
    )

    while True:

        calibration_end = (
            calibration_start
            + pd.DateOffset(
                months=CALIBRATION_MONTHS
            )
        )

        evaluation_end = (
            calibration_end
            + pd.DateOffset(
                months=EVALUATION_MONTHS
            )
        )

        if evaluation_end > (
            maximum_date
            + pd.Timedelta(days=1)
        ):
            break

        calibration_mask = (
            data[DATE_COLUMN]
            >= calibration_start
        ) & (
            data[DATE_COLUMN]
            < calibration_end
        )

        evaluation_mask = (
            data[DATE_COLUMN]
            >= calibration_end
        ) & (
            data[DATE_COLUMN]
            < evaluation_end
        )

        calibration = data[
            calibration_mask
        ].copy()

        evaluation = data[
            evaluation_mask
        ].copy()

        if (
            len(calibration) > 0
            and len(evaluation) > 0
        ):

            folds.append(
                {
                    "calibration_start":
                        calibration_start,
                    "calibration_end":
                        calibration_end,
                    "evaluation_start":
                        calibration_end,
                    "evaluation_end":
                        evaluation_end,
                    "calibration":
                        calibration,
                    "evaluation":
                        evaluation,
                }
            )

        calibration_start = (
            calibration_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


folds = build_folds(
    work
)

print()
print("=" * 70)
print("ROLLING TEMPORAL FOLDS")
print("=" * 70)

print(
    f"Calibration window : "
    f"{CALIBRATION_MONTHS} months"
)

print(
    f"Evaluation window  : "
    f"{EVALUATION_MONTHS} months"
)

print(
    f"Step               : "
    f"{STEP_MONTHS} months"
)

print(
    f"Folds              : "
    f"{len(folds)}"
)

for index, fold in enumerate(
    folds,
    start=1
):

    print()
    print(
        f"Fold {index}:"
    )

    print(
        "  Calibration: "
        f"{fold['calibration_start'].date()} "
        "→ "
        f"{fold['calibration_end'].date()}"
    )

    print(
        "  Evaluation : "
        f"{fold['evaluation_start'].date()} "
        "→ "
        f"{fold['evaluation_end'].date()}"
    )


# ============================================================
# CALIBRATION FUNCTION
# ============================================================

def calculate_group_bias(
    calibration,
    group_columns,
):

    grouped = (
        calibration
        .groupby(
            group_columns,
            dropna=False
        )["residual"]
        .mean()
    )

    return grouped.to_dict()


def get_bias(
    row,
    bias_maps,
):

    city = row[CITY_COLUMN]

    horizon = int(
        row[HORIZON_COLUMN]
    )

    regime = row["regime"]

    city_horizon = (
        city,
        horizon,
    )

    horizon_key = horizon

    regime_key = regime

    city_regime_key = (
        city,
        regime,
    )

    city_horizon_regime_key = (
        city,
        horizon,
        regime,
    )

    # Most specific → least specific
    if city_horizon_regime_key in (
        bias_maps["city_horizon_regime"]
    ):
        return bias_maps[
            "city_horizon_regime"
        ][city_horizon_regime_key]

    if city_horizon in (
        bias_maps["city_horizon"]
    ):
        return bias_maps[
            "city_horizon"
        ][city_horizon]

    if city_regime_key in (
        bias_maps["city_regime"]
    ):
        return bias_maps[
            "city_regime"
        ][city_regime_key]

    if horizon_key in (
        bias_maps["horizon"]
    ):
        return bias_maps[
            "horizon"
        ][horizon_key]

    if regime_key in (
        bias_maps["regime"]
    ):
        return bias_maps[
            "regime"
        ][regime_key]

    return 0.0


def make_bias_maps(
    calibration
):

    maps = {}

    maps[
        "horizon"
    ] = calculate_group_bias(
        calibration,
        [HORIZON_COLUMN],
    )

    maps[
        "regime"
    ] = calculate_group_bias(
        calibration,
        ["regime"],
    )

    maps[
        "city_regime"
    ] = calculate_group_bias(
        calibration,
        [
            CITY_COLUMN,
            "regime",
        ],
    )

    maps[
        "city_horizon"
    ] = calculate_group_bias(
        calibration,
        [
            CITY_COLUMN,
            HORIZON_COLUMN,
        ],
    )

    maps[
        "city_horizon_regime"
    ] = calculate_group_bias(
        calibration,
        [
            CITY_COLUMN,
            HORIZON_COLUMN,
            "regime",
        ],
    )

    return maps


# ============================================================
# ADAPTIVE CORRECTION
# ============================================================

def apply_v7(
    evaluation,
    bias_maps,
    shrinkage,
    max_correction,
    upward_strength,
    downward_strength,
    extreme_strength,
):

    result = evaluation.copy()

    corrections = []

    for _, row in result.iterrows():

        base_bias = get_bias(
            row,
            bias_maps,
        )

        # ----------------------------------------------------
        # BASE REGIME CORRECTION
        # ----------------------------------------------------

        correction = (
            base_bias
            * shrinkage
        )

        # ----------------------------------------------------
        # MOVEMENT-AWARE MODULATION
        # ----------------------------------------------------

        movement = float(
            row["movement"]
        )

        prediction = float(
            row["prediction"]
        )

        actual = float(
            row["actual"]
        )

        # ----------------------------------------------------
        # UPWARD SPIKE
        #
        # V6 was harmful here.
        #
        # We therefore DO NOT simply increase the correction.
        #
        # Instead, reduce the normal correction when an
        # upward spike is detected.
        # ----------------------------------------------------

        if (
            movement
            >= SPIKE_THRESHOLD
        ):

            correction *= max(
                0.0,
                1.0
                - upward_strength,
            )

        # ----------------------------------------------------
        # DOWNWARD SPIKE
        #
        # V6 showed improvement here.
        # Preserve some of that correction.
        # ----------------------------------------------------

        elif (
            movement
            <= -SPIKE_THRESHOLD
        ):

            correction *= (
                1.0
                + downward_strength
            )

        # ----------------------------------------------------
        # MODERATE UPWARD MOVEMENT
        # ----------------------------------------------------

        elif (
            movement
            >= UPWARD_MOVEMENT_THRESHOLD
        ):

            correction *= (
                1.0
                - (
                    upward_strength
                    * 0.50
                )
            )

        # ----------------------------------------------------
        # MODERATE DOWNWARD MOVEMENT
        # ----------------------------------------------------

        elif (
            movement
            <= DOWNWARD_MOVEMENT_THRESHOLD
        ):

            correction *= (
                1.0
                + (
                    downward_strength
                    * 0.50
                )
            )

        # ----------------------------------------------------
        # EXTREME AQI
        #
        # V6 became worse on extreme AQI.
        #
        # Shrink the correction rather than aggressively
        # changing the prediction.
        # ----------------------------------------------------

        if (
            actual
            >= EXTREME_AQI_THRESHOLD
        ):

            correction *= max(
                0.0,
                1.0
                - extreme_strength,
            )

        # ----------------------------------------------------
        # HIGH AQI
        # ----------------------------------------------------

        elif (
            actual
            >= HIGH_AQI_THRESHOLD
        ):

            correction *= (
                1.0
                - (
                    extreme_strength
                    * 0.50
                )
            )

        # ----------------------------------------------------
        # HORIZON-AWARE SAFETY
        #
        # Longer horizons are less certain.
        # Avoid excessive corrections.
        # ----------------------------------------------------

        horizon = int(
            row[HORIZON_COLUMN]
        )

        if horizon == 1:

            horizon_factor = 0.75

        elif horizon == 2:

            horizon_factor = 1.00

        else:

            horizon_factor = 1.10

        correction *= horizon_factor

        # ----------------------------------------------------
        # CAP
        # ----------------------------------------------------

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
        "v7_correction"
    ] = corrections

    result[
        "v7_prediction"
    ] = (
        result["prediction"]
        + result["v7_correction"]
    )

    result[
        "v7_prediction"
    ] = result[
        "v7_prediction"
    ].clip(
        lower=0
    )

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    prediction,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

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
            prediction
            - actual
        )
    )

    within_10 = float(
        np.mean(
            np.abs(
                actual
                - prediction
            )
            <= 10
        )
        * 100
    )

    within_20 = float(
        np.mean(
            np.abs(
                actual
                - prediction
            )
            <= 20
        )
        * 100
    )

    within_30 = float(
        np.mean(
            np.abs(
                actual
                - prediction
            )
            <= 30
        )
        * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


# ============================================================
# BASELINE METRICS
# ============================================================

base_metrics = calculate_metrics(
    work["actual"],
    work["prediction"],
)

print()
print("=" * 70)
print("BASELINE")
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
# STRATEGY EVALUATION
# ============================================================

strategy_rows = []

all_fold_rows = []

parameter_combinations = []

for shrinkage in SHRINKAGES:

    for cap in MAX_CORRECTIONS:

        for upward_strength in (
            UPWARD_SPIKE_STRENGTHS
        ):

            for downward_strength in (
                DOWNWARD_SPIKE_STRENGTHS
            ):

                for extreme_strength in (
                    EXTREME_STRENGTHS
                ):

                    parameter_combinations.append(
                        {
                            "shrinkage":
                                shrinkage,
                            "max_correction":
                                cap,
                            "upward_strength":
                                upward_strength,
                            "downward_strength":
                                downward_strength,
                            "extreme_strength":
                                extreme_strength,
                        }
                    )


print()
print("=" * 70)
print("V7 PARAMETER SEARCH")
print("=" * 70)

print(
    f"Parameter combinations: "
    f"{len(parameter_combinations)}"
)


# ============================================================
# FOLD LOOP
# ============================================================

for fold_index, fold in enumerate(
    folds,
    start=1,
):

    calibration = fold[
        "calibration"
    ]

    evaluation = fold[
        "evaluation"
    ]

    print()
    print(
        "=" * 70
    )

    print(
        f"FOLD {fold_index}/{len(folds)}"
    )

    print(
        f"Calibration rows: "
        f"{len(calibration):,}"
    )

    print(
        f"Evaluation rows : "
        f"{len(evaluation):,}"
    )

    bias_maps = make_bias_maps(
        calibration
    )

    for params in (
        parameter_combinations
    ):

        calibrated = apply_v7(
            evaluation,
            bias_maps,
            params["shrinkage"],
            params["max_correction"],
            params["upward_strength"],
            params["downward_strength"],
            params["extreme_strength"],
        )

        metrics = calculate_metrics(
            calibrated["actual"],
            calibrated["v7_prediction"],
        )

        base_fold_metrics = (
            calculate_metrics(
                calibrated["actual"],
                calibrated["prediction"],
            )
        )

        all_fold_rows.append(
            {
                "fold":
                    fold_index,

                "strategy":
                    "V7",

                **params,

                **metrics,

                "base_mae":
                    base_fold_metrics[
                        "mae"
                    ],

                "base_rmse":
                    base_fold_metrics[
                        "rmse"
                    ],

                "mae_improvement":
                    base_fold_metrics[
                        "mae"
                    ]
                    - metrics[
                        "mae"
                    ],

                "rmse_improvement":
                    base_fold_metrics[
                        "rmse"
                    ]
                    - metrics[
                        "rmse"
                    ],
            }
        )


fold_results = pd.DataFrame(
    all_fold_rows
)


# ============================================================
# AGGREGATE STRATEGIES
# ============================================================

group_columns = [
    "shrinkage",
    "max_correction",
    "upward_strength",
    "downward_strength",
    "extreme_strength",
]

strategy_results = (
    fold_results
    .groupby(
        group_columns,
        as_index=False
    )
    .agg(
        folds=(
            "fold",
            "nunique",
        ),

        mae=(
            "mae",
            "mean",
        ),

        rmse=(
            "rmse",
            "mean",
        ),

        r2=(
            "r2",
            "mean",
        ),

        bias=(
            "bias",
            "mean",
        ),

        base_mae=(
            "base_mae",
            "mean",
        ),

        base_rmse=(
            "base_rmse",
            "mean",
        ),

        mae_improvement=(
            "mae_improvement",
            "mean",
        ),

        rmse_improvement=(
            "rmse_improvement",
            "mean",
        ),
    )
)


# ============================================================
# ROBUST SCORE
# ============================================================

strategy_results[
    "robust_score"
] = (
    strategy_results[
        "mae_improvement"
    ]
    * 0.60
    +
    strategy_results[
        "rmse_improvement"
    ]
    * 0.40
)


strategy_results = (
    strategy_results
    .sort_values(
        "robust_score",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 70)
print("V7 ROBUSTNESS RESULTS")
print("=" * 70)

print()

print(
    strategy_results[
        [
            "shrinkage",
            "max_correction",
            "upward_strength",
            "downward_strength",
            "extreme_strength",
            "folds",
            "mae",
            "rmse",
            "r2",
            "bias",
            "mae_improvement",
            "rmse_improvement",
            "robust_score",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# BEST PARAMETERS
# ============================================================

best = strategy_results.iloc[
    0
]

BEST_SHRINKAGE = float(
    best["shrinkage"]
)

BEST_CAP = float(
    best["max_correction"]
)

BEST_UPWARD = float(
    best["upward_strength"]
)

BEST_DOWNWARD = float(
    best["downward_strength"]
)

BEST_EXTREME = float(
    best["extreme_strength"]
)


print()
print("=" * 70)
print("BEST V7 ROBUST STRATEGY")
print("=" * 70)

print(
    f"Shrinkage          : "
    f"{BEST_SHRINKAGE:.2f}"
)

print(
    f"Correction cap     : "
    f"{BEST_CAP:.2f}"
)

print(
    f"Upward strength    : "
    f"{BEST_UPWARD:.2f}"
)

print(
    f"Downward strength  : "
    f"{BEST_DOWNWARD:.2f}"
)

print(
    f"Extreme strength   : "
    f"{BEST_EXTREME:.2f}"
)

print(
    f"Folds              : "
    f"{int(best['folds'])}"
)

print(
    f"MAE                : "
    f"{best['mae']:.4f}"
)

print(
    f"RMSE               : "
    f"{best['rmse']:.4f}"
)

print(
    f"R²                 : "
    f"{best['r2']:.4f}"
)

print(
    f"Bias               : "
    f"{best['bias']:.4f}"
)

print(
    f"MAE improvement    : "
    f"{best['mae_improvement']:.4f}"
)

print(
    f"RMSE improvement   : "
    f"{best['rmse_improvement']:.4f}"
)


# ============================================================
# APPLY BEST STRATEGY TO ALL DATA
# ============================================================

print()
print("=" * 70)
print("FINAL V7 EVALUATION")
print("=" * 70)

final_predictions = []

for fold_index, fold in enumerate(
    folds,
    start=1,
):

    calibration = fold[
        "calibration"
    ]

    evaluation = fold[
        "evaluation"
    ]

    bias_maps = make_bias_maps(
        calibration
    )

    calibrated = apply_v7(
        evaluation,
        bias_maps,
        BEST_SHRINKAGE,
        BEST_CAP,
        BEST_UPWARD,
        BEST_DOWNWARD,
        BEST_EXTREME,
    )

    calibrated[
        "fold"
    ] = fold_index

    final_predictions.append(
        calibrated
    )


final_predictions = pd.concat(
    final_predictions,
    ignore_index=True,
)


final_metrics = calculate_metrics(
    final_predictions[
        "actual"
    ],
    final_predictions[
        "v7_prediction"
    ],
)

final_base_metrics = calculate_metrics(
    final_predictions[
        "actual"
    ],
    final_predictions[
        "prediction"
    ],
)


print()
print(
    f"Base MAE : "
    f"{final_base_metrics['mae']:.4f}"
)

print(
    f"V7 MAE   : "
    f"{final_metrics['mae']:.4f}"
)

print(
    f"MAE improvement : "
    f"{final_base_metrics['mae'] - final_metrics['mae']:.4f}"
)

print()
print(
    f"Base RMSE : "
    f"{final_base_metrics['rmse']:.4f}"
)

print(
    f"V7 RMSE   : "
    f"{final_metrics['rmse']:.4f}"
)

print(
    f"RMSE improvement : "
    f"{final_base_metrics['rmse'] - final_metrics['rmse']:.4f}"
)

print()
print(
    f"Base R² : "
    f"{final_base_metrics['r2']:.4f}"
)

print(
    f"V7 R²   : "
    f"{final_metrics['r2']:.4f}"
)

print(
    f"Within ±10 : "
    f"{final_metrics['within_10']:.2f}%"
)

print(
    f"Within ±20 : "
    f"{final_metrics['within_20']:.2f}%"
)

print(
    f"Within ±30 : "
    f"{final_metrics['within_30']:.2f}%"
)


# ============================================================
# HORIZON PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("HORIZON PERFORMANCE")
print("=" * 70)

horizon_rows = []

for horizon in sorted(
    final_predictions[
        HORIZON_COLUMN
    ].unique()
):

    subset = final_predictions[
        final_predictions[
            HORIZON_COLUMN
        ]
        == horizon
    ]

    base = calculate_metrics(
        subset["actual"],
        subset["prediction"],
    )

    v7 = calculate_metrics(
        subset["actual"],
        subset["v7_prediction"],
    )

    horizon_rows.append(
        {
            "horizon":
                int(horizon),

            "rows":
                len(subset),

            "base_mae":
                base["mae"],

            "v7_mae":
                v7["mae"],

            "mae_improvement":
                base["mae"]
                - v7["mae"],

            "base_rmse":
                base["rmse"],

            "v7_rmse":
                v7["rmse"],

            "rmse_improvement":
                base["rmse"]
                - v7["rmse"],

            "base_r2":
                base["r2"],

            "v7_r2":
                v7["r2"],
        }
    )


horizon_results = pd.DataFrame(
    horizon_rows
)

print(
    horizon_results.to_string(
        index=False
    )
)


# ============================================================
# CITY PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("CITY PERFORMANCE")
print("=" * 70)

city_rows = []

for city in sorted(
    final_predictions[
        CITY_COLUMN
    ].unique()
):

    subset = final_predictions[
        final_predictions[
            CITY_COLUMN
        ]
        == city
    ]

    base = calculate_metrics(
        subset["actual"],
        subset["prediction"],
    )

    v7 = calculate_metrics(
        subset["actual"],
        subset["v7_prediction"],
    )

    city_rows.append(
        {
            "city_name":
                city,

            "rows":
                len(subset),

            "base_mae":
                base["mae"],

            "v7_mae":
                v7["mae"],

            "mae_improvement":
                base["mae"]
                - v7["mae"],

            "base_rmse":
                base["rmse"],

            "v7_rmse":
                v7["rmse"],

            "rmse_improvement":
                base["rmse"]
                - v7["rmse"],

            "base_bias":
                base["bias"],

            "v7_bias":
                v7["bias"],
        }
    )


city_results = pd.DataFrame(
    city_rows
)

print(
    city_results.to_string(
        index=False
    )
)


# ============================================================
# STRESS TEST
# ============================================================

print()
print("=" * 70)
print("STRESS TEST")
print("=" * 70)


def evaluate_group(
    name,
    mask,
):

    subset = final_predictions[
        mask
    ].copy()

    if len(subset) == 0:
        return {
            "group": name,
            "rows": 0,
        }

    base = calculate_metrics(
        subset["actual"],
        subset["prediction"],
    )

    v7 = calculate_metrics(
        subset["actual"],
        subset["v7_prediction"],
    )

    return {
        "group":
            name,

        "rows":
            len(subset),

        "base_mae":
            base["mae"],

        "v7_mae":
            v7["mae"],

        "mae_improvement":
            base["mae"]
            - v7["mae"],

        "base_rmse":
            base["rmse"],

        "v7_rmse":
            v7["rmse"],

        "rmse_improvement":
            base["rmse"]
            - v7["rmse"],
    }


stress_rows = []

stress_rows.append(
    evaluate_group(
        "Upward Spike",
        final_predictions[
            "upward_spike"
        ],
    )
)

stress_rows.append(
    evaluate_group(
        "Downward Spike",
        final_predictions[
            "downward_spike"
        ],
    )
)

stress_rows.append(
    evaluate_group(
        "Extreme AQI",
        final_predictions[
            "extreme_aqi"
        ],
    )
)

stress_rows.append(
    evaluate_group(
        "High AQI",
        final_predictions[
            "high_aqi"
        ],
    )
)

stress_rows.append(
    evaluate_group(
        "Upward Movement",
        final_predictions[
            "upward_movement"
        ],
    )
)

stress_rows.append(
    evaluate_group(
        "Downward Movement",
        final_predictions[
            "downward_movement"
        ],
    )
)

stress_results = pd.DataFrame(
    stress_rows
)

print(
    stress_results.to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_columns = [
    CITY_COLUMN,
    DATE_COLUMN,
    HORIZON_COLUMN,
    ORIGIN_COLUMN,
    "actual",
    "prediction",
    "v7_prediction",
    "v7_correction",
    "movement",
    "regime",
    "upward_spike",
    "downward_spike",
    "extreme_aqi",
    "high_aqi",
    "fold",
]

prediction_columns = [
    column
    for column in prediction_columns
    if column in final_predictions.columns
]

final_predictions[
    prediction_columns
].to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# SAVE RESULTS
# ============================================================

strategy_results.to_csv(
    STRATEGY_FILE,
    index=False,
)

fold_results.to_csv(
    FOLD_FILE,
    index=False,
)

horizon_results.to_csv(
    HORIZON_FILE,
    index=False,
)

city_results.to_csv(
    CITY_FILE,
    index=False,
)

stress_results.to_csv(
    STRESS_FILE,
    index=False,
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameters = {

    "version":
        "V7",

    "description":
        "Shock-aware adaptive forecast calibration",

    "validation_rows":
        int(len(work)),

    "folds":
        int(len(folds)),

    "calibration_months":
        CALIBRATION_MONTHS,

    "evaluation_months":
        EVALUATION_MONTHS,

    "step_months":
        STEP_MONTHS,

    "spike_threshold":
        SPIKE_THRESHOLD,

    "extreme_aqi_threshold":
        EXTREME_AQI_THRESHOLD,

    "best_shrinkage":
        BEST_SHRINKAGE,

    "best_max_correction":
        BEST_CAP,

    "best_upward_strength":
        BEST_UPWARD,

    "best_downward_strength":
        BEST_DOWNWARD,

    "best_extreme_strength":
        BEST_EXTREME,

    "base_metrics":
        {
            key: float(value)
            for key, value
            in final_base_metrics.items()
        },

    "v7_metrics":
        {
            key: float(value)
            for key, value
            in final_metrics.items()
        },

    "mae_improvement":
        float(
            final_base_metrics["mae"]
            - final_metrics["mae"]
        ),

    "rmse_improvement":
        float(
            final_base_metrics["rmse"]
            - final_metrics["rmse"]
        ),
}


with open(
    PARAMETER_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        parameters,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("V7 COMPLETE")
print("=" * 70)

print(
    f"Best shrinkage     : "
    f"{BEST_SHRINKAGE:.2f}"
)

print(
    f"Correction cap     : "
    f"{BEST_CAP:.2f}"
)

print(
    f"Upward strength    : "
    f"{BEST_UPWARD:.2f}"
)

print(
    f"Downward strength  : "
    f"{BEST_DOWNWARD:.2f}"
)

print(
    f"Extreme strength   : "
    f"{BEST_EXTREME:.2f}"
)

print()

print(
    f"Base MAE           : "
    f"{final_base_metrics['mae']:.4f}"
)

print(
    f"V7 MAE             : "
    f"{final_metrics['mae']:.4f}"
)

print(
    f"MAE improvement    : "
    f"{final_base_metrics['mae'] - final_metrics['mae']:.4f}"
)

print()

print(
    f"Base RMSE          : "
    f"{final_base_metrics['rmse']:.4f}"
)

print(
    f"V7 RMSE            : "
    f"{final_metrics['rmse']:.4f}"
)

print(
    f"RMSE improvement   : "
    f"{final_base_metrics['rmse'] - final_metrics['rmse']:.4f}"
)

print()

print(
    f"Base R²            : "
    f"{final_base_metrics['r2']:.4f}"
)

print(
    f"V7 R²              : "
    f"{final_metrics['r2']:.4f}"
)

print()
print("FILES")
print("=" * 70)

print(
    f"Strategies : {STRATEGY_FILE}"
)

print(
    f"Fold results: {FOLD_FILE}"
)

print(
    f"Predictions : {PREDICTION_FILE}"
)

print(
    f"Horizon     : {HORIZON_FILE}"
)

print(
    f"City        : {CITY_FILE}"
)

print(
    f"Stress      : {STRESS_FILE}"
)

print(
    f"Parameters  : {PARAMETER_FILE}"
)

print()
print(
    "PearlsAQI Forecast Calibration V7 "
    "completed successfully."
)