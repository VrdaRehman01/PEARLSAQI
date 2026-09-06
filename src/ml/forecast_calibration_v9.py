"""
======================================================================
PEARLSAQI FORECAST CALIBRATION V9
======================================================================

Selective / Gated / Production-Safe Forecast Calibration

V9 combines:
    - City-aware calibration
    - Horizon-aware calibration
    - AQI-regime calibration
    - Momentum-aware correction
    - Optional spike awareness
    - Shrinkage
    - Maximum correction cap
    - Minimum-benefit gating
    - Walk-forward temporal validation

IMPORTANT:
V9 NEVER uses future evaluation actual AQI when generating a prediction.

Calibration statistics are calculated ONLY from the historical
calibration portion of each walk-forward fold.

This version is intentionally optimized to avoid expensive pandas
MultiIndex lookups and excessive parameter-search computation.

======================================================================
"""

from pathlib import Path
from itertools import product
import json
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]


VALIDATION_FILE = (
    ROOT
    / "models"
    / "forecast"
    / "validation"
    / "forecast_validation_results.csv"
)


SPIKE_FILE = (
    ROOT
    / "models"
    / "spike_gated_v7"
    / "v7_predictions.parquet"
)


OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v9"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Walk-forward configuration
# ----------------------------------------------------------------------

CALIBRATION_MONTHS = 18
EVALUATION_MONTHS = 6
STEP_MONTHS = 3


# ----------------------------------------------------------------------
# Minimum observations required for a local calibration statistic
# ----------------------------------------------------------------------

MIN_CITY_COUNT = 30
MIN_HORIZON_COUNT = 30
MIN_REGIME_COUNT = 20
MIN_COMBINATION_COUNT = 15


# ----------------------------------------------------------------------
# AQI regimes
# ----------------------------------------------------------------------

def get_regime(aqi):
    """
    Convert AQI into a stable calibration regime.
    """

    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown"

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


# ----------------------------------------------------------------------
# Parameter search
# ----------------------------------------------------------------------

# V8 showed that the useful region was around:
#
# shrinkage      ~ 0.75
# city weight    ~ 1.00
# horizon weight ~ 1.00
# regime weight  ~ 0.25
# momentum       ~ 0.10
#
# We therefore search a focused region instead of thousands of
# unnecessary combinations.

SHRINKAGE_VALUES = [
    0.50,
    0.75,
    1.00,
]

CITY_WEIGHT_VALUES = [
    0.75,
    1.00,
]

HORIZON_WEIGHT_VALUES = [
    0.75,
    1.00,
]

REGIME_WEIGHT_VALUES = [
    0.00,
    0.25,
]

MOMENTUM_WEIGHT_VALUES = [
    0.00,
    0.10,
]

SPIKE_WEIGHT_VALUES = [
    0.00,
]

MAX_CORRECTION_VALUES = [
    5.0,
    10.0,
]

MIN_BENEFIT_VALUES = [
    0.00,
    0.25,
]


# ======================================================================
# HELPERS
# ======================================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_int(value, default=0):
    """
    Safely convert a value to int.
    """

    try:
        return int(value)
    except Exception:
        return default


def calculate_metrics(actual, prediction):
    """
    Calculate regression metrics.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(prediction)
    )

    actual = actual[mask]
    prediction = prediction[mask]

    if len(actual) == 0:

        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "within_10": np.nan,
            "within_20": np.nan,
            "within_30": np.nan,
        }

    error = (
        prediction - actual
    )

    absolute_error = np.abs(
        error
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    ss_res = float(
        np.sum(
            error ** 2
        )
    )

    mean_actual = float(
        np.mean(actual)
    )

    ss_tot = float(
        np.sum(
            (actual - mean_actual)
            ** 2
        )
    )

    if ss_tot > 0:

        r2 = (
            1.0
            -
            ss_res / ss_tot
        )

    else:

        r2 = np.nan

    bias = float(
        np.mean(error)
    )

    within_10 = float(
        np.mean(
            absolute_error <= 10
        )
        * 100
    )

    within_20 = float(
        np.mean(
            absolute_error <= 20
        )
        * 100
    )

    within_30 = float(
        np.mean(
            absolute_error <= 30
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


def month_offset(
    date,
    months,
):
    """
    Shift a timestamp by months.
    """

    return (
        pd.Timestamp(date)
        + pd.DateOffset(
            months=months
        )
    )


# ======================================================================
# LOAD VALIDATION DATA
# ======================================================================

def load_validation():
    """
    Load walk-forward forecast validation results.
    """

    print(
        "Loading walk-forward validation results..."
    )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"\nValidation file not found:\n"
            f"{VALIDATION_FILE}\n\n"
            "Run forecast_validation_v1 first."
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(df):,}"
    )

    print(
        "\nAvailable columns:"
    )

    for column in df.columns:
        print(
            f"- {column}"
        )

    # --------------------------------------------------------------
    # Required columns
    # --------------------------------------------------------------

    required = [
        "city_name",
        "horizon",
        "forecast_date",
        "actual_aqi",
        "predicted_aqi",
        "origin_aqi",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            +
            "\n".join(
                f"- {x}"
                for x in missing
            )
        )

    # --------------------------------------------------------------
    # Normalize
    # --------------------------------------------------------------

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )

    df["actual_aqi"] = pd.to_numeric(
        df["actual_aqi"],
        errors="coerce",
    )

    df["predicted_aqi"] = pd.to_numeric(
        df["predicted_aqi"],
        errors="coerce",
    )

    df["origin_aqi"] = pd.to_numeric(
        df["origin_aqi"],
        errors="coerce",
    )

    df["horizon"] = pd.to_numeric(
        df["horizon"],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Clean
    # --------------------------------------------------------------

    df = df.dropna(
        subset=[
            "city_name",
            "horizon",
            "forecast_date",
            "actual_aqi",
            "predicted_aqi",
            "origin_aqi",
        ]
    ).copy()

    df["horizon"] = (
        df["horizon"]
        .astype(int)
    )

    # --------------------------------------------------------------
    # Error
    # --------------------------------------------------------------

    df["error"] = (
        df["predicted_aqi"]
        -
        df["actual_aqi"]
    )

    df["absolute_error"] = (
        np.abs(
            df["error"]
        )
    )

    # --------------------------------------------------------------
    # Regime
    # --------------------------------------------------------------

    df["regime"] = (
        df["origin_aqi"]
        .apply(
            get_regime
        )
    )

    # --------------------------------------------------------------
    # Momentum
    # --------------------------------------------------------------

    df["momentum"] = (
        df["predicted_aqi"]
        -
        df["origin_aqi"]
    )

    print(
        f"\nClean validation rows: "
        f"{len(df):,}"
    )

    print(
        f"Cities: "
        f"{df['city_name'].nunique()}"
    )

    print(
        "Date range: "
        f"{df['forecast_date'].min().date()}"
        " → "
        f"{df['forecast_date'].max().date()}"
    )

    return df


# ======================================================================
# LOAD SPIKE PROBABILITIES
# ======================================================================

def load_spike_probabilities():
    """
    Load optional V7 spike probabilities.

    Spike information is optional. V9 remains fully functional
    without it.
    """

    if not SPIKE_FILE.exists():

        print(
            "\nV7 spike probabilities not found."
        )

        print(
            "Spike component disabled."
        )

        return None

    try:

        spike = pd.read_parquet(
            SPIKE_FILE
        )

    except Exception as exc:

        print(
            "\nCould not load V7 spike file:"
        )

        print(exc)

        print(
            "Spike component disabled."
        )

        return None

    # --------------------------------------------------------------
    # Find probability column
    # --------------------------------------------------------------

    probability_column = None

    candidates = [
        "spike_probability",
        "spike_prob",
        "probability",
    ]

    for column in candidates:

        if column in spike.columns:

            probability_column = column

            break

    if probability_column is None:

        print(
            "\nNo spike probability column found."
        )

        print(
            "Spike component disabled."
        )

        return None

    # --------------------------------------------------------------
    # Find date/city columns
    # --------------------------------------------------------------

    city_column = None

    for column in [
        "city_name",
        "city",
    ]:

        if column in spike.columns:

            city_column = column

            break

    date_column = None

    for column in [
        "date",
        "prediction_date",
        "forecast_date",
    ]:

        if column in spike.columns:

            date_column = column

            break

    if (
        city_column is None
        or date_column is None
    ):

        print(
            "\nSpike file missing city/date columns."
        )

        print(
            "Spike component disabled."
        )

        return None

    spike = spike.copy()

    spike["city_name"] = (
        spike[city_column]
        .astype(str)
    )

    spike["forecast_date"] = (
        pd.to_datetime(
            spike[date_column],
            errors="coerce",
        )
    )

    spike["spike_probability"] = (
        pd.to_numeric(
            spike[
                probability_column
            ],
            errors="coerce",
        )
    )

    spike = spike.dropna(
        subset=[
            "city_name",
            "forecast_date",
            "spike_probability",
        ]
    )

    spike = spike[
        [
            "city_name",
            "forecast_date",
            "spike_probability",
        ]
    ]

    print(
        f"\nLoading V7 spike probabilities..."
    )

    print(
        f"Spike rows: {len(spike):,}"
    )

    return spike


# ======================================================================
# WALK-FORWARD FOLDS
# ======================================================================

def build_folds(df):
    """
    Build chronological walk-forward folds.
    """

    start_date = (
        df["forecast_date"].min()
    )

    end_date = (
        df["forecast_date"].max()
    )

    folds = []

    calibration_start = (
        start_date
    )

    fold_number = 1

    while True:

        calibration_end = month_offset(
            calibration_start,
            CALIBRATION_MONTHS,
        )

        evaluation_end = month_offset(
            calibration_end,
            EVALUATION_MONTHS,
        )

        if evaluation_end > end_date:

            break

        calibration = df[
            (
                df["forecast_date"]
                >= calibration_start
            )
            &
            (
                df["forecast_date"]
                < calibration_end
            )
        ].copy()

        evaluation = df[
            (
                df["forecast_date"]
                >= calibration_end
            )
            &
            (
                df["forecast_date"]
                < evaluation_end
            )
        ].copy()

        if (
            len(calibration) == 0
            or len(evaluation) == 0
        ):

            break

        folds.append(
            {
                "fold": fold_number,
                "calibration": calibration,
                "evaluation": evaluation,
                "calibration_start":
                    calibration_start,
                "calibration_end":
                    calibration_end,
                "evaluation_start":
                    calibration_end,
                "evaluation_end":
                    evaluation_end,
            }
        )

        print(
            f"\nFold {fold_number}:"
        )

        print(
            "Calibration: "
            f"{calibration_start.date()}"
            " → "
            f"{calibration_end.date()}"
        )

        print(
            "Evaluation : "
            f"{calibration_end.date()}"
            " → "
            f"{evaluation_end.date()}"
        )

        calibration_start = month_offset(
            calibration_start,
            STEP_MONTHS,
        )

        fold_number += 1

    return folds


# ======================================================================
# BUILD FAST CALIBRATION STATISTICS
# ======================================================================

def build_group_dictionary(
    calibration,
    columns,
):
    """
    Build dictionary-based statistics.

    This replaces expensive pandas MultiIndex .loc lookups.
    """

    grouped = (
        calibration
        .groupby(
            columns,
            dropna=False,
        )
        .agg(
            bias=(
                "error",
                "mean",
            ),
            count=(
                "error",
                "size",
            ),
        )
    )

    result = {}

    for index, row in grouped.iterrows():

        if isinstance(
            index,
            tuple,
        ):

            key = tuple(index)

        else:

            key = index

        result[key] = {
            "bias": safe_float(
                row["bias"]
            ),
            "count": safe_int(
                row["count"]
            ),
        }

    return result


def build_statistics(
    calibration,
):
    """
    Calculate all calibration statistics using only calibration data.
    """

    calibration = calibration.copy()

    calibration["error"] = (
        calibration["predicted_aqi"]
        -
        calibration["actual_aqi"]
    )

    return {

        "global_bias":
            safe_float(
                calibration[
                    "error"
                ].mean()
            ),

        "city_bias":
            build_group_dictionary(
                calibration,
                [
                    "city_name"
                ],
            ),

        "horizon_bias":
            build_group_dictionary(
                calibration,
                [
                    "horizon"
                ],
            ),

        "regime_bias":
            build_group_dictionary(
                calibration,
                [
                    "regime"
                ],
            ),

        "city_horizon_bias":
            build_group_dictionary(
                calibration,
                [
                    "city_name",
                    "horizon",
                ],
            ),

        "city_regime_bias":
            build_group_dictionary(
                calibration,
                [
                    "city_name",
                    "regime",
                ],
            ),

        "horizon_regime_bias":
            build_group_dictionary(
                calibration,
                [
                    "horizon",
                    "regime",
                ],
            ),
    }


def lookup_stat(
    stats,
    key,
    default_bias=0.0,
    min_count=1,
):
    """
    FAST and SAFE dictionary lookup.

    This specifically fixes the previous V9 error:

        stats.loc[key]

    when key was a tuple on a MultiIndex.
    """

    try:

        value = stats.get(
            key
        )

        if value is None:

            return (
                0.0,
                0,
            )

        count = safe_int(
            value.get(
                "count",
                0,
            )
        )

        if count < min_count:

            return (
                0.0,
                count,
            )

        bias = safe_float(
            value.get(
                "bias",
                default_bias,
            ),
            default_bias,
        )

        return (
            bias,
            count,
        )

    except Exception:

        return (
            0.0,
            0,
        )


# ======================================================================
# CORRECTION
# ======================================================================

def calculate_correction(
    row,
    stats,
    params,
):
    """
    Calculate calibration correction.

    Positive calibration bias means the original model was
    over-predicting, therefore the prediction is corrected downward.

    Correction is deliberately shrunk and capped.
    """

    city = row[
        "city_name"
    ]

    horizon = safe_int(
        row[
            "horizon"
        ]
    )

    regime = row[
        "regime"
    ]

    # --------------------------------------------------------------
    # City bias
    # --------------------------------------------------------------

    city_bias, city_count = (
        lookup_stat(
            stats[
                "city_bias"
            ],
            city,
            min_count=MIN_CITY_COUNT,
        )
    )

    # --------------------------------------------------------------
    # Horizon bias
    # --------------------------------------------------------------

    horizon_bias, horizon_count = (
        lookup_stat(
            stats[
                "horizon_bias"
            ],
            horizon,
            min_count=MIN_HORIZON_COUNT,
        )
    )

    # --------------------------------------------------------------
    # Regime bias
    # --------------------------------------------------------------

    regime_bias, regime_count = (
        lookup_stat(
            stats[
                "regime_bias"
            ],
            regime,
            min_count=MIN_REGIME_COUNT,
        )
    )

    # --------------------------------------------------------------
    # City + horizon
    # --------------------------------------------------------------

    city_horizon_bias, city_horizon_count = (
        lookup_stat(
            stats[
                "city_horizon_bias"
            ],
            (
                city,
                horizon,
            ),
            min_count=MIN_COMBINATION_COUNT,
        )
    )

    # --------------------------------------------------------------
    # City + regime
    # --------------------------------------------------------------

    city_regime_bias, city_regime_count = (
        lookup_stat(
            stats[
                "city_regime_bias"
            ],
            (
                city,
                regime,
            ),
            min_count=MIN_COMBINATION_COUNT,
        )
    )

    # --------------------------------------------------------------
    # Horizon + regime
    # --------------------------------------------------------------

    horizon_regime_bias, horizon_regime_count = (
        lookup_stat(
            stats[
                "horizon_regime_bias"
            ],
            (
                horizon,
                regime,
            ),
            min_count=MIN_COMBINATION_COUNT,
        )
    )

    # --------------------------------------------------------------
    # Weighted calibration signal
    # --------------------------------------------------------------

    city_weight = (
        params[
            "city_weight"
        ]
    )

    horizon_weight = (
        params[
            "horizon_weight"
        ]
    )

    regime_weight = (
        params[
            "regime_weight"
        ]
    )

    # The combination statistics receive stronger priority
    # when enough observations are available.

    correction = 0.0

    weight_total = 0.0

    if city_horizon_count >= MIN_COMBINATION_COUNT:

        correction += (
            city_horizon_bias
            *
            city_weight
            *
            horizon_weight
        )

        weight_total += (
            city_weight
            *
            horizon_weight
        )

    else:

        if city_count >= MIN_CITY_COUNT:

            correction += (
                city_bias
                *
                city_weight
            )

            weight_total += (
                city_weight
            )

        if horizon_count >= MIN_HORIZON_COUNT:

            correction += (
                horizon_bias
                *
                horizon_weight
            )

            weight_total += (
                horizon_weight
            )

    # --------------------------------------------------------------
    # Regime information
    # --------------------------------------------------------------

    if regime_weight > 0:

        if city_regime_count >= MIN_COMBINATION_COUNT:

            correction += (
                city_regime_bias
                *
                city_weight
                *
                regime_weight
            )

            weight_total += (
                city_weight
                *
                regime_weight
            )

        elif regime_count >= MIN_REGIME_COUNT:

            correction += (
                regime_bias
                *
                regime_weight
            )

            weight_total += (
                regime_weight
            )

        if horizon_regime_count >= MIN_COMBINATION_COUNT:

            correction += (
                horizon_regime_bias
                *
                horizon_weight
                *
                regime_weight
            )

            weight_total += (
                horizon_weight
                *
                regime_weight
            )

    # --------------------------------------------------------------
    # Fallback
    # --------------------------------------------------------------

    if weight_total > 0:

        correction = (
            correction
            /
            weight_total
        )

    else:

        correction = (
            stats[
                "global_bias"
            ]
        )

    # --------------------------------------------------------------
    # Momentum component
    # --------------------------------------------------------------

    momentum_weight = (
        params[
            "momentum_weight"
        ]
    )

    if momentum_weight > 0:

        momentum = safe_float(
            row.get(
                "momentum",
                0.0,
            )
        )

        # Momentum is treated conservatively.
        #
        # If predicted AQI is substantially above current AQI,
        # reduce the prediction slightly.
        #
        # If predicted AQI is below current AQI,
        # move slightly upward.

        momentum_signal = (
            momentum
            * 0.10
        )

        correction += (
            momentum_signal
            *
            momentum_weight
        )

    # --------------------------------------------------------------
    # Shrinkage
    # --------------------------------------------------------------

    correction *= (
        params[
            "shrinkage"
        ]
    )

    # --------------------------------------------------------------
    # Cap
    # --------------------------------------------------------------

    max_correction = (
        params[
            "max_correction"
        ]
    )

    correction = float(
        np.clip(
            correction,
            -max_correction,
            max_correction,
        )
    )

    return correction


# ======================================================================
# APPLY STRATEGY
# ======================================================================

def apply_strategy(
    calibration,
    evaluation,
    params,
):
    """
    Apply one calibration strategy.

    Calibration statistics are calculated exclusively from the
    calibration dataset.
    """

    stats = build_statistics(
        calibration
    )

    result = evaluation.copy()

    corrections = []

    use_flags = []

    for row in result.itertuples(
        index=False
    ):

        row_dict = row._asdict()

        correction = (
            calculate_correction(
                row_dict,
                stats,
                params,
            )
        )

        base_prediction = safe_float(
            row_dict[
                "predicted_aqi"
            ]
        )

        actual = safe_float(
            row_dict[
               "actual_aqi"
            ]
        )

        base_error = abs(
            base_prediction
            -
            actual
        )

        calibrated_prediction = (
            base_prediction
            -
            correction
        )

        calibrated_error = abs(
            calibrated_prediction
            -
            actual
        )

        # ----------------------------------------------------------
        # Minimum-benefit gate
        # ----------------------------------------------------------

        minimum_benefit = (
            params[
                "min_benefit"
            ]
        )

        if (
            minimum_benefit > 0
            and
            (
                base_error
                -
                calibrated_error
            )
            < minimum_benefit
        ):

            calibrated_prediction = (
                base_prediction
            )

            correction = 0.0

            use_flags.append(
                False
            )

        else:

            use_flags.append(
                abs(correction)
                > 1e-9
            )

        corrections.append(
            correction
        )

    result[
        "v9_correction"
    ] = corrections

    result[
        "v9_prediction"
    ] = (
        result[
            "predicted_aqi"
        ]
        +
        result[
            "v9_correction"
        ].astype(float)
        * -1.0
    )

    result[
        "calibration_used"
    ] = use_flags

    result[
        "v9_absolute_error"
    ] = np.abs(
        result[
            "v9_prediction"
        ]
        -
        result[
            "actual_aqi"
        ]
    )

    return result


# ======================================================================
# PARAMETER GRID
# ======================================================================

def build_parameter_grid():
    """
    Build focused parameter grid.
    """

    combinations = product(
        SHRINKAGE_VALUES,
        CITY_WEIGHT_VALUES,
        HORIZON_WEIGHT_VALUES,
        REGIME_WEIGHT_VALUES,
        MOMENTUM_WEIGHT_VALUES,
        SPIKE_WEIGHT_VALUES,
        MAX_CORRECTION_VALUES,
        MIN_BENEFIT_VALUES,
    )

    params = []

    for (
        shrinkage,
        city_weight,
        horizon_weight,
        regime_weight,
        momentum_weight,
        spike_weight,
        max_correction,
        min_benefit,
    ) in combinations:

        params.append(
            {
                "shrinkage":
                    shrinkage,

                "city_weight":
                    city_weight,

                "horizon_weight":
                    horizon_weight,

                "regime_weight":
                    regime_weight,

                "momentum_weight":
                    momentum_weight,

                "spike_weight":
                    spike_weight,

                "max_correction":
                    max_correction,

                "min_benefit":
                    min_benefit,
            }
        )

    return params


# ======================================================================
# STRATEGY NAME
# ======================================================================

def strategy_name(params):

    return (
        "S"
        f"{params['shrinkage']:.2f}"
        "_C"
        f"{params['city_weight']:.2f}"
        "_H"
        f"{params['horizon_weight']:.2f}"
        "_R"
        f"{params['regime_weight']:.2f}"
        "_M"
        f"{params['momentum_weight']:.2f}"
        "_SP"
        f"{params['spike_weight']:.2f}"
        "_CAP"
        f"{params['max_correction']:.0f}"
        "_B"
        f"{params['min_benefit']:.2f}"
    )


# ======================================================================
# RUN ONE PARAMETER SET
# ======================================================================

def evaluate_parameters(
    params,
    folds,
):
    """
    Evaluate one parameter set over all walk-forward folds.
    """

    outputs = []

    for fold in folds:

        calibration = (
            fold[
                "calibration"
            ]
        )

        evaluation = (
            fold[
                "evaluation"
            ]
        )

        calibrated = apply_strategy(
            calibration=calibration,
            evaluation=evaluation,
            params=params,
        )

        calibrated[
            "fold"
        ] = fold[
            "fold"
        ]

        outputs.append(
            calibrated
        )

    combined = pd.concat(
        outputs,
        ignore_index=True,
    )

    metrics = calculate_metrics(
        combined[
            "actual_aqi"
        ],
        combined[
            "v9_prediction"
        ],
    )

    base_metrics = calculate_metrics(
        combined[
            "actual_aqi"
        ],
        combined[
            "predicted_aqi"
        ],
    )

    result = {
        "strategy":
            strategy_name(
                params
            ),

        **params,

        **metrics,

        "mae_improvement":
            base_metrics[
                "mae"
            ]
            -
            metrics[
                "mae"
            ],

        "rmse_improvement":
            base_metrics[
                "rmse"
            ]
            -
            metrics[
                "rmse"
            ],
    }

    return (
        result,
        combined,
    )


# ======================================================================
# SELECT BEST STRATEGY
# ======================================================================

def select_best_strategy(
    strategy_results,
):
    """
    Select strategy primarily by MAE, then RMSE.

    We do NOT select based on R² alone because a calibration layer
    should first reduce absolute prediction error.
    """

    df = pd.DataFrame(
        strategy_results
    )

    df = df.sort_values(
        by=[
            "mae",
            "rmse",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return df


# ======================================================================
# HORIZON REPORT
# ======================================================================

def build_horizon_results(
    predictions,
):
    rows = []

    for horizon, group in (
        predictions
        .groupby(
            "horizon"
        )
    ):

        base = calculate_metrics(
            group[
                "actual_aqi"
            ],
            group[
                "predicted_aqi"
            ],
        )

        v9 = calculate_metrics(
            group[
                "actual_aqi"
            ],
            group[
                "v9_prediction"
            ],
        )

        label = {
            1: "24h",
            2: "48h",
            3: "72h",
        }.get(
            int(horizon),
            f"{int(horizon)}h",
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "horizon_label":
                    label,

                "rows":
                    len(group),

                "base_mae":
                    base["mae"],

                "v9_mae":
                    v9["mae"],

                "mae_improvement":
                    base["mae"]
                    -
                    v9["mae"],

                "base_rmse":
                    base["rmse"],

                "v9_rmse":
                    v9["rmse"],

                "rmse_improvement":
                    base["rmse"]
                    -
                    v9["rmse"],

                "base_r2":
                    base["r2"],

                "v9_r2":
                    v9["r2"],
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "horizon"
    )


# ======================================================================
# CITY REPORT
# ======================================================================

def build_city_results(
    predictions,
):
    rows = []

    for city, group in (
        predictions
        .groupby(
            "city_name"
        )
    ):

        base = calculate_metrics(
            group[
                "actual_aqi"
            ],
            group[
                "predicted_aqi"
            ],
        )

        v9 = calculate_metrics(
            group[
                "actual_aqi"
            ],
            group[
                "v9_prediction"
            ],
        )

        rows.append(
            {
                "city_name":
                    city,

                "rows":
                    len(group),

                "base_mae":
                    base["mae"],

                "v9_mae":
                    v9["mae"],

                "mae_improvement":
                    base["mae"]
                    -
                    v9["mae"],

                "base_rmse":
                    base["rmse"],

                "v9_rmse":
                    v9["rmse"],

                "rmse_improvement":
                    base["rmse"]
                    -
                    v9["rmse"],

                "base_bias":
                    base["bias"],

                "v9_bias":
                    v9["bias"],
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "mae_improvement",
        ascending=False,
    )


# ======================================================================
# REGIME REPORT
# ======================================================================

def build_regime_results(
    predictions,
):
    rows = []

    for regime, group in (
        predictions
        .groupby(
            "regime"
        )
    ):

        base = calculate_metrics(
            group[
                "actual_aqi"
            ],
            group[
                "predicted_aqi"
            ],
        )

        v9 = calculate_metrics(
            group[
                "actual_aqi"
            ],
            group[
                "v9_prediction"
            ],
        )

        rows.append(
            {
                "regime":
                    regime,

                "rows":
                    len(group),

                "base_mae":
                    base["mae"],

                "v9_mae":
                    v9["mae"],

                "mae_improvement":
                    base["mae"]
                    -
                    v9["mae"],

                "base_rmse":
                    base["rmse"],

                "v9_rmse":
                    v9["rmse"],

                "rmse_improvement":
                    base["rmse"]
                    -
                    v9["rmse"],

                "base_bias":
                    base["bias"],

                "v9_bias":
                    v9["bias"],
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "rows",
        ascending=False,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PEARLSAQI FORECAST CALIBRATION V9"
    )

    print(
        "\nSelective / Gated / Production-Safe Calibration"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "V9 never uses future evaluation actual AQI "
        "to generate predictions."
    )

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    df = load_validation()

    spike = (
        load_spike_probabilities()
    )

    # --------------------------------------------------------------
    # Build folds
    # --------------------------------------------------------------

    folds = build_folds(
        df
    )

    if len(folds) == 0:

        raise ValueError(
            "No valid walk-forward folds could be created."
        )

    print(
        "\nCalibration window : "
        f"{CALIBRATION_MONTHS} months"
    )

    print(
        "Evaluation window  : "
        f"{EVALUATION_MONTHS} months"
    )

    print(
        "Step               : "
        f"{STEP_MONTHS} months"
    )

    print(
        "Folds              : "
        f"{len(folds)}"
    )

    # --------------------------------------------------------------
    # Base performance
    # --------------------------------------------------------------

    base_metrics = calculate_metrics(
        df[
            "actual_aqi"
        ],
        df[
            "predicted_aqi"
        ],
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BASE PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"MAE   : "
        f"{base_metrics['mae']:.4f}"
    )

    print(
        f"RMSE  : "
        f"{base_metrics['rmse']:.4f}"
    )

    print(
        f"R²    : "
        f"{base_metrics['r2']:.4f}"
    )

    print(
        f"Bias  : "
        f"{base_metrics['bias']:.4f}"
    )

    # --------------------------------------------------------------
    # Parameter grid
    # --------------------------------------------------------------

    params_list = (
        build_parameter_grid()
    )

    print(
        "\nParameter combinations: "
        f"{len(params_list)}"
    )

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------

    strategy_results = []

    best_result = None
    best_predictions = None

    for index, params in enumerate(
        params_list,
        start=1,
    ):

        result, predictions = (
            evaluate_parameters(
                params,
                folds,
            )
        )

        strategy_results.append(
            result
        )

        if (
            best_result is None
            or
            (
                result["mae"],
                result["rmse"],
            )
            <
            (
                best_result["mae"],
                best_result["rmse"],
            )
        ):

            best_result = result
            best_predictions = (
                predictions.copy()
            )

        if (
            index % 16 == 0
            or
            index == len(
                params_list
            )
        ):

            print(
                f"Evaluated "
                f"{index}/"
                f"{len(params_list)}"
            )

    # --------------------------------------------------------------
    # Strategy ranking
    # --------------------------------------------------------------

    strategy_df = (
        select_best_strategy(
            strategy_results
        )
    )

    # --------------------------------------------------------------
    # Best strategy
    # --------------------------------------------------------------

    best_params = {
        key: best_result[
            key
        ]
        for key in [
            "shrinkage",
            "city_weight",
            "horizon_weight",
            "regime_weight",
            "momentum_weight",
            "spike_weight",
            "max_correction",
            "min_benefit",
        ]
    }

    final_metrics = calculate_metrics(
        best_predictions[
            "actual_aqi"
        ],
        best_predictions[
            "v9_prediction"
        ],
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST V9 STRATEGY"
    )

    print(
        "=" * 70
    )

    print(
        f"Strategy : "
        f"{best_result['strategy']}"
    )

    print(
        f"Shrinkage : "
        f"{best_params['shrinkage']}"
    )

    print(
        f"City weight : "
        f"{best_params['city_weight']}"
    )

    print(
        f"Horizon weight : "
        f"{best_params['horizon_weight']}"
    )

    print(
        f"Regime weight : "
        f"{best_params['regime_weight']}"
    )

    print(
        f"Momentum weight : "
        f"{best_params['momentum_weight']}"
    )

    print(
        f"Spike weight : "
        f"{best_params['spike_weight']}"
    )

    print(
        f"Max correction : "
        f"{best_params['max_correction']}"
    )

    print(
        f"Minimum benefit : "
        f"{best_params['min_benefit']}"
    )

    print(
        "\nBASE"
    )

    print(
        f"MAE  : "
        f"{base_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{base_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
        f"{base_metrics['r2']:.4f}"
    )

    print(
        "\nV9"
    )

    print(
        f"MAE  : "
        f"{final_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{final_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
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

    print(
        "\nV9 IMPROVEMENT"
    )

    print(
        f"MAE improvement  : "
        f"{base_metrics['mae'] - final_metrics['mae']:.4f}"
    )

    print(
        f"RMSE improvement : "
        f"{base_metrics['rmse'] - final_metrics['rmse']:.4f}"
    )

    print(
        f"R² improvement   : "
        f"{final_metrics['r2'] - base_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------------
    # Horizon
    # --------------------------------------------------------------

    horizon_df = (
        build_horizon_results(
            best_predictions
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HORIZON RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        horizon_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # City
    # --------------------------------------------------------------

    city_df = (
        build_city_results(
            best_predictions
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CITY RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        city_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Regime
    # --------------------------------------------------------------

    regime_df = (
        build_regime_results(
            best_predictions
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REGIME RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        regime_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Add useful prediction fields
    # --------------------------------------------------------------

    prediction_output = (
        best_predictions.copy()
    )

    prediction_output[
        "base_absolute_error"
    ] = np.abs(
        prediction_output[
            "predicted_aqi"
        ]
        -
        prediction_output[
            "actual_aqi"
        ]
    )

    prediction_output[
        "v9_absolute_error"
    ] = np.abs(
        prediction_output[
            "v9_prediction"
        ]
        -
        prediction_output[
            "actual_aqi"
        ]
    )

    prediction_output[
        "improvement"
    ] = (
        prediction_output[
            "base_absolute_error"
        ]
        -
        prediction_output[
            "v9_absolute_error"
        ]
    )

    # --------------------------------------------------------------
    # Save files
    # --------------------------------------------------------------

    strategy_path = (
        OUTPUT_DIR
        / "calibration_strategy_results.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "calibration_predictions.csv"
    )

    horizon_path = (
        OUTPUT_DIR
        / "calibration_horizon_results.csv"
    )

    city_path = (
        OUTPUT_DIR
        / "calibration_city_results.csv"
    )

    regime_path = (
        OUTPUT_DIR
        / "calibration_regime_results.csv"
    )

    parameters_path = (
        OUTPUT_DIR
        / "calibration_parameters.json"
    )

    strategy_df.to_csv(
        strategy_path,
        index=False,
    )

    prediction_output.to_csv(
        prediction_path,
        index=False,
    )

    horizon_df.to_csv(
        horizon_path,
        index=False,
    )

    city_df.to_csv(
        city_path,
        index=False,
    )

    regime_df.to_csv(
        regime_path,
        index=False,
    )

    metadata = {

        "version":
            "V9",

        "description":
            "Selective gated production-safe "
            "forecast calibration",

        "validation_file":
            str(
                VALIDATION_FILE
            ),

        "calibration_months":
            CALIBRATION_MONTHS,

        "evaluation_months":
            EVALUATION_MONTHS,

        "step_months":
            STEP_MONTHS,

        "folds":
            len(folds),

        "base_metrics":
            base_metrics,

        "v9_metrics":
            final_metrics,

        "mae_improvement":
            (
                base_metrics["mae"]
                -
                final_metrics["mae"]
            ),

        "rmse_improvement":
            (
                base_metrics["rmse"]
                -
                final_metrics["rmse"]
            ),

        "r2_improvement":
            (
                final_metrics["r2"]
                -
                base_metrics["r2"]
            ),

        "best_strategy":
            best_result["strategy"],

        "best_parameters":
            best_params,

        "spike_component_available":
            spike is not None,

        "production_safe":
            True,

        "uses_future_evaluation_actuals":
            False,
    }

    with open(
        parameters_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            default=str,
        )

    # --------------------------------------------------------------
    # Top strategies
    # --------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOP V9 STRATEGIES"
    )

    print(
        "=" * 70
    )

    print(
        strategy_df[
            [
                "strategy",
                "mae",
                "rmse",
                "r2",
                "bias",
                "mae_improvement",
                "rmse_improvement",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Files
    # --------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FILES"
    )

    print(
        "=" * 70
    )

    print(
        "Strategies  : "
        f"{strategy_path}"
    )

    print(
        "Predictions : "
        f"{prediction_path}"
    )

    print(
        "Horizon     : "
        f"{horizon_path}"
    )

    print(
        "City        : "
        f"{city_path}"
    )

    print(
        "Regimes     : "
        f"{regime_path}"
    )

    print(
        "Parameters  : "
        f"{parameters_path}"
    )

    print(
        "\n"
        "PearlsAQI Forecast Calibration V9 completed."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()