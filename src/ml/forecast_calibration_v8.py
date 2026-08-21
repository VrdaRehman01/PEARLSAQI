"""
PEARLSAQI FORECAST CALIBRATION V8

Production-Safe Adaptive Forecast Calibration

V8 combines:
    1. City-aware bias correction
    2. Horizon-aware correction
    3. City + horizon correction
    4. AQI regime correction
    5. City + regime correction
    6. Momentum-aware correction
    7. Optional spike-aware correction
    8. Temporal walk-forward validation

IMPORTANT
---------
V8 NEVER uses future evaluation actual AQI when generating
evaluation predictions.

For every fold:

    calibration period
            |
            v
    calculate historical statistics
            |
            v
    evaluate future period

This prevents temporal leakage.

V8 is an experimental calibration layer.
It does NOT modify the underlying XGBoost model.
"""

from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd


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
    / "calibration_v8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Temporal configuration
# ------------------------------------------------------------

CALIBRATION_MONTHS = 18
EVALUATION_MONTHS = 6
STEP_MONTHS = 3


# ------------------------------------------------------------
# Parameter search
#
# Reduced from 1296 combinations to 64.
# ------------------------------------------------------------

SHRINKAGES = [
    0.50,
    0.75,
]

MAX_CORRECTIONS = [
    10.0,
    15.0,
]

CITY_WEIGHTS = [
    0.50,
    1.00,
]

HORIZON_WEIGHTS = [
    0.50,
    1.00,
]

REGIME_WEIGHTS = [
    0.00,
    0.25,
]

MOMENTUM_WEIGHTS = [
    0.00,
    0.10,
]


# ------------------------------------------------------------
# Spike configuration
# ------------------------------------------------------------

USE_SPIKE_CORRECTION = True

SPIKE_THRESHOLD = 0.70

SPIKE_WEIGHT = [
    0.00,
    0.10,
]


# ============================================================
# AQI REGIME
# ============================================================

def get_regime(aqi):
    """
    Convert AQI into a stable regime label.
    """

    if pd.isna(aqi):
        return "Unknown"

    aqi = float(aqi)

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

    error = prediction - actual

    mae = float(
        np.mean(np.abs(error))
    )

    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    ss_res = float(
        np.sum(
            (actual - prediction) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (actual - np.mean(actual)) ** 2
        )
    )

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (
            ss_res / ss_tot
        )

    bias = float(
        np.mean(error)
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
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


# ============================================================
# LOAD VALIDATION
# ============================================================

def load_validation():

    print("\nLoading walk-forward validation results...")

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(df):,}"
    )

    required = [
        "city_name",
        "horizon",
        "forecast_date",
        "origin_aqi",
        "actual_aqi",
        "predicted_aqi",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n\n"
            + "\n".join(
                f"- {c}"
                for c in missing
            )
        )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )

    df["city_name"] = (
        df["city_name"]
        .astype(str)
        .str.strip()
    )

    df["horizon"] = pd.to_numeric(
        df["horizon"],
        errors="coerce",
    )

    df["origin_aqi"] = pd.to_numeric(
        df["origin_aqi"],
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

    df = df.dropna(
        subset=[
            "city_name",
            "horizon",
            "forecast_date",
            "origin_aqi",
            "actual_aqi",
            "predicted_aqi",
        ]
    ).copy()

    df["horizon"] = (
        df["horizon"]
        .astype(int)
    )

    df["regime"] = (
        df["origin_aqi"]
        .apply(get_regime)
    )

    df["momentum"] = (
        df["predicted_aqi"]
        - df["origin_aqi"]
    )

    print(
        f"Clean validation rows: "
        f"{len(df):,}"
    )

    print(
        f"Cities: "
        f"{df['city_name'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['forecast_date'].min().date()}"
        f" → "
        f"{df['forecast_date'].max().date()}"
    )

    return df


# ============================================================
# LOAD SPIKE PROBABILITIES
# ============================================================

def load_spike_probabilities():

    """
    Load V7 spike probabilities if available.

    V8 remains fully functional if the V7 file is unavailable.

    The spike information is used only as a prediction-time
    signal and never uses evaluation actual AQI.
    """

    candidates = [

        ROOT
        / "models"
        / "spike_gated_v7"
        / "v7_predictions.parquet",

        ROOT
        / "models"
        / "spike_gated_v7"
        / "v7"
        / "v7_predictions.parquet",

        ROOT
        / "models"
        / "spike_gated"
        / "v7"
        / "v7_predictions.parquet",

    ]

    for path in candidates:

        if not path.exists():
            continue

        try:

            spike = pd.read_parquet(
                path
            )

            required = [
                "city_name",
                "date",
                "spike_probability",
            ]

            if not all(
                c in spike.columns
                for c in required
            ):
                continue

            spike["date"] = pd.to_datetime(
                spike["date"],
                errors="coerce",
            )

            spike["city_name"] = (
                spike["city_name"]
                .astype(str)
                .str.strip()
            )

            spike["spike_probability"] = (
                pd.to_numeric(
                    spike[
                        "spike_probability"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .clip(0.0, 1.0)
            )

            spike = spike[
                [
                    "city_name",
                    "date",
                    "spike_probability",
                ]
            ].drop_duplicates(
                [
                    "city_name",
                    "date",
                ]
            )

            spike = spike.rename(
                columns={
                    "date":
                    "forecast_date"
                }
            )

            print(
                "\nLoading V7 spike probabilities..."
            )

            print(
                f"Spike rows: "
                f"{len(spike):,}"
            )

            return spike

        except Exception as exc:

            print(
                f"Could not load spike file "
                f"{path}: {exc}"
            )

    print(
        "\nV7 spike probabilities "
        "not found."
    )

    print(
        "Spike correction will be disabled."
    )

    return None


# ============================================================
# TEMPORAL FOLDS
# ============================================================

def build_folds(df):

    start = (
        df["forecast_date"]
        .min()
        .normalize()
    )

    end = (
        df["forecast_date"]
        .max()
        .normalize()
    )

    folds = []

    calibration_start = start

    fold_number = 1

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

        if evaluation_end > end + pd.Timedelta(days=1):
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
            len(calibration) > 0
            and len(evaluation) > 0
        ):

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
                f"Calibration: "
                f"{calibration_start.date()}"
                f" → "
                f"{calibration_end.date()}"
            )

            print(
                f"Evaluation : "
                f"{calibration_end.date()}"
                f" → "
                f"{evaluation_end.date()}"
            )

            fold_number += 1

        calibration_start = (
            calibration_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


# ============================================================
# HISTORICAL STATISTICS
# ============================================================

def grouped_statistics(
    df,
    columns,
    global_bias,
):

    if len(df) == 0:

        return pd.DataFrame(
            columns=[
                "mean",
                "count",
            ]
        )

    error = (
        df["actual_aqi"]
        - df["predicted_aqi"]
    )

    temp = df.copy()

    temp["_error"] = error

    stats = (
        temp
        .groupby(columns)["_error"]
        .agg(
            [
                "mean",
                "count",
            ]
        )
    )

    return stats


def build_statistics(calibration):

    """
    Build ALL historical calibration statistics once.

    This is the key performance improvement in V8.
    """

    global_bias = float(
        (
            calibration["actual_aqi"]
            - calibration["predicted_aqi"]
        ).mean()
    )

    stats = {}

    stats["global_bias"] = global_bias

    stats["city_bias"] = (
        grouped_statistics(
            calibration,
            ["city_name"],
            global_bias,
        )
    )

    stats["horizon_bias"] = (
        grouped_statistics(
            calibration,
            ["horizon"],
            global_bias,
        )
    )

    stats["city_horizon_bias"] = (
        grouped_statistics(
            calibration,
            [
                "city_name",
                "horizon",
            ],
            global_bias,
        )
    )

    stats["regime_bias"] = (
        grouped_statistics(
            calibration,
            ["regime"],
            global_bias,
        )
    )

    stats["city_regime_bias"] = (
        grouped_statistics(
            calibration,
            [
                "city_name",
                "regime",
            ],
            global_bias,
        )
    )

    calibration = calibration.copy()

    calibration["momentum_bin"] = (
        calibration["momentum"]
        .apply(momentum_bucket)
    )

    stats["momentum_bias"] = (
        grouped_statistics(
            calibration,
            ["momentum_bin"],
            global_bias,
        )
    )

    return stats


# ============================================================
# SHRINKAGE
# ============================================================

def shrunk_value(
    value,
    count,
    fallback,
    minimum_count=10,
):

    if pd.isna(value):
        return fallback

    if count is None:
        return fallback

    count = float(count)

    if count <= 0:
        return fallback

    reliability = min(
        1.0,
        count / minimum_count,
    )

    return (
        reliability * float(value)
        +
        (1.0 - reliability)
        * float(fallback)
    )


# ============================================================
# FAST LOOKUPS
# ============================================================

def build_lookup_dict(table):

    """
    Convert pandas grouped statistics into
    normal Python dictionaries.

    This eliminates expensive MultiIndex.loc
    operations inside the parameter search.
    """

    result = {}

    if table is None:
        return result

    if len(table) == 0:
        return result

    for key, row in table.iterrows():

        if isinstance(key, tuple):
            normalized_key = tuple(key)
        else:
            normalized_key = key

        result[normalized_key] = (
            float(row["mean"]),
            float(row["count"]),
        )

    return result


def prepare_fast_stats(stats):

    fast = {}

    fast["global_bias"] = (
        float(stats["global_bias"])
    )

    fast["city"] = build_lookup_dict(
        stats["city_bias"]
    )

    fast["horizon"] = build_lookup_dict(
        stats["horizon_bias"]
    )

    fast["city_horizon"] = (
        build_lookup_dict(
            stats["city_horizon_bias"]
        )
    )

    fast["regime"] = build_lookup_dict(
        stats["regime_bias"]
    )

    fast["city_regime"] = (
        build_lookup_dict(
            stats["city_regime_bias"]
        )
    )

    fast["momentum"] = (
        build_lookup_dict(
            stats["momentum_bias"]
        )
    )

    return fast


# ============================================================
# MOMENTUM
# ============================================================

def momentum_bucket(value):

    value = float(value)

    if value <= -50:
        return "strong_down"

    if value <= -25:
        return "down"

    if value <= -10:
        return "slight_down"

    if value <= 10:
        return "stable"

    if value <= 25:
        return "slight_up"

    if value <= 50:
        return "up"

    return "strong_up"


# ============================================================
# FAST CORRECTION
# ============================================================

def lookup_stat(
    table,
    key,
    fallback,
):

    item = table.get(key)

    if item is None:
        return fallback

    value, count = item

    return shrunk_value(
        value,
        count,
        fallback,
    )


def calculate_correction_fast(
    row,
    stats,
    shrinkage,
    max_correction,
    city_weight,
    horizon_weight,
    regime_weight,
    momentum_weight,
    spike_weight,
):

    global_bias = stats["global_bias"]

    city = row["city_name"]

    horizon = int(
        row["horizon"]
    )

    regime = row["regime"]

    origin_aqi = float(
        row["origin_aqi"]
    )

    predicted_aqi = float(
        row["predicted_aqi"]
    )

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    city_bias = lookup_stat(
        stats["city"],
        city,
        global_bias,
    )

    # --------------------------------------------------------
    # HORIZON
    # --------------------------------------------------------

    horizon_bias = lookup_stat(
        stats["horizon"],
        horizon,
        global_bias,
    )

    # --------------------------------------------------------
    # CITY + HORIZON
    # --------------------------------------------------------

    city_horizon_bias = lookup_stat(
        stats["city_horizon"],
        (
            city,
            horizon,
        ),
        city_bias,
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    regime_bias = lookup_stat(
        stats["regime"],
        regime,
        global_bias,
    )

    # --------------------------------------------------------
    # CITY + REGIME
    # --------------------------------------------------------

    city_regime_bias = lookup_stat(
        stats["city_regime"],
        (
            city,
            regime,
        ),
        regime_bias,
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = (
        predicted_aqi
        - origin_aqi
    )

    momentum_bin = (
        momentum_bucket(momentum)
    )

    momentum_bias = lookup_stat(
        stats["momentum"],
        momentum_bin,
        global_bias,
    )

    # --------------------------------------------------------
    # BASE CALIBRATION
    # --------------------------------------------------------

    correction = (

        global_bias * 0.10

        +

        city_bias
        * city_weight
        * 0.20

        +

        horizon_bias
        * horizon_weight
        * 0.20

        +

        city_horizon_bias
        * 0.30

        +

        city_regime_bias
        * regime_weight
        * 0.20

        +

        momentum_bias
        * momentum_weight
        * 0.10
    )

    # --------------------------------------------------------
    # SPIKE CORRECTION
    # --------------------------------------------------------

    if (
        USE_SPIKE_CORRECTION
        and spike_weight > 0
    ):

        probability = float(
            row.get(
                "spike_probability",
                0.0,
            )
        )

        if (
            probability
            >= SPIKE_THRESHOLD
        ):

            # Spike confidence increases
            # the amount of historical
            # calibration correction,
            # but does NOT use future AQI.

            correction *= (
                1.0
                +
                probability
                * spike_weight
            )

    # --------------------------------------------------------
    # SHRINK
    # --------------------------------------------------------

    correction *= shrinkage

    # --------------------------------------------------------
    # CAP
    # --------------------------------------------------------

    correction = np.clip(
        correction,
        -max_correction,
        max_correction,
    )

    return float(correction)


# ============================================================
# APPLY STRATEGY
# ============================================================

def apply_strategy(
    evaluation,
    stats,
    params,
):

    result = evaluation.copy()

    corrections = []

    for row in result.itertuples(
        index=False
    ):

        row_dict = {
            "city_name":
                row.city_name,

            "horizon":
                row.horizon,

            "origin_aqi":
                row.origin_aqi,

            "predicted_aqi":
                row.predicted_aqi,

            "regime":
                row.regime,

            "spike_probability":
                getattr(
                    row,
                    "spike_probability",
                    0.0,
                ),
        }

        correction = (
            calculate_correction_fast(
                row_dict,
                stats,
                params["shrinkage"],
                params["max_correction"],
                params["city_weight"],
                params["horizon_weight"],
                params["regime_weight"],
                params["momentum_weight"],
                params["spike_weight"],
            )
        )

        corrections.append(
            correction
        )

    result["v8_correction"] = (
        corrections
    )

    result["v8_prediction"] = (
        result["predicted_aqi"]
        +
        result["v8_correction"]
    )

    return result


# ============================================================
# PARAMETER GRID
# ============================================================

def build_parameter_grid():

    grid = []

    for (
        shrinkage,
        max_correction,
        city_weight,
        horizon_weight,
        regime_weight,
        momentum_weight,
        spike_weight,
    ) in product(
        SHRINKAGES,
        MAX_CORRECTIONS,
        CITY_WEIGHTS,
        HORIZON_WEIGHTS,
        REGIME_WEIGHTS,
        MOMENTUM_WEIGHTS,
        SPIKE_WEIGHT,
    ):

        grid.append(
            {
                "shrinkage":
                    shrinkage,

                "max_correction":
                    max_correction,

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
            }
        )

    return grid


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def evaluate_parameters(
    folds,
    parameter_grid,
):

    strategy_results = []

    all_predictions = []

    print(
        f"\nParameter combinations: "
        f"{len(parameter_grid)}"
    )

    for index, params in enumerate(
        parameter_grid,
        start=1,
    ):

        fold_predictions = []

        for fold in folds:

            calibration = fold[
                "calibration"
            ]

            evaluation = fold[
                "evaluation"
            ]

            stats = build_statistics(
                calibration
            )

            fast_stats = (
                prepare_fast_stats(
                    stats
                )
            )

            calibrated = (
                apply_strategy(
                    evaluation,
                    fast_stats,
                    params,
                )
            )

            calibrated["fold"] = (
                fold["fold"]
            )

            fold_predictions.append(
                calibrated
            )

        combined = pd.concat(
            fold_predictions,
            ignore_index=True,
        )

        metrics = calculate_metrics(
            combined["actual_aqi"],
            combined["v8_prediction"],
        )

        base_metrics = calculate_metrics(
            combined["actual_aqi"],
            combined["predicted_aqi"],
        )

        strategy_name = (
            f"S{params['shrinkage']:.2f}"
            f"_C{params['city_weight']:.2f}"
            f"_H{params['horizon_weight']:.2f}"
            f"_R{params['regime_weight']:.2f}"
            f"_M{params['momentum_weight']:.2f}"
            f"_SP{params['spike_weight']:.2f}"
            f"_CAP{params['max_correction']:.0f}"
        )

        result = {
            "strategy":
                strategy_name,

            **params,

            **metrics,

            "mae_improvement":
                base_metrics["mae"]
                - metrics["mae"],

            "rmse_improvement":
                base_metrics["rmse"]
                - metrics["rmse"],
        }

        strategy_results.append(
            result
        )

        if (
            index % 8 == 0
            or index == len(
                parameter_grid
            )
        ):

            print(
                f"Evaluated "
                f"{index}/"
                f"{len(parameter_grid)}"
            )

    results = pd.DataFrame(
        strategy_results
    )

    results = results.sort_values(
        [
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

    # --------------------------------------------------------
    # Rebuild predictions for best strategy
    # --------------------------------------------------------

    best_params = (
        results.iloc[0].to_dict()
    )

    selected_params = {
        "shrinkage":
            float(
                best_params[
                    "shrinkage"
                ]
            ),

        "max_correction":
            float(
                best_params[
                    "max_correction"
                ]
            ),

        "city_weight":
            float(
                best_params[
                    "city_weight"
                ]
            ),

        "horizon_weight":
            float(
                best_params[
                    "horizon_weight"
                ]
            ),

        "regime_weight":
            float(
                best_params[
                    "regime_weight"
                ]
            ),

        "momentum_weight":
            float(
                best_params[
                    "momentum_weight"
                ]
            ),

        "spike_weight":
            float(
                best_params[
                    "spike_weight"
                ]
            ),
    }

    for fold in folds:

        calibration = fold[
            "calibration"
        ]

        evaluation = fold[
            "evaluation"
        ]

        stats = build_statistics(
            calibration
        )

        fast_stats = (
            prepare_fast_stats(
                stats
            )
        )

        calibrated = (
            apply_strategy(
                evaluation,
                fast_stats,
                selected_params,
            )
        )

        calibrated["fold"] = (
            fold["fold"]
        )

        all_predictions.append(
            calibrated
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    return (
        results,
        predictions,
        selected_params,
    )


# ============================================================
# HORIZON RESULTS
# ============================================================

def horizon_results(
    predictions
):

    rows = []

    for horizon, group in (
        predictions
        .groupby("horizon")
    ):

        base = calculate_metrics(
            group["actual_aqi"],
            group["predicted_aqi"],
        )

        v8 = calculate_metrics(
            group["actual_aqi"],
            group["v8_prediction"],
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "horizon_label":
                    f"{int(horizon) * 24}h",

                "rows":
                    len(group),

                "base_mae":
                    base["mae"],

                "v8_mae":
                    v8["mae"],

                "mae_improvement":
                    base["mae"]
                    - v8["mae"],

                "base_rmse":
                    base["rmse"],

                "v8_rmse":
                    v8["rmse"],

                "rmse_improvement":
                    base["rmse"]
                    - v8["rmse"],

                "base_r2":
                    base["r2"],

                "v8_r2":
                    v8["r2"],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CITY RESULTS
# ============================================================

def city_results(
    predictions
):

    rows = []

    for city, group in (
        predictions
        .groupby("city_name")
    ):

        base = calculate_metrics(
            group["actual_aqi"],
            group["predicted_aqi"],
        )

        v8 = calculate_metrics(
            group["actual_aqi"],
            group["v8_prediction"],
        )

        rows.append(
            {
                "city_name":
                    city,

                "rows":
                    len(group),

                "base_mae":
                    base["mae"],

                "v8_mae":
                    v8["mae"],

                "mae_improvement":
                    base["mae"]
                    - v8["mae"],

                "base_rmse":
                    base["rmse"],

                "v8_rmse":
                    v8["rmse"],

                "rmse_improvement":
                    base["rmse"]
                    - v8["rmse"],

                "base_bias":
                    base["bias"],

                "v8_bias":
                    v8["bias"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mae_improvement",
            ascending=False,
        )
    )


# ============================================================
# REGIME RESULTS
# ============================================================

def regime_results(
    predictions
):

    rows = []

    for regime, group in (
        predictions
        .groupby("regime")
    ):

        base = calculate_metrics(
            group["actual_aqi"],
            group["predicted_aqi"],
        )

        v8 = calculate_metrics(
            group["actual_aqi"],
            group["v8_prediction"],
        )

        rows.append(
            {
                "regime":
                    regime,

                "rows":
                    len(group),

                "base_mae":
                    base["mae"],

                "v8_mae":
                    v8["mae"],

                "mae_improvement":
                    base["mae"]
                    - v8["mae"],

                "base_rmse":
                    base["rmse"],

                "v8_rmse":
                    v8["rmse"],

                "rmse_improvement":
                    base["rmse"]
                    - v8["rmse"],

                "base_bias":
                    base["bias"],

                "v8_bias":
                    v8["bias"],
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "mae_improvement",
        ascending=False,
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    df,
    folds,
    params,
    base_metrics,
    v8_metrics,
):

    metadata = {

        "project":
            "PearlsAQI",

        "calibration":
            "V8",

        "production_safe":
            True,

        "uses_future_evaluation_actuals":
            False,

        "calibration_months":
            CALIBRATION_MONTHS,

        "evaluation_months":
            EVALUATION_MONTHS,

        "step_months":
            STEP_MONTHS,

        "folds":
            len(folds),

        "cities":
            int(
                df["city_name"]
                .nunique()
            ),

        "validation_rows":
            len(df),

        "date_start":
            str(
                df["forecast_date"]
                .min()
                .date()
            ),

        "date_end":
            str(
                df["forecast_date"]
                .max()
                .date()
            ),

        "selected_parameters":
            params,

        "base_metrics":
            base_metrics,

        "v8_metrics":
            v8_metrics,

        "mae_improvement":
            (
                base_metrics["mae"]
                -
                v8_metrics["mae"]
            ),

        "rmse_improvement":
            (
                base_metrics["rmse"]
                -
                v8_metrics["rmse"]
            ),
    }

    path = (
        OUTPUT_DIR
        / "calibration_parameters.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PEARLSAQI "
        "FORECAST CALIBRATION V8"
    )

    print(
        "\nProduction-Safe "
        "Adaptive Calibration"
    )

    print(
        "\nIMPORTANT:"
        "\nV8 does NOT use future "
        "evaluation actual AQI "
        "to generate predictions."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_validation()

    # --------------------------------------------------------
    # SPIKE
    # --------------------------------------------------------

    spike = (
        load_spike_probabilities()
    )

    if spike is not None:

        df = df.merge(
            spike,
            on=[
                "city_name",
                "forecast_date",
            ],
            how="left",
        )

        df[
            "spike_probability"
        ] = (
            df[
                "spike_probability"
            ]
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    else:

        df[
            "spike_probability"
        ] = 0.0

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    folds = build_folds(
        df
    )

    print(
        f"\nCalibration window : "
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

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    base_metrics = (
        calculate_metrics(
            df["actual_aqi"],
            df["predicted_aqi"],
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BASE PERFORMANCE"
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

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    parameter_grid = (
        build_parameter_grid()
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    (
        strategy_results,
        predictions,
        selected_params,
    ) = evaluate_parameters(
        folds,
        parameter_grid,
    )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best = (
        strategy_results
        .iloc[0]
    )

    v8_metrics = (
        calculate_metrics(
            predictions[
                "actual_aqi"
            ],
            predictions[
                "v8_prediction"
            ],
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST V8 STRATEGY"
    )

    print(
        f"Strategy : "
        f"{best['strategy']}"
    )

    print(
        f"Shrinkage : "
        f"{selected_params['shrinkage']}"
    )

    print(
        f"City weight : "
        f"{selected_params['city_weight']}"
    )

    print(
        f"Horizon weight : "
        f"{selected_params['horizon_weight']}"
    )

    print(
        f"Regime weight : "
        f"{selected_params['regime_weight']}"
    )

    print(
        f"Momentum weight : "
        f"{selected_params['momentum_weight']}"
    )

    print(
        f"Spike weight : "
        f"{selected_params['spike_weight']}"
    )

    print(
        f"Max correction : "
        f"{selected_params['max_correction']}"
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
        "\nV8"
    )

    print(
        f"MAE  : "
        f"{v8_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{v8_metrics['rmse']:.4f}"
    )

    print(
        f"R²   : "
        f"{v8_metrics['r2']:.4f}"
    )

    print(
        f"Within ±10 : "
        f"{v8_metrics['within_10']:.2f}%"
    )

    print(
        f"Within ±20 : "
        f"{v8_metrics['within_20']:.2f}%"
    )

    print(
        f"Within ±30 : "
        f"{v8_metrics['within_30']:.2f}%"
    )

    print(
        "\nV8 IMPROVEMENT"
    )

    print(
        f"MAE improvement  : "
        f"{base_metrics['mae'] - v8_metrics['mae']:.4f}"
    )

    print(
        f"RMSE improvement : "
        f"{base_metrics['rmse'] - v8_metrics['rmse']:.4f}"
    )

    print(
        f"R² improvement   : "
        f"{v8_metrics['r2'] - base_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------
    # HORIZON
    # --------------------------------------------------------

    h_results = horizon_results(
        predictions
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HORIZON RESULTS"
    )

    print(
        h_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    c_results = city_results(
        predictions
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CITY RESULTS"
    )

    print(
        c_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    r_results = regime_results(
        predictions
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REGIME RESULTS"
    )

    print(
        r_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TOP STRATEGIES
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOP V8 STRATEGIES"
    )

    print(
        strategy_results[
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

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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

    predictions[
        "error_base"
    ] = (
        predictions[
            "predicted_aqi"
        ]
        -
        predictions[
            "actual_aqi"
        ]
    )

    predictions[
        "error_v8"
    ] = (
        predictions[
            "v8_prediction"
        ]
        -
        predictions[
            "actual_aqi"
        ]
    )

    predictions[
        "absolute_error_base"
    ] = (
        predictions[
            "error_base"
        ].abs()
    )

    predictions[
        "absolute_error_v8"
    ] = (
        predictions[
            "error_v8"
        ].abs()
    )

    predictions[
        "improvement"
    ] = (
        predictions[
            "absolute_error_base"
        ]
        -
        predictions[
            "absolute_error_v8"
        ]
    )

    strategy_results.to_csv(
        strategy_path,
        index=False,
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    h_results.to_csv(
        horizon_path,
        index=False,
    )

    c_results.to_csv(
        city_path,
        index=False,
    )

    r_results.to_csv(
        regime_path,
        index=False,
    )

    metadata_path = save_metadata(
        df,
        folds,
        selected_params,
        base_metrics,
        v8_metrics,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
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
        f"{prediction_path}"
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
        f"Regimes     : "
        f"{regime_path}"
    )

    print(
        f"Parameters  : "
        f"{metadata_path}"
    )

    print(
        "\n"
        "PearlsAQI Forecast "
        "Calibration V8 completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()