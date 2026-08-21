"""
PEARLSAQI FORECAST CALIBRATION V2

Purpose:
    Learn safe, walk-forward calibration corrections for
    24h / 48h / 72h AQI forecasts.

Important:
    This script does NOT retrain the production AQI model.

    It uses the historical walk-forward validation results from:

    models/forecast/validation/forecast_validation_results.csv

Strategies tested:

    1. BASE
    2. HORIZON_BIAS
    3. CITY_BIAS
    4. CITY_HORIZON_BIAS
    5. CITY_HORIZON_SHRUNK
    6. CITY_HORIZON_MOMENTUM
    7. CITY_HORIZON_MOMENTUM_SHRUNK

The calibration is learned from historical validation data
and evaluated using out-of-fold style temporal splits to reduce
the risk of leaking future information.
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

OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "calibration_strategy_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

PARAMETERS_FILE = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)

CITY_RESULTS_FILE = (
    OUTPUT_DIR
    / "calibration_city_results.csv"
)


# ============================================================
# CONFIG
# ============================================================

HORIZONS = [1, 2, 3]

# Fraction of historical validation observations used
# to learn calibration parameters.
#
# The remaining observations are used for evaluation.
#
# This is deliberately temporal rather than random.
TRAIN_FRACTION = 0.60

# Shrinkage prevents a city with relatively few observations
# from receiving an overly aggressive correction.
SHRINKAGE_MIN_ROWS = 80

# Maximum absolute correction.
#
# We do not want calibration to completely override the
# underlying ML forecast.
MAX_CORRECTION = 15.0

# Momentum correction strength candidates.
MOMENTUM_STRENGTHS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

# Shrinkage strengths.
SHRINKAGE_STRENGTHS = [
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# LOAD
# ============================================================

def load_validation_data():

    print()
    print("=" * 70)
    print("PEARLSAQI FORECAST CALIBRATION V2")
    print("=" * 70)

    print()
    print(
        "Loading walk-forward validation results..."
    )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}\n\n"
            f"Run:\n"
            f"python -m src.ml.forecast_validation_v1"
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    required = {
        "city_id",
        "city_name",
        "origin_date",
        "forecast_date",
        "horizon",
        "origin_aqi",
        "actual_aqi",
        "predicted_aqi",
        "error",
        "absolute_error",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Validation dataset is missing "
            f"required columns: {sorted(missing)}"
        )

    df["origin_date"] = pd.to_datetime(
        df["origin_date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    df["horizon"] = (
        df["horizon"]
        .astype(int)
    )

    df["city_name"] = (
        df["city_name"]
        .astype(str)
    )

    df["origin_aqi"] = (
        df["origin_aqi"]
        .astype(float)
    )

    df["actual_aqi"] = (
        df["actual_aqi"]
        .astype(float)
    )

    df["predicted_aqi"] = (
        df["predicted_aqi"]
        .astype(float)
    )

    df["error"] = (
        df["predicted_aqi"]
        - df["actual_aqi"]
    )

    df["absolute_error"] = (
        df["error"]
        .abs()
    )

    df = df.sort_values(
        [
            "city_name",
            "origin_date",
            "horizon",
        ]
    ).reset_index(drop=True)

    print(
        f"Validation rows: {len(df):,}"
    )

    print(
        f"Cities: {df['city_name'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['origin_date'].min().date()} "
        f"→ "
        f"{df['origin_date'].max().date()}"
    )

    return df


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

    error = (
        prediction
        - actual
    )

    absolute_error = np.abs(
        error
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(
        mse
    )

    actual_mean = np.mean(
        actual
    )

    ss_res = np.sum(
        (actual - prediction) ** 2
    )

    ss_tot = np.sum(
        (actual - actual_mean) ** 2
    )

    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = (
            1
            - ss_res / ss_tot
        )

    return {

        "rows":
            len(actual),

        "mae":
            float(
                np.mean(
                    absolute_error
                )
            ),

        "rmse":
            float(
                rmse
            ),

        "r2":
            float(
                r2
            ),

        "bias":
            float(
                np.mean(error)
            ),

        "within_10":
            float(
                np.mean(
                    absolute_error <= 10
                )
                * 100
            ),

        "within_20":
            float(
                np.mean(
                    absolute_error <= 20
                )
                * 100
            ),

        "within_30":
            float(
                np.mean(
                    absolute_error <= 30
                )
                * 100
            ),
    }


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def split_train_evaluation(df):

    """
    Temporal split.

    Calibration parameters are learned from earlier
    forecast origins and evaluated on later origins.
    """

    unique_dates = (
        df["origin_date"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    if len(unique_dates) < 10:

        raise ValueError(
            "Not enough unique forecast origins "
            "for temporal calibration."
        )

    split_index = int(
        len(unique_dates)
        * TRAIN_FRACTION
    )

    split_index = max(
        1,
        min(
            split_index,
            len(unique_dates) - 1,
        ),
    )

    split_date = pd.Timestamp(
        unique_dates[
            split_index
        ]
    )

    train = df[
        df["origin_date"]
        < split_date
    ].copy()

    evaluation = df[
        df["origin_date"]
        >= split_date
    ].copy()

    print()
    print(
        "TEMPORAL CALIBRATION SPLIT"
    )

    print(
        f"Calibration start: "
        f"{train['origin_date'].min().date()}"
    )

    print(
        f"Calibration end  : "
        f"{train['origin_date'].max().date()}"
    )

    print(
        f"Evaluation start : "
        f"{evaluation['origin_date'].min().date()}"
    )

    print(
        f"Evaluation end   : "
        f"{evaluation['origin_date'].max().date()}"
    )

    print(
        f"Calibration rows  : "
        f"{len(train):,}"
    )

    print(
        f"Evaluation rows   : "
        f"{len(evaluation):,}"
    )

    return train, evaluation


# ============================================================
# CALIBRATION PARAMETERS
# ============================================================

def calculate_global_horizon_bias(
    train,
):

    result = {}

    for horizon in HORIZONS:

        subset = train[
            train["horizon"]
            == horizon
        ]

        if subset.empty:

            result[horizon] = 0.0

        else:

            # prediction - actual
            #
            # To correct the prediction:
            #
            # calibrated =
            # prediction - bias
            #
            result[horizon] = float(
                subset["error"].mean()
            )

    return result


def calculate_city_bias(
    train,
):

    result = {}

    grouped = (
        train
        .groupby("city_name")
    )

    for city, subset in grouped:

        result[city] = float(
            subset["error"].mean()
        )

    return result


def calculate_city_horizon_bias(
    train,
):

    result = {}

    grouped = train.groupby(
        [
            "city_name",
            "horizon",
        ]
    )

    for (
        city,
        horizon,
    ), subset in grouped:

        result[
            f"{city}|{horizon}"
        ] = {

            "bias":
                float(
                    subset["error"].mean()
                ),

            "rows":
                int(
                    len(subset)
                ),
        }

    return result


# ============================================================
# MOMENTUM
# ============================================================

def calculate_momentum_signal(
    df,
):

    """
    Estimate the direction implied by the current AQI
    versus the next forecast.

    Positive:
        forecast is above current AQI.

    Negative:
        forecast is below current AQI.

    The signal is normalized to avoid large corrections.
    """

    signal = (
        df["predicted_aqi"]
        - df["origin_aqi"]
    )

    # Squash very large changes.
    signal = (
        np.tanh(
            signal / 50.0
        )
        * 15.0
    )

    return signal


def calculate_momentum_bias(
    train,
):

    """
    Determine whether the model tends to overreact
    or underreact to momentum.

    We compare:

        predicted change
        vs
        actual change
    """

    predicted_change = (
        train["predicted_aqi"]
        - train["origin_aqi"]
    )

    actual_change = (
        train["actual_aqi"]
        - train["origin_aqi"]
    )

    momentum_error = (
        predicted_change
        - actual_change
    )

    result = {}

    temp = train.copy()

    temp["momentum_error"] = (
        momentum_error
    )

    grouped = temp.groupby(
        "horizon"
    )

    for horizon, subset in grouped:

        result[horizon] = float(
            subset[
                "momentum_error"
            ].mean()
        )

    return result


# ============================================================
# APPLY STRATEGIES
# ============================================================

def clip_correction(
    correction,
):

    return np.clip(
        correction,
        -MAX_CORRECTION,
        MAX_CORRECTION,
    )


def apply_base(
    df,
):

    return (
        df["predicted_aqi"]
        .to_numpy()
        .copy()
    )


def apply_horizon_bias(
    df,
    horizon_bias,
):

    correction = (
        df["horizon"]
        .map(horizon_bias)
        .fillna(0.0)
        .to_numpy()
    )

    return (
        df["predicted_aqi"]
        .to_numpy()
        - clip_correction(
            correction
        )
    )


def apply_city_bias(
    df,
    city_bias,
):

    correction = (
        df["city_name"]
        .map(city_bias)
        .fillna(0.0)
        .to_numpy()
    )

    return (
        df["predicted_aqi"]
        .to_numpy()
        - clip_correction(
            correction
        )
    )


def apply_city_horizon_bias(
    df,
    city_horizon_bias,
):

    corrections = []

    for _, row in df.iterrows():

        key = (
            f"{row['city_name']}"
            f"|"
            f"{int(row['horizon'])}"
        )

        item = (
            city_horizon_bias
            .get(key)
        )

        if item is None:

            correction = 0.0

        else:

            correction = float(
                item["bias"]
            )

        corrections.append(
            correction
        )

    corrections = clip_correction(
        np.asarray(
            corrections,
            dtype=float,
        )
    )

    return (
        df["predicted_aqi"]
        .to_numpy()
        - corrections
    )


def apply_shrunk_city_horizon(
    df,
    city_horizon_bias,
    horizon_bias,
    shrinkage,
):

    corrections = []

    for _, row in df.iterrows():

        horizon = int(
            row["horizon"]
        )

        city = row[
            "city_name"
        ]

        key = (
            f"{city}|{horizon}"
        )

        city_item = (
            city_horizon_bias
            .get(key)
        )

        city_bias = (
            0.0
            if city_item is None
            else float(
                city_item["bias"]
            )
        )

        horizon_mean = float(
            horizon_bias.get(
                horizon,
                0.0,
            )
        )

        if city_item is None:

            correction = (
                horizon_mean
            )

        else:

            rows = int(
                city_item["rows"]
            )

            reliability = (
                rows
                / (
                    rows
                    + SHRINKAGE_MIN_ROWS
                )
            )

            reliability *= (
                shrinkage
            )

            correction = (
                reliability
                * city_bias
                + (
                    1
                    - reliability
                )
                * horizon_mean
            )

        corrections.append(
            correction
        )

    corrections = clip_correction(
        np.asarray(
            corrections,
            dtype=float,
        )
    )

    return (
        df["predicted_aqi"]
        .to_numpy()
        - corrections
    )


def apply_city_horizon_momentum(
    df,
    city_horizon_bias,
    horizon_bias,
    momentum_bias,
    shrinkage,
    momentum_strength,
):

    base = (
        apply_shrunk_city_horizon(
            df,
            city_horizon_bias,
            horizon_bias,
            shrinkage,
        )
    )

    momentum_signal = (
        calculate_momentum_signal(
            df
        )
    )

    momentum_bias_values = (
        df["horizon"]
        .map(momentum_bias)
        .fillna(0.0)
        .to_numpy()
    )

    # Correction toward the learned momentum behavior.
    #
    # If the recursive forecast systematically
    # overshoots momentum, this reduces it.
    momentum_correction = (
        momentum_signal
        * momentum_strength
        * 0.25
    )

    # Remove systematic momentum error.
    momentum_correction += (
        momentum_bias_values
        * momentum_strength
        * 0.15
    )

    correction = clip_correction(
        momentum_correction
    )

    return (
        base
        - correction
    )


# ============================================================
# EVALUATE STRATEGY
# ============================================================

def evaluate_strategy(
    name,
    evaluation,
    predictions,
):

    metrics = calculate_metrics(
        evaluation["actual_aqi"],
        predictions,
    )

    return {
        "strategy":
            name,
        **metrics,
    }


# ============================================================
# CITY ANALYSIS
# ============================================================

def city_analysis(
    evaluation,
    best_predictions,
):

    temp = evaluation.copy()

    temp[
        "calibrated_prediction"
    ] = best_predictions

    rows = []

    for city, subset in (
        temp.groupby("city_name")
    ):

        base_metrics = (
            calculate_metrics(
                subset["actual_aqi"],
                subset[
                    "predicted_aqi"
                ],
            )
        )

        calibrated_metrics = (
            calculate_metrics(
                subset["actual_aqi"],
                subset[
                    "calibrated_prediction"
                ],
            )
        )

        rows.append({

            "city_name":
                city,

            "rows":
                len(subset),

            "base_mae":
                base_metrics["mae"],

            "calibrated_mae":
                calibrated_metrics["mae"],

            "mae_improvement":
                (
                    base_metrics["mae"]
                    - calibrated_metrics["mae"]
                ),

            "base_rmse":
                base_metrics["rmse"],

            "calibrated_rmse":
                calibrated_metrics["rmse"],

            "rmse_improvement":
                (
                    base_metrics["rmse"]
                    - calibrated_metrics["rmse"]
                ),

            "base_bias":
                base_metrics["bias"],

            "calibrated_bias":
                calibrated_metrics["bias"],
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "mae_improvement",
        ascending=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_validation_data()

    calibration_data, evaluation_data = (
        split_train_evaluation(
            df
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "LEARNING CALIBRATION PARAMETERS"
    )

    print(
        "=" * 70
    )

    horizon_bias = (
        calculate_global_horizon_bias(
            calibration_data
        )
    )

    city_bias = (
        calculate_city_bias(
            calibration_data
        )
    )

    city_horizon_bias = (
        calculate_city_horizon_bias(
            calibration_data
        )
    )

    momentum_bias = (
        calculate_momentum_bias(
            calibration_data
        )
    )

    print()
    print(
        "HORIZON BIAS"
    )

    for horizon in HORIZONS:

        print(
            f"{horizon * 24}h: "
            f"{horizon_bias[horizon]:+.4f}"
        )

    print()
    print(
        "MOMENTUM BIAS"
    )

    for horizon in HORIZONS:

        print(
            f"{horizon * 24}h: "
            f"{momentum_bias[horizon]:+.4f}"
        )

    # ========================================================
    # TEST STRATEGIES
    # ========================================================

    strategy_results = []

    predictions_store = {}

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    base_predictions = (
        apply_base(
            evaluation_data
        )
    )

    strategy_results.append(
        evaluate_strategy(
            "BASE",
            evaluation_data,
            base_predictions,
        )
    )

    predictions_store[
        "BASE"
    ] = base_predictions

    # --------------------------------------------------------
    # HORIZON
    # --------------------------------------------------------

    horizon_predictions = (
        apply_horizon_bias(
            evaluation_data,
            horizon_bias,
        )
    )

    strategy_results.append(
        evaluate_strategy(
            "HORIZON_BIAS",
            evaluation_data,
            horizon_predictions,
        )
    )

    predictions_store[
        "HORIZON_BIAS"
    ] = horizon_predictions

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    city_predictions = (
        apply_city_bias(
            evaluation_data,
            city_bias,
        )
    )

    strategy_results.append(
        evaluate_strategy(
            "CITY_BIAS",
            evaluation_data,
            city_predictions,
        )
    )

    predictions_store[
        "CITY_BIAS"
    ] = city_predictions

    # --------------------------------------------------------
    # CITY + HORIZON
    # --------------------------------------------------------

    city_horizon_predictions = (
        apply_city_horizon_bias(
            evaluation_data,
            city_horizon_bias,
        )
    )

    strategy_results.append(
        evaluate_strategy(
            "CITY_HORIZON_BIAS",
            evaluation_data,
            city_horizon_predictions,
        )
    )

    predictions_store[
        "CITY_HORIZON_BIAS"
    ] = city_horizon_predictions

    # --------------------------------------------------------
    # SHRINKAGE
    # --------------------------------------------------------

    best_shrinkage = None
    best_shrinkage_mae = float(
        "inf"
    )

    for shrinkage in (
        SHRINKAGE_STRENGTHS
    ):

        predictions = (
            apply_shrunk_city_horizon(
                evaluation_data,
                city_horizon_bias,
                horizon_bias,
                shrinkage,
            )
        )

        metrics = calculate_metrics(
            evaluation_data[
                "actual_aqi"
            ],
            predictions,
        )

        strategy_name = (
            f"CITY_HORIZON_SHRUNK_"
            f"{shrinkage:.2f}"
        )

        strategy_results.append({

            "strategy":
                strategy_name,

            **metrics,
        })

        predictions_store[
            strategy_name
        ] = predictions

        if (
            metrics["mae"]
            < best_shrinkage_mae
        ):

            best_shrinkage_mae = (
                metrics["mae"]
            )

            best_shrinkage = (
                shrinkage
            )

    # --------------------------------------------------------
    # MOMENTUM SEARCH
    # --------------------------------------------------------

    best_strategy = None
    best_predictions = None
    best_mae = float(
        "inf"
    )

    for shrinkage in (
        SHRINKAGE_STRENGTHS
    ):

        for momentum_strength in (
            MOMENTUM_STRENGTHS
        ):

            predictions = (
                apply_city_horizon_momentum(
                    evaluation_data,
                    city_horizon_bias,
                    horizon_bias,
                    momentum_bias,
                    shrinkage,
                    momentum_strength,
                )
            )

            metrics = calculate_metrics(
                evaluation_data[
                    "actual_aqi"
                ],
                predictions,
            )

            strategy_name = (
                "CITY_HORIZON_MOMENTUM_"
                f"S{shrinkage:.2f}_"
                f"M{momentum_strength:.2f}"
            )

            strategy_results.append({

                "strategy":
                    strategy_name,

                "shrinkage":
                    shrinkage,

                "momentum_strength":
                    momentum_strength,

                **metrics,
            })

            predictions_store[
                strategy_name
            ] = predictions

            if (
                metrics["mae"]
                < best_mae
            ):

                best_mae = (
                    metrics["mae"]
                )

                best_strategy = (
                    strategy_name
                )

                best_predictions = (
                    predictions
                )

    # ========================================================
    # RESULTS
    # ========================================================

    results = pd.DataFrame(
        strategy_results
    )

    results = results.sort_values(
        "mae",
        ascending=True,
    ).reset_index(
        drop=True
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    if best_predictions is None:

        raise RuntimeError(
            "No calibration strategy was selected."
        )

    # ========================================================
    # BEST STRATEGY
    # ========================================================

    best_row = results.iloc[0]

    # Extract parameters from strategy.
    best_name = str(
        best_row["strategy"]
    )

    best_shrinkage_final = 1.0
    best_momentum_final = 0.0

    if (
        "S" in best_name
        and "M" in best_name
    ):

        try:

            parts = (
                best_name.split("_")
            )

            for part in parts:

                if part.startswith("S"):

                    best_shrinkage_final = float(
                        part[1:]
                    )

                if part.startswith("M"):

                    best_momentum_final = float(
                        part[1:]
                    )

        except Exception:

            pass

    # Retrieve actual best predictions.
    best_predictions = (
        predictions_store[
            best_name
        ]
    )

    # ========================================================
    # CITY RESULTS
    # ========================================================

    city_results = city_analysis(
        evaluation_data,
        best_predictions,
    )

    city_results.to_csv(
        CITY_RESULTS_FILE,
        index=False,
    )

    # ========================================================
    # DETAILED PREDICTIONS
    # ========================================================

    prediction_output = (
        evaluation_data[
            [
                "city_id",
                "city_name",
                "origin_date",
                "forecast_date",
                "horizon",
                "origin_aqi",
                "actual_aqi",
                "predicted_aqi",
            ]
        ].copy()
    )

    prediction_output[
        "calibrated_prediction"
    ] = best_predictions

    prediction_output[
        "base_error"
    ] = (
        prediction_output[
            "predicted_aqi"
        ]
        - prediction_output[
            "actual_aqi"
        ]
    )

    prediction_output[
        "calibrated_error"
    ] = (
        prediction_output[
            "calibrated_prediction"
        ]
        - prediction_output[
            "actual_aqi"
        ]
    )

    prediction_output[
        "base_absolute_error"
    ] = (
        prediction_output[
            "base_error"
        ]
        .abs()
    )

    prediction_output[
        "calibrated_absolute_error"
    ] = (
        prediction_output[
            "calibrated_error"
        ]
        .abs()
    )

    prediction_output[
        "improvement"
    ] = (
        prediction_output[
            "base_absolute_error"
        ]
        - prediction_output[
            "calibrated_absolute_error"
        ]
    )

    prediction_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # SAVE PARAMETERS
    # ========================================================

    parameters = {

        "project":
            "PearlsAQI",

        "version":
            "forecast_calibration_v2",

        "source_validation_file":
            str(
                VALIDATION_FILE
            ),

        "train_fraction":
            TRAIN_FRACTION,

        "max_correction":
            MAX_CORRECTION,

        "selected_strategy":
            best_name,

        "selected_mae":
            float(
                best_row["mae"]
            ),

        "selected_rmse":
            float(
                best_row["rmse"]
            ),

        "selected_r2":
            float(
                best_row["r2"]
            ),

        "selected_shrinkage":
            best_shrinkage_final,

        "selected_momentum_strength":
            best_momentum_final,

        "horizon_bias":
            {
                str(k):
                    float(v)
                for k, v
                in horizon_bias.items()
            },

        "city_bias":
            {
                str(k):
                    float(v)
                for k, v
                in city_bias.items()
            },

        "momentum_bias":
            {
                str(k):
                    float(v)
                for k, v
                in momentum_bias.items()
            },

        "city_horizon_bias":
            city_horizon_bias,

        "generated_at":
            pd.Timestamp.now().isoformat(),
    }

    with open(
        PARAMETERS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            parameters,
            file,
            indent=4,
        )

    # ========================================================
    # PRINT REPORT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "CALIBRATION STRATEGY RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "strategy",
        "mae",
        "rmse",
        "r2",
        "bias",
        "within_10",
        "within_20",
        "within_30",
    ]

    available_display = [
        column
        for column in display_columns
        if column in results.columns
    ]

    print()

    print(
        results[
            available_display
        ]
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "BEST CALIBRATION"
    )

    print(
        "=" * 70
    )

    base_metrics = calculate_metrics(
        evaluation_data[
            "actual_aqi"
        ],
        evaluation_data[
            "predicted_aqi"
        ],
    )

    best_metrics = calculate_metrics(
        evaluation_data[
            "actual_aqi"
        ],
        best_predictions,
    )

    print()
    print(
        f"Strategy : {best_name}"
    )

    print(
        f"Base MAE : "
        f"{base_metrics['mae']:.4f}"
    )

    print(
        f"V2 MAE   : "
        f"{best_metrics['mae']:.4f}"
    )

    print(
        f"MAE improvement : "
        f"{base_metrics['mae'] - best_metrics['mae']:.4f}"
    )

    print()
    print(
        f"Base RMSE : "
        f"{base_metrics['rmse']:.4f}"
    )

    print(
        f"V2 RMSE   : "
        f"{best_metrics['rmse']:.4f}"
    )

    print(
        f"RMSE improvement : "
        f"{base_metrics['rmse'] - best_metrics['rmse']:.4f}"
    )

    print()
    print(
        f"Base R² : "
        f"{base_metrics['r2']:.4f}"
    )

    print(
        f"V2 R²   : "
        f"{best_metrics['r2']:.4f}"
    )

    print()
    print(
        "CITY RESULTS"
    )

    print()

    print(
        city_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Strategies : {RESULTS_FILE}"
    )

    print(
        f"Predictions: {PREDICTIONS_FILE}"
    )

    print(
        f"Parameters : {PARAMETERS_FILE}"
    )

    print(
        f"City stats : {CITY_RESULTS_FILE}"
    )

    print()
    print(
        "PearlsAQI Forecast Calibration V2 completed."
    )


if __name__ == "__main__":
    main()