"""
PEARLSAQI - FORECAST CALIBRATION V6

ROBUST ROLLING TEMPORAL CALIBRATION EXPERIMENT

Purpose
-------
V6 does NOT replace V3/V4/V5.

It evaluates whether calibration strategies genuinely generalize
across multiple temporal walk-forward folds.

The validation dataset must contain:

    city_id
    city_name
    origin_date
    forecast_date
    horizon
    horizon_label
    origin_aqi
    actual_aqi
    predicted_aqi
    error
    absolute_error

V6 tests:

    1. BASELINE
    2. GLOBAL BIAS CORRECTION
    3. HORIZON BIAS CORRECTION
    4. CITY BIAS CORRECTION
    5. CITY + HORIZON CORRECTION
    6. REGIME CORRECTION
    7. CITY + HORIZON + REGIME CORRECTION

with several shrinkage strengths and correction caps.

IMPORTANT
---------
Calibration for every evaluation fold only uses observations
strictly BEFORE that fold's evaluation period.

No future evaluation observations are used for calibration.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
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
    / "calibration_v6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STRATEGY_PATH = (
    OUTPUT_DIR
    / "calibration_strategy_results.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

FOLD_PATH = (
    OUTPUT_DIR
    / "calibration_fold_results.csv"
)

HORIZON_PATH = (
    OUTPUT_DIR
    / "calibration_horizon_results.csv"
)

CITY_PATH = (
    OUTPUT_DIR
    / "calibration_city_results.csv"
)

REGIME_PATH = (
    OUTPUT_DIR
    / "calibration_regime_results.csv"
)

STRESS_PATH = (
    OUTPUT_DIR
    / "calibration_stress_results.csv"
)

PARAMETERS_PATH = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)


# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================

# Number of months used to estimate calibration parameters.
CALIBRATION_MONTHS = 18

# Evaluation window.
EVALUATION_MONTHS = 6

# Move forward by this many months for every fold.
STEP_MONTHS = 3

# Minimum observations required for a group-specific bias.
MIN_GROUP_ROWS = 30

# Shrinkage values.
SHRINKAGES = [
    0.25,
    0.50,
    0.75,
    1.00,
]

# Maximum absolute correction.
CAPS = [
    5.0,
    10.0,
    15.0,
    20.0,
]

# Extreme AQI threshold.
EXTREME_THRESHOLD = 300

# Shock threshold.
SPIKE_THRESHOLD = 40

# Minimum number of folds required for promotion.
MIN_FOLDS = 3


# ============================================================
# PRINT HELPERS
# ============================================================

def banner(title: str):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_validation() -> pd.DataFrame:

    if not VALIDATION_PATH.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n{VALIDATION_PATH}"
        )

    print(
        "\nLoading walk-forward validation results..."
    )

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: {len(df):,}"
    )

    print("\nAvailable columns:")

    for column in df.columns:

        print(
            f"- {column}"
        )

    required = [
        "city_name",
        "horizon",
        "forecast_date",
        "actual_aqi",
        "predicted_aqi",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

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

    df["horizon"] = pd.to_numeric(
        df["horizon"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "forecast_date",
            "actual_aqi",
            "predicted_aqi",
            "horizon",
            "city_name",
        ]
    ).copy()

    df["horizon"] = (
        df["horizon"]
        .astype(int)
    )

    df["city_name"] = (
        df["city_name"]
        .astype(str)
    )

    df = df.sort_values(
        [
            "forecast_date",
            "city_name",
            "horizon",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# AQI REGIME
# ============================================================

def get_regime(aqi: float) -> str:

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

def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:

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

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                prediction,
            )
        )
    )

    try:

        r2 = float(
            r2_score(
                actual,
                prediction,
            )
        )

    except Exception:

        r2 = float("nan")

    return {

        "rows": int(
            len(actual)
        ),

        "mae": float(
            mean_absolute_error(
                actual,
                prediction,
            )
        ),

        "rmse": rmse,

        "r2": r2,

        "bias": float(
            error.mean()
        ),

        "within_10": float(
            (
                absolute_error <= 10
            ).mean()
            * 100
        ),

        "within_20": float(
            (
                absolute_error <= 20
            ).mean()
            * 100
        ),

        "within_30": float(
            (
                absolute_error <= 30
            ).mean()
            * 100
        ),
    }


# ============================================================
# TEMPORAL FOLDS
# ============================================================

def build_folds(
    df: pd.DataFrame,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:

    minimum_date = df[
        "forecast_date"
    ].min()

    maximum_date = df[
        "forecast_date"
    ].max()

    calibration_start = (
        minimum_date
        + pd.DateOffset(
            months=1
        )
    )

    folds = []

    evaluation_start = (
        calibration_start
        + pd.DateOffset(
            months=CALIBRATION_MONTHS
        )
    )

    while True:

        evaluation_end = (
            evaluation_start
            + pd.DateOffset(
                months=EVALUATION_MONTHS
            )
        )

        if evaluation_start >= maximum_date:

            break

        if evaluation_end > maximum_date:

            evaluation_end = (
                maximum_date
                + pd.Timedelta(
                    days=1
                )
            )

        calibration_end = evaluation_start

        folds.append(
            (
                calibration_start,
                calibration_end,
                evaluation_start,
                evaluation_end,
            )
        )

        next_start = (
            evaluation_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

        if next_start >= maximum_date:

            break

        evaluation_start = next_start

        calibration_start = (
            calibration_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


# ============================================================
# BIAS ESTIMATION
# ============================================================

def calculate_bias(
    frame: pd.DataFrame,
) -> float:

    if len(frame) == 0:

        return 0.0

    # Actual - predicted.
    #
    # Positive value means the model is underpredicting.
    bias = (
        frame["actual_aqi"]
        - frame["predicted_aqi"]
    ).mean()

    if not np.isfinite(bias):

        return 0.0

    return float(bias)


# ============================================================
# BUILD CALIBRATION TABLES
# ============================================================

def build_calibration_tables(
    calibration: pd.DataFrame,
):

    global_bias = calculate_bias(
        calibration
    )

    horizon_bias = (
        calibration
        .groupby("horizon")
        .apply(calculate_bias)
        .to_dict()
    )

    city_bias = (
        calibration
        .groupby("city_name")
        .apply(calculate_bias)
        .to_dict()
    )

    city_horizon = (
        calibration
        .groupby(
            [
                "city_name",
                "horizon",
            ]
        )
        .apply(calculate_bias)
        .to_dict()
    )

    regime_bias = (
        calibration
        .groupby("regime")
        .apply(calculate_bias)
        .to_dict()
    )

    city_regime = (
        calibration
        .groupby(
            [
                "city_name",
                "regime",
            ]
        )
        .apply(calculate_bias)
        .to_dict()
    )

    city_horizon_regime = (
        calibration
        .groupby(
            [
                "city_name",
                "horizon",
                "regime",
            ]
        )
        .apply(calculate_bias)
        .to_dict()
    )

    # Observation counts are important.
    # We do not trust tiny groups.
    counts = {

        "horizon":
            calibration
            .groupby("horizon")
            .size()
            .to_dict(),

        "city":
            calibration
            .groupby("city_name")
            .size()
            .to_dict(),

        "city_horizon":
            calibration
            .groupby(
                [
                    "city_name",
                    "horizon",
                ]
            )
            .size()
            .to_dict(),

        "regime":
            calibration
            .groupby("regime")
            .size()
            .to_dict(),

        "city_regime":
            calibration
            .groupby(
                [
                    "city_name",
                    "regime",
                ]
            )
            .size()
            .to_dict(),

        "city_horizon_regime":
            calibration
            .groupby(
                [
                    "city_name",
                    "horizon",
                    "regime",
                ]
            )
            .size()
            .to_dict(),
    }

    return {
        "global": global_bias,
        "horizon": horizon_bias,
        "city": city_bias,
        "city_horizon": city_horizon,
        "regime": regime_bias,
        "city_regime": city_regime,
        "city_horizon_regime": city_horizon_regime,
        "counts": counts,
    }


# ============================================================
# SAFE LOOKUP
# ============================================================

def safe_lookup(
    table: Dict,
    counts: Dict,
    key,
    fallback: float,
) -> float:

    if key not in table:

        return fallback

    if counts.get(
        key,
        0,
    ) < MIN_GROUP_ROWS:

        return fallback

    value = table[key]

    if not np.isfinite(value):

        return fallback

    return float(value)


# ============================================================
# CALIBRATION STRATEGIES
# ============================================================

def get_raw_bias(
    row,
    tables,
    strategy: str,
) -> float:

    global_bias = tables[
        "global"
    ]

    horizon = int(
        row["horizon"]
    )

    city = row[
        "city_name"
    ]

    regime = row[
        "regime"
    ]

    if strategy == "GLOBAL":

        return global_bias

    if strategy == "HORIZON":

        return safe_lookup(
            tables["horizon"],
            tables["counts"]["horizon"],
            horizon,
            global_bias,
        )

    if strategy == "CITY":

        return safe_lookup(
            tables["city"],
            tables["counts"]["city"],
            city,
            global_bias,
        )

    if strategy == "CITY_HORIZON":

        horizon_bias = safe_lookup(
            tables["horizon"],
            tables["counts"]["horizon"],
            horizon,
            global_bias,
        )

        return safe_lookup(
            tables["city_horizon"],
            tables["counts"]["city_horizon"],
            (
                city,
                horizon,
            ),
            horizon_bias,
        )

    if strategy == "REGIME":

        return safe_lookup(
            tables["regime"],
            tables["counts"]["regime"],
            regime,
            global_bias,
        )

    if strategy == "CITY_REGIME":

        regime_bias = safe_lookup(
            tables["regime"],
            tables["counts"]["regime"],
            regime,
            global_bias,
        )

        return safe_lookup(
            tables["city_regime"],
            tables["counts"]["city_regime"],
            (
                city,
                regime,
            ),
            regime_bias,
        )

    if strategy == "CITY_HORIZON_REGIME":

        city_horizon_bias = safe_lookup(
            tables["city_horizon"],
            tables["counts"]["city_horizon"],
            (
                city,
                horizon,
            ),
            global_bias,
        )

        return safe_lookup(
            tables[
                "city_horizon_regime"
            ],
            tables["counts"][
                "city_horizon_regime"
            ],
            (
                city,
                horizon,
                regime,
            ),
            city_horizon_bias,
        )

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# APPLY CALIBRATION
# ============================================================

def apply_calibration(
    evaluation: pd.DataFrame,
    tables,
    strategy: str,
    shrinkage: float,
    cap: float,
) -> pd.DataFrame:

    result = evaluation.copy()

    raw_biases = []

    for _, row in result.iterrows():

        bias = get_raw_bias(
            row,
            tables,
            strategy,
        )

        raw_biases.append(
            bias
        )

    raw_biases = np.asarray(
        raw_biases,
        dtype=float,
    )

    corrections = (
        raw_biases
        * shrinkage
    )

    corrections = np.clip(
        corrections,
        -cap,
        cap,
    )

    result[
        "calibration_bias"
    ] = raw_biases

    result[
        "correction"
    ] = corrections

    result[
        "calibrated_prediction"
    ] = (
        result["predicted_aqi"]
        + corrections
    )

    # AQI cannot be negative.
    result[
        "calibrated_prediction"
    ] = result[
        "calibrated_prediction"
    ].clip(
        lower=0
    )

    return result


# ============================================================
# STRATEGY EVALUATION
# ============================================================

def evaluate_strategy(
    evaluation: pd.DataFrame,
    calibrated: pd.DataFrame,
    strategy_name: str,
    shrinkage: float,
    cap: float,
    fold_id: int,
):

    metrics = calculate_metrics(
        evaluation["actual_aqi"],
        calibrated[
            "calibrated_prediction"
        ],
    )

    baseline_metrics = calculate_metrics(
        evaluation["actual_aqi"],
        evaluation["predicted_aqi"],
    )

    return {

        "fold": fold_id,

        "strategy": strategy_name,

        "shrinkage": shrinkage,

        "max_correction": cap,

        **metrics,

        "mae_improvement":
            baseline_metrics["mae"]
            - metrics["mae"],

        "rmse_improvement":
            baseline_metrics["rmse"]
            - metrics["rmse"],
    }


# ============================================================
# SPECIALIZED STRESS TEST
# ============================================================

def stress_test(
    frame: pd.DataFrame,
    prediction_column: str,
):

    actual = frame[
        "actual_aqi"
    ].to_numpy()

    prediction = frame[
        prediction_column
    ].to_numpy()

    base = frame[
        "predicted_aqi"
    ].to_numpy()

    actual_change = (
        frame["actual_aqi"]
        - frame["origin_aqi"]
    ).to_numpy()

    extreme_mask = (
        actual >= EXTREME_THRESHOLD
    )

    upward_mask = (
        actual_change
        >= SPIKE_THRESHOLD
    )

    downward_mask = (
        actual_change
        <= -SPIKE_THRESHOLD
    )

    rows = []

    def add_group(
        name,
        mask,
    ):

        if mask.sum() == 0:

            return

        base_metrics = calculate_metrics(
            actual[mask],
            base[mask],
        )

        calibrated_metrics = calculate_metrics(
            actual[mask],
            prediction[mask],
        )

        rows.append({

            "group": name,

            "rows": int(
                mask.sum()
            ),

            "base_mae":
                base_metrics["mae"],

            "calibrated_mae":
                calibrated_metrics["mae"],

            "mae_improvement":
                base_metrics["mae"]
                - calibrated_metrics["mae"],

            "base_rmse":
                base_metrics["rmse"],

            "calibrated_rmse":
                calibrated_metrics["rmse"],

            "rmse_improvement":
                base_metrics["rmse"]
                - calibrated_metrics["rmse"],
        })

    add_group(
        "Extreme AQI",
        extreme_mask,
    )

    add_group(
        "Upward Spike",
        upward_mask,
    )

    add_group(
        "Downward Spike",
        downward_mask,
    )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "PEARLSAQI FORECAST CALIBRATION V6"
    )

    print(
        "\nRolling temporal robustness experiment."
    )

    print(
        "V5 remains untouched."
    )

    df = load_validation()

    df["regime"] = (
        df["origin_aqi"]
        .fillna(
            df["predicted_aqi"]
        )
        .apply(
            get_regime
        )
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
        f"Date range: "
        f"{df['forecast_date'].min().date()} "
        f"→ "
        f"{df['forecast_date'].max().date()}"
    )

    # --------------------------------------------------------
    # BUILD TEMPORAL FOLDS
    # --------------------------------------------------------

    folds = build_folds(
        df
    )

    if len(folds) < MIN_FOLDS:

        raise ValueError(
            f"Only {len(folds)} temporal folds "
            f"were created. Need at least "
            f"{MIN_FOLDS}."
        )

    banner(
        "ROLLING TEMPORAL FOLDS"
    )

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

    for i, (
        cal_start,
        cal_end,
        eval_start,
        eval_end,
    ) in enumerate(
        folds,
        start=1,
    ):

        print(
            f"\nFold {i}:"
        )

        print(
            f"  Calibration: "
            f"{cal_start.date()} "
            f"→ "
            f"{cal_end.date()}"
        )

        print(
            f"  Evaluation : "
            f"{eval_start.date()} "
            f"→ "
            f"{eval_end.date()}"
        )

    # --------------------------------------------------------
    # STRATEGIES
    # --------------------------------------------------------

    strategies = [
        "GLOBAL",
        "HORIZON",
        "CITY",
        "CITY_HORIZON",
        "REGIME",
        "CITY_REGIME",
        "CITY_HORIZON_REGIME",
    ]

    all_results = []

    all_predictions = []

    fold_results = []

    stress_results = []

    # --------------------------------------------------------
    # RUN FOLDS
    # --------------------------------------------------------

    for fold_id, (
        cal_start,
        cal_end,
        eval_start,
        eval_end,
    ) in enumerate(
        folds,
        start=1,
    ):

        banner(
            f"FOLD {fold_id}/{len(folds)}"
        )

        calibration = df[
            (
                df["forecast_date"]
                >= cal_start
            )
            &
            (
                df["forecast_date"]
                < cal_end
            )
        ].copy()

        evaluation = df[
            (
                df["forecast_date"]
                >= eval_start
            )
            &
            (
                df["forecast_date"]
                < eval_end
            )
        ].copy()

        print(
            f"Calibration rows: "
            f"{len(calibration):,}"
        )

        print(
            f"Evaluation rows : "
            f"{len(evaluation):,}"
        )

        if len(calibration) == 0:

            print(
                "Skipping fold: "
                "no calibration rows."
            )

            continue

        if len(evaluation) == 0:

            print(
                "Skipping fold: "
                "no evaluation rows."
            )

            continue

        tables = (
            build_calibration_tables(
                calibration
            )
        )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        baseline_metrics = (
            calculate_metrics(
                evaluation[
                    "actual_aqi"
                ],
                evaluation[
                    "predicted_aqi"
                ],
            )
        )

        fold_results.append({

            "fold": fold_id,

            "strategy": "BASELINE",

            "shrinkage": 0.0,

            "max_correction": 0.0,

            **baseline_metrics,

            "calibration_start":
                cal_start.strftime(
                    "%Y-%m-%d"
                ),

            "calibration_end":
                cal_end.strftime(
                    "%Y-%m-%d"
                ),

            "evaluation_start":
                eval_start.strftime(
                    "%Y-%m-%d"
                ),

            "evaluation_end":
                eval_end.strftime(
                    "%Y-%m-%d"
                ),
        })

        # ----------------------------------------------------
        # STRATEGIES
        # ----------------------------------------------------

        for strategy in strategies:

            for shrinkage in SHRINKAGES:

                for cap in CAPS:

                    calibrated = (
                        apply_calibration(
                            evaluation,
                            tables,
                            strategy,
                            shrinkage,
                            cap,
                        )
                    )

                    result = (
                        evaluate_strategy(
                            evaluation,
                            calibrated,
                            strategy,
                            shrinkage,
                            cap,
                            fold_id,
                        )
                    )

                    result[
                        "calibration_start"
                    ] = (
                        cal_start.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    result[
                        "calibration_end"
                    ] = (
                        cal_end.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    result[
                        "evaluation_start"
                    ] = (
                        eval_start.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    result[
                        "evaluation_end"
                    ] = (
                        eval_end.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    all_results.append(
                        result
                    )

        # ----------------------------------------------------
        # BEST STRATEGY FOR THIS FOLD
        # ----------------------------------------------------

        fold_frame = pd.DataFrame(
            [
                x
                for x in all_results
                if x["fold"] == fold_id
            ]
        )

        if not fold_frame.empty:

            best = (
                fold_frame
                .sort_values(
                    [
                        "mae",
                        "rmse",
                    ]
                )
                .iloc[0]
            )

            print(
                "\nBest fold strategy:"
            )

            print(
                f"  {best['strategy']}"
            )

            print(
                f"  shrinkage = "
                f"{best['shrinkage']:.2f}"
            )

            print(
                f"  cap = "
                f"{best['max_correction']:.1f}"
            )

            print(
                f"  MAE = "
                f"{best['mae']:.4f}"
            )

            print(
                f"  RMSE = "
                f"{best['rmse']:.4f}"
            )

            print(
                f"  R² = "
                f"{best['r2']:.4f}"
            )

        # ----------------------------------------------------
        # STRESS TEST BEST V5-LIKE STRATEGY
        # ----------------------------------------------------

        stress_calibrated = (
            apply_calibration(
                evaluation,
                tables,
                "CITY_HORIZON_REGIME",
                0.50,
                10.0,
            )
        )

        stress_rows = stress_test(
            stress_calibrated,
            "calibrated_prediction",
        )

        for row in stress_rows:

            row[
                "fold"
            ] = fold_id

            row[
                "strategy"
            ] = (
                "CITY_HORIZON_REGIME"
            )

            stress_results.append(
                row
            )

        # ----------------------------------------------------
        # SAVE PREDICTIONS FOR V5-LIKE STRATEGY
        # ----------------------------------------------------

        selected = (
            apply_calibration(
                evaluation,
                tables,
                "CITY_HORIZON_REGIME",
                0.50,
                10.0,
            )
        )

        selected[
            "fold"
        ] = fold_id

        selected[
            "strategy"
        ] = (
            "CITY_HORIZON_REGIME"
        )

        all_predictions.append(
            selected
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:

        raise RuntimeError(
            "No calibration results were generated."
        )

    # ========================================================
    # AGGREGATE STRATEGIES
    # ========================================================

    banner(
        "V6 ROBUSTNESS RESULTS"
    )

    strategy_summary = (
        results_df
        .groupby(
            [
                "strategy",
                "shrinkage",
                "max_correction",
            ]
        )
        .agg(

            folds=(
                "fold",
                "nunique",
            ),

            rows=(
                "rows",
                "sum",
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

            within_10=(
                "within_10",
                "mean",
            ),

            within_20=(
                "within_20",
                "mean",
            ),

            within_30=(
                "within_30",
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
        .reset_index()
    )

    # --------------------------------------------------------
    # ROBUSTNESS SCORE
    # --------------------------------------------------------

    # A strategy should:
    #
    # 1. have low MAE
    # 2. improve RMSE
    # 3. improve MAE
    # 4. have low absolute bias
    # 5. work across all folds
    #
    # This is NOT a production metric.
    # It is only used to rank experiments.

    strategy_summary[
        "abs_bias"
    ] = (
        strategy_summary[
            "bias"
        ].abs()
    )

    strategy_summary[
        "robust_score"
    ] = (

        strategy_summary[
            "mae_improvement"
        ]

        + (
            strategy_summary[
                "rmse_improvement"
            ]
            * 0.50
        )

        - (
            strategy_summary[
                "abs_bias"
            ]
            * 0.10
        )

    )

    # Require all folds.
    strategy_summary[
        "fully_robust"
    ] = (
        strategy_summary[
            "folds"
        ]
        >= len(folds)
    )

    strategy_summary = (
        strategy_summary
        .sort_values(
            [
                "fully_robust",
                "robust_score",
                "mae",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTOP STRATEGIES"
    )

    display_columns = [
        "strategy",
        "shrinkage",
        "max_correction",
        "folds",
        "mae",
        "rmse",
        "r2",
        "bias",
        "mae_improvement",
        "rmse_improvement",
        "robust_score",
    ]

    print(
        strategy_summary[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # BEST STRATEGY
    # ========================================================

    best = (
        strategy_summary
        .iloc[0]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST V6 ROBUST STRATEGY"
    )

    print(
        "=" * 70
    )

    print(
        f"Strategy   : "
        f"{best['strategy']}"
    )

    print(
        f"Shrinkage  : "
        f"{best['shrinkage']:.2f}"
    )

    print(
        f"Cap        : "
        f"{best['max_correction']:.2f}"
    )

    print(
        f"Folds      : "
        f"{int(best['folds'])}"
    )

    print(
        f"MAE        : "
        f"{best['mae']:.4f}"
    )

    print(
        f"RMSE       : "
        f"{best['rmse']:.4f}"
    )

    print(
        f"R²         : "
        f"{best['r2']:.4f}"
    )

    print(
        f"Bias       : "
        f"{best['bias']:.4f}"
    )

    print(
        f"MAE improvement : "
        f"{best['mae_improvement']:.4f}"
    )

    print(
        f"RMSE improvement: "
        f"{best['rmse_improvement']:.4f}"
    )

    # ========================================================
    # SELECT BEST PARAMETERS
    # ========================================================

    best_strategy = str(
        best["strategy"]
    )

    best_shrinkage = float(
        best["shrinkage"]
    )

    best_cap = float(
        best["max_correction"]
    )

    # ========================================================
    # HORIZON ANALYSIS
    # ========================================================

    banner(
        "HORIZON PERFORMANCE"
    )

    if all_predictions:

        prediction_df = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        prediction_df[
            "horizon_label"
        ] = (
            prediction_df[
                "horizon"
            ]
            .map(
                {
                    1: "24h",
                    2: "48h",
                    3: "72h",
                }
            )
        )

        horizon_rows = []

        for horizon, group in (
            prediction_df
            .groupby("horizon")
        ):

            base_metrics = (
                calculate_metrics(
                    group[
                        "actual_aqi"
                    ],
                    group[
                        "predicted_aqi"
                    ],
                )
            )

            calibrated_metrics = (
                calculate_metrics(
                    group[
                        "actual_aqi"
                    ],
                    group[
                        "calibrated_prediction"
                    ],
                )
            )

            horizon_rows.append({

                "horizon":
                    int(horizon),

                "horizon_label":
                    (
                        "24h"
                        if horizon == 1
                        else
                        "48h"
                        if horizon == 2
                        else
                        "72h"
                    ),

                "rows":
                    len(group),

                "base_mae":
                    base_metrics["mae"],

                "v6_mae":
                    calibrated_metrics["mae"],

                "mae_improvement":
                    (
                        base_metrics["mae"]
                        - calibrated_metrics["mae"]
                    ),

                "base_rmse":
                    base_metrics["rmse"],

                "v6_rmse":
                    calibrated_metrics["rmse"],

                "rmse_improvement":
                    (
                        base_metrics["rmse"]
                        - calibrated_metrics["rmse"]
                    ),

                "base_r2":
                    base_metrics["r2"],

                "v6_r2":
                    calibrated_metrics["r2"],
            })

        horizon_df = pd.DataFrame(
            horizon_rows
        )

        print(
            horizon_df.to_string(
                index=False
            )
        )

    else:

        horizon_df = pd.DataFrame()

    # ========================================================
    # CITY ANALYSIS
    # ========================================================

    banner(
        "CITY PERFORMANCE"
    )

    if all_predictions:

        city_rows = []

        for city, group in (
            prediction_df
            .groupby(
                "city_name"
            )
        ):

            base_metrics = (
                calculate_metrics(
                    group[
                        "actual_aqi"
                    ],
                    group[
                        "predicted_aqi"
                    ],
                )
            )

            calibrated_metrics = (
                calculate_metrics(
                    group[
                        "actual_aqi"
                    ],
                    group[
                        "calibrated_prediction"
                    ],
                )
            )

            city_rows.append({

                "city_name":
                    city,

                "rows":
                    len(group),

                "base_mae":
                    base_metrics["mae"],

                "v6_mae":
                    calibrated_metrics["mae"],

                "mae_improvement":
                    (
                        base_metrics["mae"]
                        - calibrated_metrics["mae"]
                    ),

                "base_rmse":
                    base_metrics["rmse"],

                "v6_rmse":
                    calibrated_metrics["rmse"],

                "rmse_improvement":
                    (
                        base_metrics["rmse"]
                        - calibrated_metrics["rmse"]
                    ),

                "base_bias":
                    base_metrics["bias"],

                "v6_bias":
                    calibrated_metrics["bias"],
            })

        city_df = (
            pd.DataFrame(
                city_rows
            )
            .sort_values(
                "mae_improvement",
                ascending=False,
            )
        )

        print(
            city_df.to_string(
                index=False
            )
        )

    else:

        city_df = pd.DataFrame()

    # ========================================================
    # STRESS RESULTS
    # ========================================================

    stress_df = pd.DataFrame(
        stress_results
    )

    if not stress_df.empty:

        stress_summary = (
            stress_df
            .groupby(
                "group"
            )
            .agg(
                folds=(
                    "fold",
                    "nunique",
                ),
                rows=(
                    "rows",
                    "sum",
                ),
                base_mae=(
                    "base_mae",
                    "mean",
                ),
                calibrated_mae=(
                    "calibrated_mae",
                    "mean",
                ),
                mae_improvement=(
                    "mae_improvement",
                    "mean",
                ),
                base_rmse=(
                    "base_rmse",
                    "mean",
                ),
                calibrated_rmse=(
                    "calibrated_rmse",
                    "mean",
                ),
                rmse_improvement=(
                    "rmse_improvement",
                    "mean",
                ),
            )
            .reset_index()
        )

        banner(
            "STRESS TEST"
        )

        print(
            stress_summary.to_string(
                index=False
            )
        )

    else:

        stress_summary = pd.DataFrame()

    # ========================================================
    # SAVE
    # ========================================================

    strategy_summary.to_csv(
        STRATEGY_PATH,
        index=False,
    )

    results_df.to_csv(
        FOLD_PATH,
        index=False,
    )

    if all_predictions:

        prediction_df.to_csv(
            PREDICTIONS_PATH,
            index=False,
        )

    if not horizon_df.empty:

        horizon_df.to_csv(
            HORIZON_PATH,
            index=False,
        )

    if not city_df.empty:

        city_df.to_csv(
            CITY_PATH,
            index=False,
        )

    if not stress_summary.empty:

        stress_summary.to_csv(
            STRESS_PATH,
            index=False,
        )

    # ========================================================
    # SAVE PARAMETERS
    # ========================================================

    parameters = {

        "version": "V6",

        "experiment_type":
            "rolling_temporal_robustness",

        "validation_file":
            str(
                VALIDATION_PATH
            ),

        "calibration_months":
            CALIBRATION_MONTHS,

        "evaluation_months":
            EVALUATION_MONTHS,

        "step_months":
            STEP_MONTHS,

        "minimum_group_rows":
            MIN_GROUP_ROWS,

        "shrinkages":
            SHRINKAGES,

        "caps":
            CAPS,

        "extreme_threshold":
            EXTREME_THRESHOLD,

        "spike_threshold":
            SPIKE_THRESHOLD,

        "fold_count":
            len(folds),

        "selected_strategy":
            best_strategy,

        "selected_shrinkage":
            best_shrinkage,

        "selected_cap":
            best_cap,

        "best_mae":
            float(
                best["mae"]
            ),

        "best_rmse":
            float(
                best["rmse"]
            ),

        "best_r2":
            float(
                best["r2"]
            ),

        "best_bias":
            float(
                best["bias"]
            ),

        "mae_improvement":
            float(
                best[
                    "mae_improvement"
                ]
            ),

        "rmse_improvement":
            float(
                best[
                    "rmse_improvement"
                ]
            ),
    }

    with open(
        PARAMETERS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            parameters,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    banner(
        "V6 COMPLETE"
    )

    print(
        f"Best strategy : "
        f"{best_strategy}"
    )

    print(
        f"Shrinkage     : "
        f"{best_shrinkage:.2f}"
    )

    print(
        f"Correction cap: "
        f"{best_cap:.2f}"
    )

    print(
        f"Folds         : "
        f"{len(folds)}"
    )

    print(
        f"MAE           : "
        f"{best['mae']:.4f}"
    )

    print(
        f"RMSE          : "
        f"{best['rmse']:.4f}"
    )

    print(
        f"R²            : "
        f"{best['r2']:.4f}"
    )

    print(
        f"\nStrategies : "
        f"{STRATEGY_PATH}"
    )

    print(
        f"Fold results: "
        f"{FOLD_PATH}"
    )

    print(
        f"Predictions : "
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"Horizon     : "
        f"{HORIZON_PATH}"
    )

    print(
        f"City        : "
        f"{CITY_PATH}"
    )

    print(
        f"Stress      : "
        f"{STRESS_PATH}"
    )

    print(
        f"Parameters  : "
        f"{PARAMETERS_PATH}"
    )

    print(
        "\nPearlsAQI Forecast Calibration V6 "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()