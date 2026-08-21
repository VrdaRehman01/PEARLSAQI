"""
======================================================================
PEARLSAQI FORECAST CALIBRATION V9
======================================================================

Production-safe calibration layer for the V4 XGBoost forecast model.

The calibration layer learns historical residual bias from the
walk-forward forecast validation dataset.

It produces:

    - city_bias
    - horizon_bias
    - regime_bias
    - city_horizon_bias
    - city_regime_bias

The production forecast engine then applies:

    calibrated_prediction =
        base_prediction - correction

where correction represents historical prediction error:

    error = prediction - actual

Therefore:

    positive bias  -> prediction was historically too high
                   -> calibration subtracts it

    negative bias  -> prediction was historically too low
                   -> calibration adds it

SAFETY:

    - No future production AQI is used.
    - Only historical validation results are used.
    - Lookup tables require minimum observations.
    - Extreme residuals are robustly clipped.
    - Corrections are shrunk.
    - Maximum correction is capped.
    - Final calibration tables are generated separately from
      parameter-selection evaluation.
======================================================================
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# PATHS
# ======================================================================

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
    / "calibration_v9"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "calibration_parameters.json"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

# Minimum number of historical observations required before a local
# calibration statistic is trusted.

MIN_CITY_COUNT = 30
MIN_HORIZON_COUNT = 30
MIN_REGIME_COUNT = 20
MIN_CITY_HORIZON_COUNT = 15
MIN_CITY_REGIME_COUNT = 15


# Robust residual limits.
#
# Calibration should correct systematic bias, not chase individual
# catastrophic observations.

MAX_RAW_BIAS = 40.0
MAX_FINAL_CORRECTION = 10.0


# Shrinkage applied to learned residual bias.

SHRINKAGE = 0.75


# Weights used by production engine.

CITY_WEIGHT = 1.0
HORIZON_WEIGHT = 1.0
REGIME_WEIGHT = 0.25

# Momentum is deliberately disabled initially.
#
# Your production engine already calculates momentum, but historical
# momentum calibration is much more sensitive to recursive state
# differences. We will enable this later only if validation proves
# that it improves performance.

MOMENTUM_WEIGHT = 0.0

# Spike component is disabled because the current production engine
# does not actually consume a spike lookup table.

SPIKE_WEIGHT = 0.0


# Minimum improvement required for a correction to be useful.

MINIMUM_BENEFIT = 0.25


# Walk-forward selection.

CALIBRATION_MONTHS = 18
EVALUATION_MONTHS = 6
STEP_MONTHS = 3


# Candidate parameter search.

SHRINKAGE_VALUES = [
    0.50,
    0.75,
    1.00,
]

CITY_WEIGHT_VALUES = [
    0.50,
    0.75,
    1.00,
]

HORIZON_WEIGHT_VALUES = [
    0.50,
    0.75,
    1.00,
]

REGIME_WEIGHT_VALUES = [
    0.00,
    0.25,
    0.50,
]

MAX_CORRECTION_VALUES = [
    5.0,
    7.5,
    10.0,
]


# ======================================================================
# SAFE HELPERS
# ======================================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def get_regime(aqi):

    aqi = safe_float(aqi)

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


def calculate_metrics(actual, prediction):

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
            "mae": np.inf,
            "rmse": np.inf,
            "bias": np.nan,
            "within_20": 0.0,
        }

    error = prediction - actual

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    bias = float(
        np.mean(error)
    )

    within_20 = float(
        np.mean(
            np.abs(error) <= 20
        )
        * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "within_20": within_20,
    }


# ======================================================================
# LOAD VALIDATION DATA
# ======================================================================

def load_validation():

    print()
    print("=" * 70)
    print("LOADING FORECAST VALIDATION DATA")
    print("=" * 70)

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            "\nValidation file not found:\n"
            f"{VALIDATION_FILE}\n\n"
            "Run the recursive/walk-forward forecast validation first."
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Raw validation rows : {len(df):,}"
    )

    required_columns = [
        "city_name",
        "horizon",
        "forecast_date",
        "actual_aqi",
        "predicted_aqi",
        "origin_aqi",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            +
            "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------------
    # Normalize
    # --------------------------------------------------------------

    df["city_name"] = (
        df["city_name"]
        .astype(str)
        .str.strip()
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )

    df["horizon"] = pd.to_numeric(
        df["horizon"],
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

    df = df.dropna(
        subset=[
            "city_name",
            "forecast_date",
            "horizon",
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
    # Historical residual
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
    # Regime based on information available at origin.
    # --------------------------------------------------------------

    df["regime"] = (
        df["origin_aqi"]
        .apply(get_regime)
    )

    # --------------------------------------------------------------
    # Remove impossible values
    # --------------------------------------------------------------

    df = df[
        np.isfinite(df["error"])
    ].copy()

    df = (
        df
        .sort_values(
            "forecast_date"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Clean validation rows: {len(df):,}"
    )

    print(
        f"Cities               : "
        f"{df['city_name'].nunique()}"
    )

    print(
        f"Date range           : "
        f"{df['forecast_date'].min().date()} "
        f"-> "
        f"{df['forecast_date'].max().date()}"
    )

    print()
    print("Historical baseline:")

    baseline = calculate_metrics(
        df["actual_aqi"],
        df["predicted_aqi"],
    )

    print(
        f"MAE      : {baseline['mae']:.4f}"
    )

    print(
        f"RMSE     : {baseline['rmse']:.4f}"
    )

    print(
        f"Bias     : {baseline['bias']:+.4f}"
    )

    print(
        f"Within20 : {baseline['within_20']:.2f}%"
    )

    return df


# ======================================================================
# ROBUST BIAS
# ======================================================================

def robust_bias(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    # Winsorize using robust percentile limits.
    low = values.quantile(0.10)
    high = values.quantile(0.90)

    clipped = values.clip(
        lower=low,
        upper=high,
    )

    bias = float(
        clipped.mean()
    )

    return float(
        np.clip(
            bias,
            -MAX_RAW_BIAS,
            MAX_RAW_BIAS,
        )
    )


# ======================================================================
# BUILD CALIBRATION TABLE
# ======================================================================

def build_group_bias(
    train,
    group_columns,
    minimum_count,
):

    if train.empty:
        return {}

    grouped = (
        train
        .groupby(
            group_columns,
            dropna=False,
        )
    )

    output = {}

    for key, group in grouped:

        count = len(group)

        if count < minimum_count:
            continue

        bias = robust_bias(
            group["error"]
        )

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # pandas may return a tuple even for a single group column.
        #
        # Example:
        #
        # ('Karachi',)
        #
        # But production lookup expects:
        #
        # Karachi
        #
        # Therefore normalize single-column keys explicitly.
        # ----------------------------------------------------------

        if len(group_columns) == 1:

            if isinstance(key, tuple):

                key_string = str(
                    key[0]
                )

            else:

                key_string = str(
                    key
                )

        else:

            if not isinstance(
                key,
                tuple,
            ):

                key = (
                    key,
                )

            key_string = "|".join(
                str(x)
                for x in key
            )

        output[key_string] = round(
            bias,
            6,
        )

    return output

# ======================================================================
# BUILD TABLES
# ======================================================================

def build_tables(train):

    city_bias = build_group_bias(
        train,
        ["city_name"],
        MIN_CITY_COUNT,
    )

    horizon_bias = build_group_bias(
        train,
        ["horizon"],
        MIN_HORIZON_COUNT,
    )

    regime_bias = build_group_bias(
        train,
        ["regime"],
        MIN_REGIME_COUNT,
    )

    city_horizon_bias = build_group_bias(
        train,
        [
            "city_name",
            "horizon",
        ],
        MIN_CITY_HORIZON_COUNT,
    )

    city_regime_bias = build_group_bias(
        train,
        [
            "city_name",
            "regime",
        ],
        MIN_CITY_REGIME_COUNT,
    )

    return {
        "city_bias": city_bias,
        "horizon_bias": horizon_bias,
        "regime_bias": regime_bias,
        "city_horizon_bias": city_horizon_bias,
        "city_regime_bias": city_regime_bias,
    }


# ======================================================================
# LOOKUP
# ======================================================================

def lookup(
    table,
    keys,
):

    if not table:
        return 0.0

    for key in keys:

        if key is None:
            continue

        key = str(key)

        if key in table:

            return safe_float(
                table[key]
            )

    return 0.0


# ======================================================================
# CALCULATE CORRECTION
# ======================================================================

def calculate_correction(
    base_prediction,
    city,
    horizon,
    current_aqi,
    parameters,
    tables,
):

    city_bias = lookup(
        tables.get("city_bias"),
        [city],
    )

    horizon_bias = lookup(
        tables.get("horizon_bias"),
        [
            str(horizon),
            horizon,
            f"horizon_{horizon}",
        ],
    )

    regime = get_regime(
        current_aqi
    )

    regime_bias = lookup(
        tables.get("regime_bias"),
        [regime],
    )

    city_horizon_bias = lookup(
        tables.get("city_horizon_bias"),
        [
            f"{city}|{horizon}",
            f"{city}|{str(horizon)}",
        ],
    )

    city_regime_bias = lookup(
        tables.get("city_regime_bias"),
        [
            f"{city}|{regime}",
        ],
    )

    correction = (

        parameters["city_weight"]
        * city_bias

        +

        parameters["horizon_weight"]
        * horizon_bias

        +

        parameters["regime_weight"]
        * regime_bias

        +

        parameters["city_weight"]
        * parameters["horizon_weight"]
        * city_horizon_bias

        +

        parameters["city_weight"]
        * parameters["regime_weight"]
        * city_regime_bias
    )

    correction *= (
        parameters["shrinkage"]
    )

    correction = float(
        np.clip(
            correction,
            -parameters["max_correction"],
            parameters["max_correction"],
        )
    )

    calibrated = (
        base_prediction
        - correction
    )

    calibrated = max(
        0.0,
        calibrated,
    )

    return calibrated


# ======================================================================
# APPLY TABLES TO DATAFRAME
# ======================================================================

def apply_calibration(
    df,
    tables,
    parameters,
):

    calibrated = []

    for _, row in df.iterrows():

        prediction = safe_float(
            row["predicted_aqi"]
        )

        correction_prediction = (
            calculate_correction(
                base_prediction=prediction,
                city=str(
                    row["city_name"]
                ),
                horizon=int(
                    row["horizon"]
                ),
                current_aqi=safe_float(
                    row["origin_aqi"]
                ),
                parameters=parameters,
                tables=tables,
            )
        )

        calibrated.append(
            correction_prediction
        )

    return np.asarray(
        calibrated,
        dtype=float,
    )


# ======================================================================
# WALK-FORWARD FOLDS
# ======================================================================

def generate_walk_forward_folds(df):

    dates = (
        pd.to_datetime(
            df["forecast_date"]
        )
        .sort_values()
        .drop_duplicates()
    )

    if dates.empty:
        return []

    first_date = dates.iloc[0]
    last_date = dates.iloc[-1]

    folds = []

    evaluation_start = (
        first_date
        + pd.DateOffset(
            months=CALIBRATION_MONTHS
        )
    )

    while (
        evaluation_start
        < last_date
    ):

        evaluation_end = (
            evaluation_start
            + pd.DateOffset(
                months=EVALUATION_MONTHS
            )
        )

        train_start = (
            evaluation_start
            - pd.DateOffset(
                months=CALIBRATION_MONTHS
            )
        )

        train_mask = (
            (df["forecast_date"] >= train_start)
            &
            (df["forecast_date"] < evaluation_start)
        )

        evaluation_mask = (
            (df["forecast_date"] >= evaluation_start)
            &
            (df["forecast_date"] < evaluation_end)
        )

        train = df[
            train_mask
        ].copy()

        evaluation = df[
            evaluation_mask
        ].copy()

        if (
            len(train) >= 100
            and len(evaluation) >= 30
        ):

            folds.append(
                (
                    train,
                    evaluation,
                )
            )

        evaluation_start = (
            evaluation_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


# ======================================================================
# PARAMETER SEARCH
# ======================================================================

def search_parameters(df):

    print()
    print("=" * 70)
    print("WALK-FORWARD CALIBRATION SEARCH")
    print("=" * 70)

    folds = generate_walk_forward_folds(
        df
    )

    print(
        f"Walk-forward folds: {len(folds)}"
    )

    if not folds:

        print(
            "Not enough historical validation data "
            "for walk-forward parameter selection."
        )

        print(
            "Using conservative production defaults."
        )

        return {
            "shrinkage": SHRINKAGE,
            "city_weight": CITY_WEIGHT,
            "horizon_weight": HORIZON_WEIGHT,
            "regime_weight": REGIME_WEIGHT,
            "momentum_weight": MOMENTUM_WEIGHT,
            "spike_weight": SPIKE_WEIGHT,
            "max_correction": MAX_FINAL_CORRECTION,
            "minimum_benefit": MINIMUM_BENEFIT,
        }

    candidates = []

    for shrinkage in SHRINKAGE_VALUES:

        for city_weight in CITY_WEIGHT_VALUES:

            for horizon_weight in HORIZON_WEIGHT_VALUES:

                for regime_weight in REGIME_WEIGHT_VALUES:

                    for max_correction in MAX_CORRECTION_VALUES:

                        candidates.append({

                            "shrinkage":
                                shrinkage,

                            "city_weight":
                                city_weight,

                            "horizon_weight":
                                horizon_weight,

                            "regime_weight":
                                regime_weight,

                            "momentum_weight":
                                0.0,

                            "spike_weight":
                                0.0,

                            "max_correction":
                                max_correction,

                            "minimum_benefit":
                                MINIMUM_BENEFIT,
                        })

    print(
        f"Parameter combinations: "
        f"{len(candidates)}"
    )

    best_parameters = None
    best_score = np.inf

    results = []

    for index, parameters in enumerate(
        candidates,
        start=1,
    ):

        fold_scores = []

        for train, evaluation in folds:

            tables = build_tables(
                train
            )

            predictions = apply_calibration(
                evaluation,
                tables,
                parameters,
            )

            metrics = calculate_metrics(
                evaluation["actual_aqi"],
                predictions,
            )

            fold_scores.append(
                metrics["mae"]
            )

        if not fold_scores:
            continue

        score = float(
            np.mean(
                fold_scores
            )
        )

        results.append({

            **parameters,

            "walk_forward_mae":
                score,

        })

        if score < best_score:

            best_score = score

            best_parameters = (
                parameters.copy()
            )

    if best_parameters is None:

        best_parameters = {

            "shrinkage":
                SHRINKAGE,

            "city_weight":
                CITY_WEIGHT,

            "horizon_weight":
                HORIZON_WEIGHT,

            "regime_weight":
                REGIME_WEIGHT,

            "momentum_weight":
                0.0,

            "spike_weight":
                0.0,

            "max_correction":
                MAX_FINAL_CORRECTION,

            "minimum_benefit":
                MINIMUM_BENEFIT,
        }

    print()
    print("Best calibration parameters:")

    for key, value in best_parameters.items():

        print(
            f"{key:<20}: {value}"
        )

    print(
        f"\nWalk-forward MAE : "
        f"{best_score:.4f}"
    )

    return best_parameters


# ======================================================================
# CALIBRATION IMPACT REPORT
# ======================================================================

def print_calibration_report(
    df,
    tables,
    parameters,
):

    print()
    print("=" * 70)
    print("CALIBRATION IMPACT")
    print("=" * 70)

    base_metrics = calculate_metrics(
        df["actual_aqi"],
        df["predicted_aqi"],
    )

    calibrated_predictions = (
        apply_calibration(
            df,
            tables,
            parameters,
        )
    )

    calibrated_metrics = calculate_metrics(
        df["actual_aqi"],
        calibrated_predictions,
    )

    print()
    print("BASE MODEL")

    print(
        f"MAE      : "
        f"{base_metrics['mae']:.4f}"
    )

    print(
        f"RMSE     : "
        f"{base_metrics['rmse']:.4f}"
    )

    print(
        f"Bias     : "
        f"{base_metrics['bias']:+.4f}"
    )

    print(
        f"Within20 : "
        f"{base_metrics['within_20']:.2f}%"
    )

    print()
    print("CALIBRATED MODEL")

    print(
        f"MAE      : "
        f"{calibrated_metrics['mae']:.4f}"
    )

    print(
        f"RMSE     : "
        f"{calibrated_metrics['rmse']:.4f}"
    )

    print(
        f"Bias     : "
        f"{calibrated_metrics['bias']:+.4f}"
    )

    print(
        f"Within20 : "
        f"{calibrated_metrics['within_20']:.2f}%"
    )

    improvement = (
        base_metrics["mae"]
        -
        calibrated_metrics["mae"]
    )

    print()
    print(
        f"MAE improvement: "
        f"{improvement:+.4f}"
    )

    print()
    print("Learned lookup tables:")

    for table_name, table in tables.items():

        print(
            f"  {table_name:<22}: "
            f"{len(table)} entries"
        )


# ======================================================================
# PRINT TABLES
# ======================================================================

def print_tables(tables):

    print()
    print("=" * 70)
    print("CALIBRATION LOOKUP TABLES")
    print("=" * 70)

    for table_name, table in tables.items():

        print()
        print(
            f"{table_name}:"
        )

        if not table:

            print(
                "  <empty>"
            )

            continue

        sorted_items = sorted(
            table.items(),
            key=lambda x: abs(
                safe_float(x[1])
            ),
            reverse=True,
        )

        for key, value in sorted_items[:30]:

            print(
                f"  {key:<45} "
                f"{value:+.4f}"
            )

        if len(sorted_items) > 30:

            print(
                f"  ... "
                f"{len(sorted_items) - 30} more"
            )


# ======================================================================
# SAVE CALIBRATION
# ======================================================================

def save_calibration(
    parameters,
    tables,
    df,
):

    output = {

        "version":
            "V9",

        "calibration_type":
            "walk_forward_residual_bias",

        "description":
            (
                "Production-safe V9 forecast calibration "
                "using historical recursive forecast residuals."
            ),

        "validation_file":
            str(
                VALIDATION_FILE
            ),

        "validation_rows":
            int(
                len(df)
            ),

        "validation_start":
            str(
                df["forecast_date"]
                .min()
                .date()
            ),

        "validation_end":
            str(
                df["forecast_date"]
                .max()
                .date()
            ),

        "selected_parameters":
            parameters,

        "calibration_tables":
            tables,

        # Duplicate top-level tables for compatibility
        # with different production readers.

        "city_bias":
            tables.get(
                "city_bias",
                {},
            ),

        "horizon_bias":
            tables.get(
                "horizon_bias",
                {},
            ),

        "regime_bias":
            tables.get(
                "regime_bias",
                {},
            ),

        "city_horizon_bias":
            tables.get(
                "city_horizon_bias",
                {},
            ),

        "city_regime_bias":
            tables.get(
                "city_regime_bias",
                {},
            ),

        "safety": {

            "minimum_city_count":
                MIN_CITY_COUNT,

            "minimum_horizon_count":
                MIN_HORIZON_COUNT,

            "minimum_regime_count":
                MIN_REGIME_COUNT,

            "minimum_city_horizon_count":
                MIN_CITY_HORIZON_COUNT,

            "minimum_city_regime_count":
                MIN_CITY_REGIME_COUNT,

            "max_raw_bias":
                MAX_RAW_BIAS,

            "max_final_correction":
                MAX_FINAL_CORRECTION,

        },

    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("CALIBRATION FILE SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("PEARLSAQI FORECAST CALIBRATION V9")
    print("=" * 70)

    print(
        "\nProduction-safe residual calibration"
    )

    df = load_validation()

    # --------------------------------------------------------------
    # Search parameters using historical walk-forward folds.
    # --------------------------------------------------------------

    parameters = search_parameters(
        df
    )

    # --------------------------------------------------------------
    # Build final lookup tables using all historical validation
    # results available to the production calibration process.
    # --------------------------------------------------------------

    print()
    print(
        "Building final calibration lookup tables..."
    )

    tables = build_tables(
        df
    )

    print_tables(
        tables
    )

    # --------------------------------------------------------------
    # Report impact.
    #
    # NOTE:
    # This is a diagnostic on the historical calibration dataset.
    # It is NOT the unbiased walk-forward score.
    # --------------------------------------------------------------

    print_calibration_report(
        df,
        tables,
        parameters,
    )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    save_calibration(
        parameters,
        tables,
        df,
    )

    print()
    print("=" * 70)
    print("V9 CALIBRATION COMPLETE")
    print("=" * 70)

    print(
        "\nThe production engine can now consume:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()