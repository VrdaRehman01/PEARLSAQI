"""
PEARLSAQI FORECAST VALIDATION V1

Validates the recursive 24h / 48h / 72h forecast engine
against historical known AQI values.

IMPORTANT:
- Uses only information available at the forecast origin.
- Never uses future target_aqi as an input.
- Measures each forecast horizon separately.
- Does NOT modify the production model.
"""

from pathlib import Path
from datetime import timedelta
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.forecast_engine_v1 import (
    rebuild_features,
    create_future_row,
    prepare_model_input,
    get_category,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "final_production_xgboost"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
    / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "forecast_validation_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "forecast_validation_summary.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "forecast_validation_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

HORIZONS = [1, 2, 3]

# Number of historical forecast origins per city.
# Increase later if runtime is acceptable.
MAX_ORIGINS_PER_CITY = 120

# Keep enough history for the V4 rolling features.
HISTORY_DAYS = 60


# ============================================================
# MODEL
# ============================================================

def find_model():

    candidates = [
        MODEL_DIR / "final_xgboost_model.json",
        MODEL_DIR / "model.json",
        MODEL_DIR / "xgboost_model.json",
        MODEL_DIR / "final_model.json",
    ]

    for path in candidates:

        if path.exists():
            return path

    candidates = list(
        MODEL_DIR.glob("*.json")
    )

    for path in candidates:

        name = path.name.lower()

        if (
            "metric" not in name
            and "metadata" not in name
            and "feature" not in name
        ):
            return path

    raise FileNotFoundError(
        f"No production XGBoost model found in {MODEL_DIR}"
    )


def load_model():

    model_path = find_model()

    print()
    print("Loading production model...")
    print(f"Model: {model_path}")

    model = xgb.XGBRegressor()

    model.load_model(
        str(model_path)
    )

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        booster = model.get_booster()

        feature_names = booster.feature_names

    # Convert NumPy array / tuple to normal Python list
    if feature_names is not None:
        feature_names = list(feature_names)

    if feature_names is None or len(feature_names) == 0:

        raise ValueError(
            "Production model does not contain feature names."
        )

    return (
        model,
        model_path,
        feature_names,
    )

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("Loading V4 feature dataset...")

    df = pd.read_parquet(
        FEATURE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["city_id", "date"]
    ).reset_index(drop=True)

    required = {
        "city_id",
        "city_name",
        "date",
        "aqi",
    }

    missing = required - set(
        df.columns
    )

    if missing:

        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Cities: {df['city_name'].nunique()}"
    )

    return df


# ============================================================
# SINGLE FORECAST
# ============================================================

def predict_state(
    history,
    model,
    feature_names,
):

    rebuilt = rebuild_features(
        history
    )

    future_date = (
        pd.to_datetime(
            history["date"].iloc[-1]
        )
    )

    future_rows = rebuilt[
        rebuilt["date"]
        == future_date
    ]

    if future_rows.empty:

        raise RuntimeError(
            "Unable to locate forecast row."
        )

    X = prepare_model_input(
        future_rows.tail(1),
        feature_names,
    )

    prediction = float(
        model.predict(X)[0]
    )

    return max(
        0.0,
        prediction,
    )


# ============================================================
# WALK-FORWARD FORECAST
# ============================================================

def forecast_from_origin(
    city_df,
    origin_index,
    model,
    feature_names,
):

    """
    Forecast +1 / +2 / +3 days.

    Only rows <= origin date are available
    when the forecast is generated.
    """

    origin_row = city_df.iloc[
        origin_index
    ]

    origin_date = pd.to_datetime(
        origin_row["date"]
    )

    current_aqi = float(
        origin_row["aqi"]
    )

    history_start = max(
        0,
        origin_index - HISTORY_DAYS + 1,
    )

    history = city_df.iloc[
        history_start:
        origin_index + 1
    ].copy()

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    predictions = []

    previous_prediction = current_aqi

    for horizon in HORIZONS:

        future_date = (
            origin_date
            + timedelta(days=horizon)
        )

        seed_aqi = (
            current_aqi
            if horizon == 1
            else previous_prediction
        )

        latest_row = history.iloc[-1]

        future_row = create_future_row(
            latest_row,
            future_date,
            seed_aqi,
        )

        working = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

        working = working.sort_values(
            "date"
        ).reset_index(drop=True)

        prediction = predict_state(
            working,
            model,
            feature_names,
        )

        predictions.append({
            "horizon": horizon,
            "forecast_date": future_date,
            "prediction": prediction,
        })

        previous_prediction = prediction

        # Carry forecast forward recursively.
        future_row["aqi"] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

        history = history.sort_values(
            "date"
        ).tail(
            HISTORY_DAYS
        )

    return predictions


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

    error = prediction - actual

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                squared_error
            )
        )
    )

    actual_mean = float(
        np.mean(actual)
    )

    ss_res = float(
        np.sum(
            (actual - prediction) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (actual - actual_mean) ** 2
        )
    )

    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = 1 - (
            ss_res / ss_tot
        )

    within_10 = float(
        np.mean(
            absolute_error <= 10
        ) * 100
    )

    within_20 = float(
        np.mean(
            absolute_error <= 20
        ) * 100
    )

    within_30 = float(
        np.mean(
            absolute_error <= 30
        ) * 100
    )

    bias = float(
        np.mean(error)
    )

    return {
        "rows": len(actual),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PEARLSAQI FORECAST VALIDATION V1")
    print("=" * 70)

    print()
    print(
        "Validating recursive 24h / 48h / 72h forecasts."
    )

    df = load_data()

    model, model_path, feature_names = (
        load_model()
    )

    print()
    print(
        f"Production feature count: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Only use cities with enough historical data.
    # --------------------------------------------------------

    all_results = []

    cities = (
        df[
            [
                "city_id",
                "city_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("city_id")
    )

    print()
    print(
        f"Cities to validate: "
        f"{len(cities)}"
    )

    for _, city in cities.iterrows():

        city_id = int(
            city["city_id"]
        )

        city_name = str(
            city["city_name"]
        )

        city_df = df[
            df["city_id"]
            == city_id
        ].copy()

        city_df = city_df.sort_values(
            "date"
        ).reset_index(drop=True)

        # Need at least 3 future days.
        max_origin = (
            len(city_df) - 4
        )

        if max_origin <= HISTORY_DAYS:

            print(
                f"Skipping {city_name}: "
                "not enough data."
            )

            continue

        # ----------------------------------------------------
        # Choose evenly distributed origins.
        # ----------------------------------------------------

        first_origin = (
            HISTORY_DAYS - 1
        )

        possible_origins = np.arange(
            first_origin,
            max_origin + 1,
        )

        if (
            len(possible_origins)
            > MAX_ORIGINS_PER_CITY
        ):

            selected_positions = np.linspace(
                0,
                len(possible_origins) - 1,
                MAX_ORIGINS_PER_CITY,
            ).astype(int)

            origins = (
                possible_origins[
                    selected_positions
                ]
            )

        else:

            origins = possible_origins

        print()
        print(
            f"{city_name}: "
            f"{len(origins)} forecast origins"
        )

        for count, origin_index in enumerate(
            origins,
            start=1,
        ):

            origin_date = pd.to_datetime(
                city_df.iloc[
                    origin_index
                ]["date"]
            )

            try:

                forecasts = forecast_from_origin(
                    city_df,
                    origin_index,
                    model,
                    feature_names,
                )

            except Exception as error:

                print()
                print(
                    f"ERROR: "
                    f"{city_name} "
                    f"{origin_date.date()}"
                )

                print(error)

                raise

            for forecast in forecasts:

                horizon = forecast[
                    "horizon"
                ]

                forecast_date = pd.to_datetime(
                    forecast[
                        "forecast_date"
                    ]
                )

                actual_rows = city_df[
                    city_df["date"]
                    == forecast_date
                ]

                if actual_rows.empty:

                    continue

                actual = float(
                    actual_rows.iloc[0]["aqi"]
                )

                prediction = float(
                    forecast["prediction"]
                )

                absolute_error = abs(
                    prediction - actual
                )

                all_results.append({

                    "city_id":
                        city_id,

                    "city_name":
                        city_name,

                    "origin_date":
                        origin_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "forecast_date":
                        forecast_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "horizon":
                        horizon,

                    "horizon_label":
                        f"{horizon * 24}h",

                    "origin_aqi":
                        float(
                            city_df.iloc[
                                origin_index
                            ]["aqi"]
                        ),

                    "actual_aqi":
                        actual,

                    "predicted_aqi":
                        prediction,

                    "error":
                        prediction - actual,

                    "absolute_error":
                        absolute_error,

                    "actual_category":
                        get_category(actual),

                    "predicted_category":
                        get_category(
                            prediction
                        ),
                })

            if count % 25 == 0:

                print(
                    f"  processed "
                    f"{count}/{len(origins)}"
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    if results.empty:

        raise ValueError(
            "No validation results were generated."
        )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Horizon summary
    # --------------------------------------------------------

    summary_rows = []

    for horizon in HORIZONS:

        subset = results[
            results["horizon"]
            == horizon
        ]

        metrics = calculate_metrics(
            subset["actual_aqi"],
            subset["predicted_aqi"],
        )

        summary_rows.append({

            "horizon":
                horizon,

            "horizon_label":
                f"{horizon * 24}h",

            **metrics,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # City + horizon analysis
    # --------------------------------------------------------

    city_rows = []

    for (
        city_name,
        horizon,
    ), subset in results.groupby(
        [
            "city_name",
            "horizon",
        ]
    ):

        metrics = calculate_metrics(
            subset["actual_aqi"],
            subset["predicted_aqi"],
        )

        city_rows.append({

            "city_name":
                city_name,

            "horizon":
                horizon,

            "horizon_label":
                f"{horizon * 24}h",

            **metrics,
        })

    city_summary = pd.DataFrame(
        city_rows
    )

    city_summary_path = (
        OUTPUT_DIR
        / "forecast_validation_city.csv"
    )

    city_summary.to_csv(
        city_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = calculate_metrics(
        results["actual_aqi"],
        results["predicted_aqi"],
    )

    metadata = {

        "project":
            "PearlsAQI",

        "validation_engine":
            "forecast_validation_v1",

        "model":
            str(model_path),

        "feature_count":
            len(feature_names),

        "cities":
            int(
                results[
                    "city_name"
                ].nunique()
            ),

        "validation_rows":
            len(results),

        "max_origins_per_city":
            MAX_ORIGINS_PER_CITY,

        "overall_metrics":
            overall,

        "generated_at":
            pd.Timestamp.now().isoformat(),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            default=str,
        )

    # ========================================================
    # PRINT REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("FORECAST VALIDATION RESULTS")
    print("=" * 70)

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("CITY PERFORMANCE")
    print("=" * 70)

    city_display = (
        city_summary
        .sort_values(
            [
                "horizon",
                "mae",
            ]
        )
    )

    print(
        city_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Rows       : "
        f"{overall['rows']}"
    )

    print(
        f"MAE        : "
        f"{overall['mae']:.4f}"
    )

    print(
        f"RMSE       : "
        f"{overall['rmse']:.4f}"
    )

    print(
        f"R²         : "
        f"{overall['r2']:.4f}"
    )

    print(
        f"Within ±10 : "
        f"{overall['within_10']:.2f}%"
    )

    print(
        f"Within ±20 : "
        f"{overall['within_20']:.2f}%"
    )

    print(
        f"Within ±30 : "
        f"{overall['within_30']:.2f}%"
    )

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"Detailed : {RESULTS_FILE}"
    )

    print(
        f"Summary  : {SUMMARY_FILE}"
    )

    print(
        f"City     : {city_summary_path}"
    )

    print(
        f"Metadata : {METADATA_FILE}"
    )

    print()
    print(
        "PearlsAQI forecast validation completed."
    )


if __name__ == "__main__":
    main()