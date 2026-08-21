"""
PEARLSAQI - PRODUCTION FORECAST ENGINE V9

Recursive production forecasting pipeline using:

    V4 production feature pipeline
    + Production XGBoost model
    + V9 production-safe calibration
    + Persistent forecast history in DuckDB

Forecast horizons:
    24h / 48h / 72h

IMPORTANT:
    - No future actual AQI is used during inference.
    - target_aqi is never used during inference.
    - Forecasts are generated recursively.
    - 48h depends on the generated 24h state.
    - 72h depends on the generated 48h state.
    - V9 calibration is applied independently to each horizon.
    - Previous forecasts are automatically matched with actual AQI.
    - Forecast history is permanently stored in DuckDB.
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb

from src.database.connection import get_connection

from src.features.feature_builder_v4 import (
    add_lag_features,
    add_aqi_rolling_features,
    add_aqi_trend_features,
    add_pollution_features,
    add_weather_features,
    add_regime_features,
    add_city_history_features,
    add_calendar_features,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "final_xgboost_model.json"
)

CALIBRATION_FILE = (
    ROOT
    / "models"
    / "forecast"
    / "calibration_v9"
    / "calibration_parameters.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
    / "v9"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "forecast_predictions.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "forecast_predictions.parquet"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "forecast_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = [1, 2, 3]

HORIZON_LABELS = {
    1: "24h",
    2: "48h",
    3: "72h",
}

FORECAST_HISTORY_DAYS = 60

MODEL_NAME = "PearlsAQI-V9-XGBoost"


# ============================================================
# AQI CATEGORY
# ============================================================

def get_category(aqi):

    aqi = float(aqi)

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


def get_health_message(category):

    messages = {

        "Good":
            "Air quality is satisfactory.",

        "Moderate":
            "Air quality is acceptable.",

        "Unhealthy for Sensitive Groups":
            "Sensitive groups may experience health effects.",

        "Unhealthy":
            "Everyone may begin to experience health effects.",

        "Very Unhealthy":
            "Health alert: everyone may experience more serious effects.",

        "Hazardous":
            "Health emergency conditions. Everyone is likely to be affected.",
    }

    return messages.get(
        category,
        "Air quality information available.",
    )


# ============================================================
# SAFE CONVERSION
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# FORECAST TABLE
# ============================================================

def ensure_forecast_table():

    connection = get_connection()

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS forecast_predictions (
                id BIGINT,
                city_id INTEGER,
                city_name VARCHAR,
                origin_date DATE,
                forecast_date DATE,
                horizon INTEGER,
                predicted_aqi DOUBLE,
                actual_aqi DOUBLE,
                error DOUBLE,
                absolute_error DOUBLE,
                model_name VARCHAR,
                created_at TIMESTAMP
            )
        """)

    finally:

        connection.close()


# ============================================================
# UPDATE PREVIOUS FORECASTS WITH ACTUAL AQI
# ============================================================

def update_forecast_actuals():

    """
    Match previously generated forecasts against newly
    available actual AQI values.

    This MUST happen before generating today's forecast.

    Example:

        prediction:
            2026-08-12 -> 2026-08-13
            predicted = 145

        new actual:
            2026-08-13
            actual = 151

        database becomes:

            actual_aqi = 151
            error = -6
            absolute_error = 6
    """

    ensure_forecast_table()

    connection = get_connection()

    try:

        before = connection.execute("""
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE actual_aqi IS NOT NULL
        """).fetchone()[0]

        connection.execute("""
            UPDATE forecast_predictions AS f
            SET
                actual_aqi = a.aqi,
                error = f.predicted_aqi - a.aqi,
                absolute_error =
                    ABS(
                        f.predicted_aqi - a.aqi
                    )
            FROM aqi AS a
            WHERE
                f.city_id = a.city_id
                AND f.forecast_date = a.date
                AND f.actual_aqi IS NULL
                AND a.aqi IS NOT NULL
        """)

        after = connection.execute("""
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE actual_aqi IS NOT NULL
        """).fetchone()[0]

        newly_evaluated = after - before

        print()
        print("=" * 70)
        print("FORECAST ACTUALS UPDATE")
        print("=" * 70)

        print(
            f"Previously evaluated : {before}"
        )

        print(
            f"Newly evaluated      : {newly_evaluated}"
        )

        print(
            f"Total evaluated      : {after}"
        )

    finally:

        connection.close()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the best registered XGBoost model for recursive V9
    next-day forecasting.

    V9 is a recursive forecasting system, so the H1 model is
    intentionally applied repeatedly:

        current actual -> +1 day
        predicted +1   -> +2 day
        predicted +2   -> +3 day

    The model registry determines which registered H1 model
    currently has the lowest validation RMSE.
    """

    from src.models.model_registry import get_best_model

    best_model = get_best_model(1)

    if best_model is None:

        raise FileNotFoundError(
            "No registered production model is available "
            "for horizon 1."
        )

    if best_model["name"] != "xgboost":

        raise RuntimeError(
            "V9 requires the registered XGBoost H1 model, "
            f"but the registry selected: "
            f"{best_model['name']}"
        )

    model_path = Path(
        best_model["model_path"]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Registered model file not found:\n"
            f"{model_path}"
        )

    print()
    print("=" * 70)
    print("REGISTERED PRODUCTION MODEL")
    print("=" * 70)

    print(
        f"Model      : {best_model['name']}"
    )

    print(
        f"Horizon    : h{best_model['horizon']}"
    )

    print(
        f"Version    : v{best_model['version']}"
    )

    print(
        f"RMSE       : {best_model['rmse']:.4f}"
    )

    print(
        f"MAE        : {best_model['mae']:.4f}"
    )

    print(
        f"R2         : {best_model['r2']:.4f}"
    )

    print(
        f"Model path : {model_path}"
    )

    print("=" * 70)

    model = joblib.load(
        model_path
    )

    if not isinstance(
        model,
        xgb.XGBRegressor
    ):

        raise TypeError(
            "Registered production model is not "
            "an XGBRegressor."
    )

    return model

# ============================================================
# LOAD MODEL FEATURE NAMES
# ============================================================

def load_feature_names(model):

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is not None:

        return list(feature_names)

    booster = model.get_booster()

    booster_names = booster.feature_names

    if booster_names is not None:

        return list(booster_names)

    raise ValueError(
        "Could not determine production XGBoost feature names."
    )


# ============================================================
# LOAD RAW DATA FROM DUCKDB
# ============================================================

def load_features():

    print(
        "\nLoading latest AQI/weather data from DuckDB..."
    )

    connection = get_connection()

    try:

        query = """
            SELECT
                a.city_id,
                c.city_name,
                a.date,
                a.aqi,
                a.pm25,
                a.pm10,
                a.no2,
                a.so2,
                a.co,
                a.o3,
                w.temperature,
                w.humidity,
                w.precipitation,
                w.windspeed

            FROM aqi a

            JOIN cities c
                ON a.city_id = c.city_id

            LEFT JOIN weather w
                ON a.city_id = w.city_id
                AND a.date = w.date

            ORDER BY
                a.city_id,
                a.date
        """

        df = connection.execute(
            query
        ).fetchdf()

    finally:

        connection.close()

    if df.empty:

        raise RuntimeError(
            "No AQI data found in DuckDB."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    numeric_columns = [
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
        "temperature",
        "humidity",
        "precipitation",
        "windspeed",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        else:

            df[column] = np.nan

    df = (
        df
        .sort_values(
            [
                "city_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df = (
        df
        .drop_duplicates(
            subset=[
                "city_id",
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    required = [
        "date",
        "city_id",
        "city_name",
        "aqi",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "DuckDB data is missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    if df["aqi"].isna().all():

        raise ValueError(
            "AQI column contains no valid values."
        )

    print(
        f"Rows        : {len(df):,}"
    )

    print(
        f"Cities      : "
        f"{df['city_name'].nunique()}"
    )

    print(
        f"Date range  : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        "\nLatest AQI date by city:"
    )

    latest_by_city = (
        df
        .groupby("city_name")["date"]
        .max()
        .sort_index()
    )

    print(
        latest_by_city.to_string()
    )

    print(
        "\nRebuilding V4 inference features..."
    )

    df = rebuild_features(
        df
    )

    print(
        f"\nInference feature rows: "
        f"{len(df):,}"
    )

    print(
        f"Inference feature count: "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# REMOVE ENGINEERED FEATURES
# ============================================================

def remove_engineered_columns(df):

    engineered_columns = [

        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_7",
        "aqi_lag_14",

        "aqi_change_1d",
        "aqi_change_2d",
        "aqi_change_3d",
        "aqi_change_7d",

        "aqi_acceleration_1d",
        "aqi_acceleration_2d",

        "aqi_rolling_3",
        "aqi_std_3",
        "aqi_min_3",
        "aqi_max_3",
        "aqi_range_3",

        "aqi_rolling_7",
        "aqi_std_7",
        "aqi_min_7",
        "aqi_max_7",
        "aqi_range_7",

        "aqi_rolling_14",
        "aqi_std_14",
        "aqi_min_14",
        "aqi_max_14",
        "aqi_range_14",

        "aqi_trend_3d",
        "aqi_trend_7d",

        "aqi_distance_from_max_7",
        "aqi_distance_from_max_14",

        "aqi_percentile_7",
        "aqi_percentile_14",

        "pm25_change_1d",
        "pm25_change_3d",
        "pm25_rolling_3",
        "pm25_rolling_7",
        "pm25_trend_3d",
        "pm25_trend_7d",

        "pm10_change_1d",
        "pm10_change_3d",
        "pm10_rolling_3",
        "pm10_rolling_7",
        "pm10_trend_3d",
        "pm10_trend_7d",

        "no2_change_1d",
        "no2_change_3d",
        "no2_rolling_3",
        "no2_rolling_7",

        "so2_change_1d",
        "so2_change_3d",
        "so2_rolling_3",
        "so2_rolling_7",

        "co_change_1d",
        "co_change_3d",
        "co_rolling_3",
        "co_rolling_7",

        "o3_change_1d",
        "o3_change_3d",
        "o3_rolling_3",
        "o3_rolling_7",

        "pm25_pm10_ratio",
        "pm25_no2_interaction",
        "pm25_co_interaction",
        "pm25_o3_interaction",
        "pollution_sum",

        "temperature_change_1d",
        "humidity_change_1d",
        "windspeed_change_1d",
        "precipitation_change_1d",
        "precipitation_rolling_3",
        "windspeed_rolling_3",

        "aqi_moderate",
        "aqi_unhealthy",
        "aqi_very_unhealthy",
        "aqi_severe",
        "aqi_extreme",
        "aqi_regime_change",
        "high_aqi_recent",
        "extreme_aqi_recent",

        "city_aqi_mean",
        "city_aqi_std",
        "city_pm25_mean",
        "city_recent_mean",
        "city_recent_max",

        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    existing = [
        c
        for c in engineered_columns
        if c in df.columns
    ]

    if existing:

        df = df.drop(
            columns=existing
        )

    return df


# ============================================================
# REBUILD V4 FEATURES
# ============================================================

def rebuild_features(df):

    work = df.copy()

    if "target_aqi" in work.columns:

        work = work.drop(
            columns=["target_aqi"]
        )

    work = remove_engineered_columns(
        work
    )

    dates = pd.to_datetime(
        work["date"]
    )

    work["year"] = dates.dt.year
    work["month"] = dates.dt.month
    work["day"] = dates.dt.day
    work["day_of_week"] = dates.dt.dayofweek
    work["day_of_year"] = dates.dt.dayofyear

    work["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
    )

    work["is_weekend"] = (
        work["day_of_week"] >= 5
    ).astype(int)

    work = add_lag_features(work)
    work = add_aqi_rolling_features(work)
    work = add_aqi_trend_features(work)
    work = add_pollution_features(work)
    work = add_weather_features(work)
    work = add_regime_features(work)
    work = add_city_history_features(work)
    work = add_calendar_features(work)

    return work


# ============================================================
# CREATE FUTURE ROW

# ============================================================

def create_future_row(
    latest_row,
    future_date,
    seed_aqi,
    city_id,
    origin_date,
):
    """
    Create one future recursive forecast row.

    AQI is recursive. Future pollutants use the latest known
    state. Future weather comes from the DuckDB
    weather_forecasts table for the exact city/date.

    No future actual AQI is used during inference.
    """

    row = latest_row.copy()

    row["date"] = pd.to_datetime(future_date)

    # --------------------------------------------------------
    # Recursive AQI state
    # --------------------------------------------------------

    row["aqi"] = float(seed_aqi)

    # --------------------------------------------------------
    # Future pollutant state
    # --------------------------------------------------------

    pollutant_columns = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
    ]

    for column in pollutant_columns:
        if column in row.index:
            row[column] = safe_float(
                latest_row[column],
                0.0,
            )

    # --------------------------------------------------------
    # FUTURE WEATHER
    #
    # Use weather_forecasts instead of copying historical
    # weather from latest_row.
    # --------------------------------------------------------

    connection = get_connection()

    try:
        weather_result = connection.execute(
            """
            SELECT
                temperature,
                humidity,
                precipitation,
                windspeed
            FROM weather_forecasts
            WHERE
                city_id = ?
                AND date = ?
                AND forecast_origin <= ?
            ORDER BY forecast_origin DESC
            LIMIT 1
            """,
            [
                int(city_id),
                pd.to_datetime(future_date).date(),
                pd.to_datetime(origin_date).date(),
            ],
        ).fetchone()
    finally:
        connection.close()

    if weather_result is None:
        raise RuntimeError(
            "Missing future weather forecast for "
            f"city_id={city_id}, "
            f"date={pd.to_datetime(future_date).date()}. "
            "Run WeatherForecastService before V9 forecasting."
        )

    if "temperature" in row.index:
        row["temperature"] = safe_float(
            weather_result[0],
            0.0,
        )

    if "humidity" in row.index:
        row["humidity"] = safe_float(
            weather_result[1],
            0.0,
        )

    if "precipitation" in row.index:
        row["precipitation"] = safe_float(
            weather_result[2],
            0.0,
        )

    if "windspeed" in row.index:
        row["windspeed"] = safe_float(
            weather_result[3],
            0.0,
        )

    # --------------------------------------------------------
    # Never allow target AQI into inference
    # --------------------------------------------------------

    if "target_aqi" in row.index:
        row["target_aqi"] = np.nan

    return row


# ============================================================

# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    future_features,
    feature_names,
):

    if future_features.empty:

        raise ValueError(
            "No future feature row available."
        )

    row = (
        future_features
        .sort_values("date")
        .iloc[-1]
    )

    missing = [
        feature
        for feature in feature_names
        if feature not in future_features.columns
    ]

    if missing:

        raise ValueError(
            "Forecast feature mismatch. "
            f"Missing {len(missing)} features:\n"
            + "\n".join(
                f"- {x}"
                for x in missing
            )
        )

    values = []

    for feature in feature_names:

        value = row[feature]

        if pd.isna(value):

            value = 0.0

        values.append(
            safe_float(
                value,
                0.0,
            )
        )

    X = pd.DataFrame(
        [values],
        columns=feature_names,
    )

    X = X.replace(
        [np.inf, -np.inf],
        0.0,
    )

    return X


# ============================================================
# V9 REGIME
# ============================================================

def get_regime(aqi):

    aqi = safe_float(
        aqi
    )

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
# MOMENTUM
# ============================================================

def calculate_momentum(row):

    values = []

    columns = [
        "aqi_change_1d",
        "aqi_change_2d",
        "aqi_change_3d",
        "aqi_acceleration_1d",
    ]

    for column in columns:

        if column in row.index:

            values.append(
                safe_float(
                    row[column],
                    0.0,
                )
            )

    if not values:

        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# LOAD V9 CALIBRATION
# ============================================================

def load_calibration():

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(
            "V9 calibration file not found:\n"
            f"{CALIBRATION_FILE}"
        )

    with open(
        CALIBRATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    print(
        f"\nV9 calibration loaded:\n"
        f"{CALIBRATION_FILE}"
    )

    return data


# ============================================================
# EXTRACT PARAMETERS
# ============================================================

def extract_v9_parameters(data):

    if "selected_parameters" in data:

        selected = data[
            "selected_parameters"
        ]

    elif "best_parameters" in data:

        selected = data[
            "best_parameters"
        ]

    elif "parameters" in data:

        selected = data[
            "parameters"
        ]

    else:

        selected = data

    if not isinstance(
        selected,
        dict,
    ):

        selected = {}

    def get_float(
        key,
        default,
    ):

        value = selected.get(
            key,
            default,
        )

        try:

            value = float(
                value
            )

            if np.isfinite(value):

                return value

        except Exception:
            pass

        return default

    return {

        "shrinkage":
            get_float(
                "shrinkage",
                1.0,
            ),

        "city_weight":
            get_float(
                "city_weight",
                0.75,
            ),

        "horizon_weight":
            get_float(
                "horizon_weight",
                0.75,
            ),

        "regime_weight":
            get_float(
                "regime_weight",
                0.25,
            ),

        "momentum_weight":
            get_float(
                "momentum_weight",
                0.0,
            ),

        "spike_weight":
            get_float(
                "spike_weight",
                0.0,
            ),

        "max_correction":
            get_float(
                "max_correction",
                10.0,
            ),

        "minimum_benefit":
            get_float(
                "minimum_benefit",
                0.25,
            ),
    }


# ============================================================
# NORMALIZE CALIBRATION TABLES
# ============================================================

def extract_lookup_tables(data):

    tables = {}

    table_names = [
        "city_bias",
        "horizon_bias",
        "regime_bias",
        "city_horizon_bias",
        "city_regime_bias",
    ]

    for table_name in table_names:

        source = data.get(
            table_name
        )

        if source is None:

            for container_name in [
                "calibration_tables",
                "tables",
                "lookup_tables",
            ]:

                container = data.get(
                    container_name
                )

                if isinstance(
                    container,
                    dict,
                ):

                    source = container.get(
                        table_name
                    )

                    if source is not None:
                        break

        if source is None:
            continue

        normalized = {}

        if isinstance(
            source,
            dict,
        ):

            for key, value in source.items():

                if isinstance(
                    value,
                    dict,
                ):

                    bias = value.get(
                        "bias",
                        value.get(
                            "mean_error",
                            value.get(
                                "correction",
                                0.0,
                            ),
                        ),
                    )

                else:

                    bias = value

                normalized[
                    str(key)
                ] = safe_float(
                    bias,
                    0.0,
                )

        elif isinstance(
            source,
            list,
        ):

            for item in source:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                bias = item.get(
                    "bias",
                    item.get(
                        "mean_error",
                        item.get(
                            "correction",
                            0.0,
                        ),
                    ),
                )

                bias = safe_float(
                    bias,
                    0.0,
                )

                if table_name == "city_bias":

                    key = item.get(
                        "city",
                        item.get(
                            "city_name"
                        ),
                    )

                    if key is not None:
                        normalized[
                            str(key)
                        ] = bias

                elif table_name == "horizon_bias":

                    key = item.get(
                        "horizon"
                    )

                    if key is not None:
                        normalized[
                            str(key)
                        ] = bias

                elif table_name == "regime_bias":

                    key = item.get(
                        "regime"
                    )

                    if key is not None:
                        normalized[
                            str(key)
                        ] = bias

                elif table_name == "city_horizon_bias":

                    city = item.get(
                        "city",
                        item.get(
                            "city_name"
                        ),
                    )

                    horizon = item.get(
                        "horizon"
                    )

                    if (
                        city is not None
                        and horizon is not None
                    ):

                        normalized[
                            f"{city}|{horizon}"
                        ] = bias

                elif table_name == "city_regime_bias":

                    city = item.get(
                        "city",
                        item.get(
                            "city_name"
                        ),
                    )

                    regime = item.get(
                        "regime"
                    )

                    if (
                        city is not None
                        and regime is not None
                    ):

                        normalized[
                            f"{city}|{regime}"
                        ] = bias

        tables[
            table_name
        ] = normalized

    return tables


# ============================================================
# BIAS LOOKUP
# ============================================================

def lookup_bias(
    table,
    keys,
    default=0.0,
):

    if not table:
        return default

    if not isinstance(
        keys,
        (list, tuple),
    ):

        keys = [
            keys
        ]

    for key in keys:

        if key is None:
            continue

        key = str(
            key
        )

        if key in table:

            return safe_float(
                table[key],
                default,
            )

    return default


# ============================================================
# APPLY V9 CALIBRATION
# ============================================================

def apply_v9_calibration(
    base_prediction,
    city,
    horizon,
    current_aqi,
    row,
    parameters,
    tables,
):

    p = parameters

    city_bias = lookup_bias(
        tables.get("city_bias"),
        [city],
    )

    horizon_bias = lookup_bias(
        tables.get("horizon_bias"),
        [
            horizon,
            str(horizon),
            f"horizon_{horizon}",
        ],
    )

    regime = get_regime(
        current_aqi
    )

    regime_bias = lookup_bias(
        tables.get("regime_bias"),
        [regime],
    )

    city_horizon_bias = lookup_bias(
        tables.get("city_horizon_bias"),
        [
            f"{city}|{horizon}",
            f"{city}_{horizon}",
            f"{city}|{str(horizon)}",
            str(
                (
                    city,
                    horizon,
                )
            ),
        ],
    )

    city_regime_bias = lookup_bias(
        tables.get("city_regime_bias"),
        [
            f"{city}|{regime}",
            f"{city}_{regime}",
            str(
                (
                    city,
                    regime,
                )
            ),
        ],
    )

    momentum = calculate_momentum(
        row
    )

    correction = (

        p["city_weight"]
        * city_bias

        +

        p["horizon_weight"]
        * horizon_bias

        +

        p["regime_weight"]
        * regime_bias

        +

        p["city_weight"]
        * p["horizon_weight"]
        * city_horizon_bias

        +

        p["city_weight"]
        * p["regime_weight"]
        * city_regime_bias

        +

        p["momentum_weight"]
        * momentum
    )

    correction *= p["shrinkage"]

    correction = float(
        np.clip(
            correction,
            -p["max_correction"],
            p["max_correction"],
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

    return (
        calibrated,
        correction,
    )


# ============================================================
# RECURSIVE FORECAST FOR ONE CITY
# ============================================================

def forecast_city(
    city_df,
    model,
    feature_names,
    parameters,
    tables,
):

    city_df = (
        city_df
        .sort_values("date")
        .copy()
    )

    if city_df.empty:
        return []

    history = city_df.tail(
        FORECAST_HISTORY_DAYS
    ).copy()

    latest_date = pd.to_datetime(
        history["date"].iloc[-1]
    )

    latest_aqi = safe_float(
        history["aqi"].iloc[-1]
    )

    city_name = str(
        history["city_name"].iloc[-1]
    )

    city_id = int(
        safe_float(
            history["city_id"].iloc[-1]
        )
    )

    results = []

    # ============================================================
    # FORECAST DATE ARCHITECTURE
    # ============================================================
    #
    # latest_date:
    #     Last date with a real observed AQI.
    #
    # forecast_origin:
    #     Date on which the forecast is generated.
    #
    # Example:
    #
    #     latest actual = 2026-08-16
    #     forecast origin = 2026-08-17
    #
    #     24h -> 2026-08-17
    #     48h -> 2026-08-18
    #     72h -> 2026-08-19
    #
    # No hidden bridge day is required.
    # Every predicted day becomes part of the recursive state.
    # ============================================================
    
    forecast_origin = latest_date
    
    working = history.copy()
    
    previous_prediction = latest_aqi
    
    # ============================================================
    # RECURSIVE 24h / 48h / 72h FORECAST
    # ============================================================
    
    for internal_step in range(1, 4):
    
        forecast_date = (
            latest_date
            + timedelta(days=internal_step)
        )
    
        user_horizon = internal_step
    
        seed_aqi = previous_prediction
    
        latest_state = (
            working
            .sort_values("date")
            .iloc[-1]
        )
    
        future_row = create_future_row(
            latest_state,
            forecast_date,
            seed_aqi,
            city_id,
            forecast_origin,
        )
    
        # --------------------------------------------------------
        # Append recursive state
        # --------------------------------------------------------
    
        working = pd.concat(
            [
                working,
                pd.DataFrame([future_row]),
            ],
            ignore_index=True,
        )
    
        working = (
            working
            .sort_values("date")
            .reset_index(drop=True)
        )
    
        # --------------------------------------------------------
        # Rebuild features using the recursive state
        # --------------------------------------------------------
    
        rebuilt = rebuild_features(
            working
        )
    
        future_features = (
            rebuilt[
                rebuilt["date"]
                == forecast_date
            ]
            .copy()
        )
    
        if future_features.empty:
    
            raise RuntimeError(
                f"Could not create forecast features "
                f"for {city_name} {forecast_date}"
            )
    
        X = prepare_model_input(
            future_features,
            feature_names,
        )
    
        # --------------------------------------------------------
        # Base prediction
        # --------------------------------------------------------
    
        base_prediction = safe_float(
            model.predict(X)[0]
        )
    
        base_prediction = max(
            0.0,
            base_prediction,
        )
    
        # --------------------------------------------------------
        # V9 calibration
        #
        # internal_step is directly the user horizon:
        #
        # 1 -> 24h
        # 2 -> 48h
        # 3 -> 72h
        # --------------------------------------------------------
    
        calibrated_prediction, correction = (
            apply_v9_calibration(
                base_prediction=base_prediction,
                city=city_name,
                horizon=user_horizon,
                current_aqi=seed_aqi,
                row=future_features.iloc[-1],
                parameters=parameters,
                tables=tables,
            )
        )
    
        calibrated_prediction = max(
            0.0,
            calibrated_prediction,
        )
    
        # --------------------------------------------------------
        # Update recursive AQI state
        #
        # The next prediction MUST use this prediction.
        # --------------------------------------------------------
    
        working.loc[
            working["date"] == forecast_date,
            "aqi"
        ] = calibrated_prediction
    
        previous_prediction = calibrated_prediction
    
        # --------------------------------------------------------
        # Prepare user-facing result
        # --------------------------------------------------------
    
        change = (
            calibrated_prediction
            - seed_aqi
        )
    
        category = get_category(
            calibrated_prediction
        )
    
        results.append({
    
            "city_id":
                city_id,
    
            "city_name":
                city_name,
    
            "origin_date":
                forecast_origin.strftime(
                    "%Y-%m-%d"
                ),
    
            "forecast_date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),
    
            "horizon":
                user_horizon,
    
            "horizon_label":
                HORIZON_LABELS[user_horizon],
    
            "current_aqi":
                round(
                    latest_aqi,
                    2,
                ),
    
            "prediction":
                round(
                    calibrated_prediction,
                    2,
                ),
    
            "base_prediction":
                round(
                    base_prediction,
                    2,
                ),
    
            "correction":
                round(
                    correction,
                    2,
                ),
    
            "change":
                round(
                    change,
                    2,
                ),
    
            "category":
                category,
        })
    

    # Return only after all 3 recursive horizons have been generated.
    return results

    # Return only after all three recursive horizons
    # have been generated for this city.
    return results
# ============================================================
# GENERATE ALL FORECASTS
# ============================================================

def generate_forecasts(
    df,
    model,
    feature_names,
    parameters,
    tables,
):

    print(
        "\nGenerating recursive V9 forecasts..."
    )

    cities = (
        df[
            [
                "city_id",
                "city_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "city_id"
        )
    )

    print(
        f"Cities: {len(cities)}"
    )

    all_results = []

    for _, city in cities.iterrows():

        city_id = int(
            safe_float(
                city["city_id"]
            )
        )

        city_name = str(
            city["city_name"]
        )

        print()
        print(
            f"Forecasting: {city_name}"
        )

        city_df = df[
            df["city_id"]
            == city_id
        ].copy()

        results = forecast_city(
            city_df=city_df,
            model=model,
            feature_names=feature_names,
            parameters=parameters,
            tables=tables,
        )

        all_results.extend(
            results
        )

        for result in results:

            print(
                f"  "
                f"{result['horizon_label']} → "
                f"{result['prediction']:.2f} "
                f"({result['category']}) "
                f"[base "
                f"{result['base_prediction']:.2f}, "
                f"correction "
                f"{result['correction']:+.2f}]"
            )

    return pd.DataFrame(
        all_results
    )


# ============================================================
# VALIDATE FORECAST STRUCTURE
# ============================================================

def validate_forecast_output(
    predictions,
):

    if predictions.empty:

        raise RuntimeError(
            "Forecast output is empty."
        )

    expected_rows = (
        predictions["city_name"].nunique()
        * 3
    )

    if len(predictions) != expected_rows:

        raise RuntimeError(
            "Unexpected forecast count. "
            f"Expected {expected_rows}, "
            f"got {len(predictions)}."
        )

    for city in predictions[
        "city_name"
    ].unique():

        city_rows = predictions[
            predictions["city_name"]
            == city
        ]

        horizons = set(
            city_rows["horizon"]
            .astype(int)
            .tolist()
        )

        if horizons != {1, 2, 3}:

            raise RuntimeError(
                f"{city} does not contain "
                f"all 24h/48h/72h forecasts."
            )

    identical_cities = 0

    for city in predictions[
        "city_name"
    ].unique():

        values = (
            predictions[
                predictions["city_name"]
                == city
            ]
            .sort_values("horizon")[
                "prediction"
            ]
            .tolist()
        )

        if len(values) == 3:

            if (
                np.allclose(
                    values[0],
                    values[1],
                    atol=1e-8,
                )
                and
                np.allclose(
                    values[1],
                    values[2],
                    atol=1e-8,
                )
            ):

                identical_cities += 1

    if identical_cities == (
        predictions["city_name"].nunique()
    ):

        raise RuntimeError(
            "CRITICAL: all cities have identical "
            "24h/48h/72h forecasts. "
            "Recursive forecasting is not working."
        )

    print()
    print(
        "Forecast structure validation: PASSED"
    )

    print(
        f"Cities validated: "
        f"{predictions['city_name'].nunique()}"
    )

    print(
        f"Forecast rows: "
        f"{len(predictions)}"
    )


# ============================================================
# SAVE FORECASTS TO DUCKDB
# ============================================================

def save_forecasts_to_database(
    predictions
):

    if predictions.empty:

        raise ValueError(
            "Cannot save empty forecast DataFrame."
        )

    ensure_forecast_table()

    connection = get_connection()

    try:

        inserted = 0
        skipped = 0

        for _, row in predictions.iterrows():

            city_id = int(
                safe_float(
                    row["city_id"]
                )
            )

            city_name = str(
                row["city_name"]
            )

            origin_date = pd.to_datetime(
                row["origin_date"]
            ).date()

            forecast_date = pd.to_datetime(
                row["forecast_date"]
            ).date()

            horizon = int(
                row["horizon"]
            )

            predicted_aqi = safe_float(
                row["prediction"]
            )

            # ------------------------------------------------
            # Prevent duplicate forecasts
            # ------------------------------------------------

            exists = connection.execute(
                """
                SELECT COUNT(*)
                FROM forecast_predictions
                WHERE
                    city_id = ?
                    AND origin_date = ?
                    AND forecast_date = ?
                    AND horizon = ?
                """,
                [
                    city_id,
                    origin_date,
                    forecast_date,
                    horizon,
                ]
            ).fetchone()[0]

            if exists > 0:

                skipped += 1
                continue

            # ------------------------------------------------
            # If actual AQI already exists, evaluate now
            # ------------------------------------------------

            actual_result = connection.execute(
                """
                SELECT aqi
                FROM aqi
                WHERE
                    city_id = ?
                    AND date = ?
                    AND aqi IS NOT NULL
                LIMIT 1
                """,
                [
                    city_id,
                    forecast_date,
                ]
            ).fetchone()

            actual_aqi = None
            error = None
            absolute_error = None

            if actual_result is not None:

                actual_aqi = safe_float(
                    actual_result[0]
                )

                error = (
                    predicted_aqi
                    - actual_aqi
                )

                absolute_error = abs(
                    error
                )

            # ------------------------------------------------
            # Generate unique ID
            # ------------------------------------------------

            next_id = connection.execute(
                """
                SELECT COALESCE(
                    MAX(id),
                    0
                ) + 1
                FROM forecast_predictions
                """
            ).fetchone()[0]

            # ------------------------------------------------
            # Insert
            # ------------------------------------------------

            connection.execute(
                """
                INSERT INTO forecast_predictions (
                    id,
                    city_id,
                    city_name,
                    origin_date,
                    forecast_date,
                    horizon,
                    predicted_aqi,
                    actual_aqi,
                    error,
                    absolute_error,
                    model_name,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    int(next_id),
                    city_id,
                    city_name,
                    origin_date,
                    forecast_date,
                    horizon,
                    predicted_aqi,
                    actual_aqi,
                    error,
                    absolute_error,
                    MODEL_NAME,
                    datetime.now(),
                ]
            )

            inserted += 1

        print()
        print("=" * 70)
        print("FORECAST DATABASE SAVE")
        print("=" * 70)

        print(
            f"Inserted : {inserted}"
        )

        print(
            f"Skipped  : {skipped}"
        )

        total = connection.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            """
        ).fetchone()[0]

        pending = connection.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE actual_aqi IS NULL
            """
        ).fetchone()[0]

        evaluated = connection.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE actual_aqi IS NOT NULL
            """
        ).fetchone()[0]

        print(
            f"Total    : {total}"
        )

        print(
            f"Evaluated: {evaluated}"
        )

        print(
            f"Pending  : {pending}"
        )

    finally:

        connection.close()


# ============================================================
# SAVE CSV / PARQUET / METADATA
# ============================================================

def save_results(
    predictions,
    parameters,
    feature_count,
):

    # --------------------------------------------------------
    # Permanent DuckDB history
    # --------------------------------------------------------

    save_forecasts_to_database(
        predictions
    )

    # --------------------------------------------------------
    # Export files
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    predictions.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    metadata = {

        "project":
            "PearlsAQI",

        "engine":
            "forecast_engine_v9",

        "model":
            str(MODEL_FILE),

        "calibration":
            str(CALIBRATION_FILE),

        "feature_count":
            feature_count,

        "generated_at":
            datetime.now().isoformat(),

        "cities":
            int(
                predictions[
                    "city_name"
                ].nunique()
            ),

        "forecasts":
            int(
                len(predictions)
            ),

        "horizons":
            [
                "24h",
                "48h",
                "72h",
            ],

        "recursive":
            True,

        "target_leakage":
            False,

        "weather_strategy":
            "future Open-Meteo forecast from weather_forecasts table",

        "pollutant_strategy":
            "latest known state persistence",

        "calibration":
            "V9 production-safe offline calibration",

        "database_history":
            True,

        "calibration_parameters":
            parameters,
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            default=str,
        )

    print()
    print("=" * 70)
    print("V9 PRODUCTION FORECAST COMPLETE")
    print("=" * 70)

    print(
        f"Cities     : {metadata['cities']}"
    )

    print(
        f"Forecasts  : {metadata['forecasts']}"
    )

    print(
        "Horizons   : 24h / 48h / 72h"
    )

    print()
    print(
        f"CSV      : {OUTPUT_CSV}"
    )

    print(
        f"Parquet  : {OUTPUT_PARQUET}"
    )

    print(
        f"Metadata : {OUTPUT_METADATA}"
    )


# ============================================================
# PRINT DATABASE SUMMARY
# ============================================================

def print_forecast_history_summary():

    connection = get_connection()

    try:

        summary = connection.execute(
            """
            SELECT
                COUNT(*) AS total_forecasts,
                COUNT(actual_aqi) AS evaluated_forecasts,
                COUNT(*) - COUNT(actual_aqi)
                    AS pending_forecasts,
                MIN(origin_date)
                    AS first_origin_date,
                MAX(origin_date)
                    AS latest_origin_date,
                MIN(forecast_date)
                    AS first_forecast_date,
                MAX(forecast_date)
                    AS latest_forecast_date
            FROM forecast_predictions
            """
        ).fetchdf()

        print()
        print("=" * 70)
        print("FORECAST HISTORY")
        print("=" * 70)

        print(
            summary.to_string(
                index=False
            )
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PEARLSAQI PRODUCTION FORECAST ENGINE V9")
    print("=" * 70)

    print()
    print(
        "Recursive V4 XGBoost + V9 calibration"
    )

    print(
        "No future actual AQI is used during inference."
    )

    # --------------------------------------------------------
    # Make sure forecast history exists
    # --------------------------------------------------------

    ensure_forecast_table()

    # --------------------------------------------------------
    # Load latest AQI/weather
    # --------------------------------------------------------

    df = load_features()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Evaluate old forecasts BEFORE creating new ones.
    # --------------------------------------------------------

    update_forecast_actuals()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    feature_names = load_feature_names(
        model
    )

    print()
    print(
        f"Production model features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Verify model features exist
    # --------------------------------------------------------

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Production feature dataset is missing "
            f"{len(missing)} model features:\n"
            + "\n".join(
                f"- {x}"
                for x in missing
            )
        )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration = load_calibration()

    parameters = extract_v9_parameters(
        calibration
    )

    tables = extract_lookup_tables(
        calibration
    )

    print()
    print(
        "Selected V9 parameters:"
    )

    for key, value in parameters.items():

        print(
            f"{key:20s}: {value}"
        )

    print()
    print(
        "Calibration tables:"
    )

    if tables:

        for name, table in tables.items():

            print(
                f"{name:22s}: "
                f"{len(table)} entries"
            )

    else:

        print(
            "No explicit lookup tables found."
        )

        print(
            "Using V9 parameters with "
            "zero unavailable-table bias."
        )

    # --------------------------------------------------------
    # Generate forecasts
    # --------------------------------------------------------

    predictions = generate_forecasts(
        df=df,
        model=model,
        feature_names=feature_names,
        parameters=parameters,
        tables=tables,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_forecast_output(
        predictions
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL V9 FORECASTS")
    print("=" * 70)

    for city in (
        predictions[
            "city_name"
        ]
        .drop_duplicates()
        .tolist()
    ):

        city_rows = (
            predictions[
                predictions["city_name"]
                == city
            ]
            .sort_values("horizon")
        )

        print()
        print(city)

        for _, row in city_rows.iterrows():

            print(
                f"  "
                f"{row['horizon_label']} → "
                f"{row['prediction']:.2f} "
                f"({row['category']}) "
                f"[base "
                f"{row['base_prediction']:.2f}, "
                f"correction "
                f"{row['correction']:+.2f}]"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        predictions=predictions,
        parameters=parameters,
        feature_count=len(feature_names),
    )

    # --------------------------------------------------------
    # Final database status
    # --------------------------------------------------------

    print_forecast_history_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

