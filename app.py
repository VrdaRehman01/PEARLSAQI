import json
import os
import traceback

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request

from src.config import CITIES
from src.alerts.aqi_alerts import get_alert
from src.models.model_registry import get_all_registered_models


# ============================================================
# APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LIVE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "feature_store",
    "live_features.parquet",
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "benchmark",
    "v4",
    "final_candidate",
    "xgboost_v4_final_108.pkl",
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "models",
    "benchmark",
    "v4",
    "final_candidate",
    "feature_columns.json",
)


# ============================================================
# V4 MODEL CACHE
# ============================================================

_model = None
_feature_columns = None


# ============================================================
# V4 MODEL
# ============================================================

def load_v4_model():
    """
    Load the verified V4 Final 108 XGBoost model and
    authoritative feature schema.
    """

    global _model, _feature_columns

    if _model is None:

        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"V4 model not found: {MODEL_FILE}"
            )

        _model = joblib.load(MODEL_FILE)

    if _feature_columns is None:

        if not os.path.exists(FEATURE_FILE):
            raise FileNotFoundError(
                f"Feature schema not found: {FEATURE_FILE}"
            )

        with open(FEATURE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # The authoritative schema should be a list.
        if isinstance(data, list):
            _feature_columns = data

        elif isinstance(data, dict):

            if "feature_columns" in data:
                _feature_columns = data["feature_columns"]

            elif "features" in data:
                _feature_columns = data["features"]

            elif "columns" in data:
                _feature_columns = data["columns"]

            else:
                raise RuntimeError(
                    "feature_columns.json does not contain "
                    "feature_columns/features/columns."
                )

        else:
            raise RuntimeError(
                "Unsupported feature_columns.json format."
            )

        _feature_columns = list(_feature_columns)

    if len(_feature_columns) != 108:
        raise RuntimeError(
            f"V4 feature contract requires 108 features, "
            f"but loaded {len(_feature_columns)}."
        )

    return _model, _feature_columns


# ============================================================
# LIVE FEATURE STORE
# ============================================================

def load_live_features():
    """
    Load the real V4 live feature store.
    """

    if not os.path.exists(LIVE_FILE):
        raise FileNotFoundError(
            f"Live feature store not found: {LIVE_FILE}"
        )

    df = pd.read_parquet(LIVE_FILE)

    if df.empty:
        raise RuntimeError(
            "Live feature store is empty."
        )

    if "city_name" not in df.columns:
        raise RuntimeError(
            "live_features.parquet has no city_name column."
        )

    # This is a mandatory feature for the final candidate.
    if "pm25_pollution_ratio" not in df.columns:
        raise RuntimeError(
            "live_features.parquet is missing "
            "pm25_pollution_ratio."
        )

    return df


# ============================================================
# CITY ROW
# ============================================================

def get_city_row(city):
    """
    Return the latest live feature row for the requested city.
    """

    df = load_live_features()

    city_clean = city.strip().lower()

    city_df = df[
        df["city_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        == city_clean
    ].copy()

    if city_df.empty:
        available_cities = sorted(
            df["city_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"No live feature data found for {city}. "
            f"Available cities: {available_cities}"
        )

    if "date" in city_df.columns:

        city_df["date"] = pd.to_datetime(
            city_df["date"],
            errors="coerce"
        )

        city_df = city_df.sort_values(
            "date",
            na_position="first"
        )

    return city_df.iloc[-1]


# ============================================================
# V4 PREDICTION
# ============================================================

def predict_v4(city):
    """
    Run the verified XGBoost V4 Final 108 model.

    IMPORTANT:
    The feature_columns.json file is the authoritative
    feature order. Never manually reorder these features.
    """

    model, feature_columns = load_v4_model()

    row = get_city_row(city)

    # --------------------------------------------------------
    # Verify every required feature exists
    # --------------------------------------------------------

    missing = [
        feature
        for feature in feature_columns
        if feature not in row.index
    ]

    if missing:

        raise RuntimeError(
            "V4 model input is missing features: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Build exactly the authoritative 108-feature input
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[row[feature] for feature in feature_columns]],
        columns=feature_columns,
    )

    # --------------------------------------------------------
    # Numerical cleanup
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # HARD CONTRACT CHECK
    # --------------------------------------------------------

    if X.shape != (1, 108):

        raise RuntimeError(
            f"Invalid V4 model input shape: {X.shape}. "
            f"Expected exactly (1, 108)."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = float(
        np.asarray(
            model.predict(X)
        ).reshape(-1)[0]
    )

    return prediction, row, X


# ============================================================
# HELPERS
# ============================================================

def native(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):

        try:
            return value.item()
        except Exception:
            pass

    return value


def pollutant_data(row):

    return {
        "pm25": native(row.get("pm25")),
        "pm10": native(row.get("pm10")),
        "no2": native(row.get("no2")),
        "so2": native(row.get("so2")),
        "o3": native(row.get("o3")),
        "co": native(row.get("co")),
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        cities=CITIES,
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    try:

        model, feature_columns = load_v4_model()
        live = load_live_features()

        return jsonify({
            "status": "ok",
            "model": "XGBoost V4 Final 108",
            "features": len(feature_columns),
            "model_file": "xgboost_v4_final_108.pkl",
            "live_feature_store": "loaded",
            "live_rows": int(len(live)),
            "live_columns": int(len(live.columns)),
            "pm25_pollution_ratio": (
                "available"
                if "pm25_pollution_ratio" in live.columns
                else "missing"
            ),
        }), 200

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e),
        }), 500


# ============================================================
# MAIN PREDICTION
# ============================================================

@app.route("/predict/<city>")
def predict(city):

    try:

        prediction, row, X = predict_v4(city)

        return jsonify({

            "city": city,

            "as_of": str(
                row.get("date", "")
            ),

            "current_conditions": {

                "temperature": native(
                    row.get("temperature")
                ),

                "humidity": native(
                    row.get("humidity")
                ),

                "wind_speed": native(
                    row.get("windspeed")
                ),

                "current_aqi": native(
                    row.get("aqi")
                ),
            },

            "pollutants": pollutant_data(row),

            "forecast": [

                {
                    "horizon": 1,

                    "predicted_aqi": round(
                        prediction,
                        1
                    ),

                    "model_used":
                        "XGBoost V4 Final 108",

                    "alert":
                        get_alert(prediction),
                }

            ],

            "model": {

                "name":
                    "XGBoost",

                "version":
                    "v4_final_108",

                "feature_count":
                    108,

                "test_rmse":
                    13.367105,

                "test_mae":
                    8.801337,

                "test_r2":
                    0.913309,
            },

            # Useful verification information.
            # The frontend can ignore this.
            "inference": {

                "input_shape":
                    list(X.shape),

                "feature_count":
                    int(X.shape[1]),

                "winning_feature":
                    "pm25_pollution_ratio",
            },

        }), 200

    except Exception as e:

        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HISTORICAL TRENDS
# ============================================================

@app.route("/api/trends/<city>")
def trends(city):

    try:

        df = load_live_features()

        city_clean = city.strip().lower()

        city_df = df[
            df["city_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == city_clean
        ].copy()

        if city_df.empty:

            raise ValueError(
                f"No historical feature data found for {city}."
            )

        if "date" in city_df.columns:

            city_df["date"] = pd.to_datetime(
                city_df["date"],
                errors="coerce"
            )

            city_df = city_df.sort_values(
                "date"
            )

        trends = []

        for _, row in city_df.iterrows():

            trends.append({

                "date": (
                    row["date"].strftime("%Y-%m-%d")
                    if "date" in row.index
                    and pd.notna(row["date"])
                    else None
                ),

                "aqi": native(
                    row.get("aqi")
                ),

                "pm25": native(
                    row.get("pm25")
                ),

                "pm10": native(
                    row.get("pm10")
                ),

                "no2": native(
                    row.get("no2")
                ),

                "so2": native(
                    row.get("so2")
                ),

                "o3": native(
                    row.get("o3")
                ),

                "co": native(
                    row.get("co")
                ),

            })

        return jsonify({

            "city": city,

            "trends": trends,

        }), 200

    except Exception as e:

        print("=" * 70)
        print("TRENDS ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# MODEL STATISTICS / MODEL REGISTRY
# ============================================================

@app.route("/api/model-stats")
def model_stats():

    try:

        models = get_all_registered_models()

        return jsonify({

            "models": models

        }), 200

    except Exception as e:

        print("=" * 70)
        print("MODEL STATS ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# WHAT-IF / MANUAL PREDICTION
# ============================================================

@app.route(
    "/api/manual-predict",
    methods=["POST"]
)
def manual_predict():

    """
    What-If prediction intentionally disabled.

    The V4 Final 108 model requires the complete engineered
    108-feature input schema.

    Weather/pollutant values supplied by a frontend form
    cannot honestly be treated as a complete V4 feature vector
    until the proper inference feature builder exists.
    """

    return jsonify({

        "error": (
            "What-if prediction is temporarily disabled "
            "while the V4 108-feature inference pipeline "
            "is being integrated."
        )

    }), 501


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PEARLSAQI V4 FLASK API")
    print("=" * 70)
    print("Model    : XGBoost V4 Final 108")
    print("Features : 108")
    print("Server   : http://127.0.0.1:5000")
    print("=" * 70)

    try:

        # Startup validation
        model, feature_columns = load_v4_model()
        live = load_live_features()

        print()
        print("Startup validation successful.")
        print(
            f"Model features : {len(feature_columns)}"
        )
        print(
            f"Live data      : "
            f"{len(live)} rows x {len(live.columns)} columns"
        )
        print(
            "pm25_pollution_ratio : "
            + (
                "OK"
                if "pm25_pollution_ratio" in live.columns
                else "MISSING"
            )
        )
        print()
        print("Starting Flask...")
        print("=" * 70)

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=True,
        )

    except Exception:

        print()
        print("=" * 70)
        print("FATAL STARTUP ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)