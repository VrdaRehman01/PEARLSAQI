
import os
import math
import threading
import time
import traceback
import json
from concurrent.futures import ThreadPoolExecutor

import duckdb
import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from flask import Flask, jsonify, request, send_from_directory
from src.config import CITIES
from src.alerts.aqi_alerts import get_alert
from src.models.model_registry import get_all_registered_models
from src.explainability.shap_explain import explain_prediction
from src.ml.forecast_engine_v9 import rebuild_features
import shap

from flask_cors import CORS


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173"
        ]
    }
})


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
    "xgboost_v4_final_108.pkl"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "models",
    "benchmark",
    "v4",
    "final_candidate",
    "feature_columns.json",
)
DATABASE_FILE = os.path.join(
    BASE_DIR,
    "database",
    "aqi.duckdb",
)

FRONTEND_DIST = os.path.join(
    BASE_DIR,
    "frontend",
    "dist",
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

    if _feature_columns is None and not os.path.exists(FEATURE_FILE):

        # feature_columns.json is git-ignored, so a fresh clone does
        # not have it. The fitted model stores the same ordered schema.
        model_features = getattr(_model, "feature_names_in_", None)

        if model_features is None:
            raise FileNotFoundError(
                f"Feature schema not found: {FEATURE_FILE}"
            )

        _feature_columns = list(model_features)

    if _feature_columns is None:

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

RAW_COLUMNS = [
    "city_id", "city_name", "date",
    "aqi", "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "precipitation", "windspeed",
]

# Same AQI + weather join that feature_builder_v4.build_features()
# uses to build the training data.
RAW_HISTORY_QUERY = """
    SELECT
        a.city_id, c.city_name, a.date,
        a.aqi, a.pm25, a.pm10, a.no2, a.so2, a.co, a.o3,
        w.temperature, w.humidity, w.precipitation, w.windspeed
    FROM aqi a
    INNER JOIN cities c ON a.city_id = c.city_id
    INNER JOIN weather w ON a.city_id = w.city_id AND a.date = w.date
    ORDER BY a.city_id, a.date
"""

_live_cache = {"key": None, "raw": None, "features": None}


def _database_key():
    return os.path.getmtime(DATABASE_FILE)


def load_raw_history():
    """
    Return the raw daily AQI/weather history (one row per city/date),
    rebuilt whenever database/aqi.duckdb changes.
    """

    _refresh_live_cache()

    return _live_cache["raw"]


def _refresh_live_cache():

    if not os.path.exists(DATABASE_FILE):

        if _live_cache["features"] is None:
            _live_cache["features"] = _read_live_parquet()
            _live_cache["raw"] = _live_cache["features"][RAW_COLUMNS].copy()

        return

    key = _database_key()

    if _live_cache["key"] == key:
        return

    try:
        with duckdb.connect(DATABASE_FILE, read_only=True) as conn:
            raw = conn.execute(RAW_HISTORY_QUERY).df()
    except duckdb.Error:
        # A pipeline may hold the write lock; keep serving the last
        # good snapshot instead of failing the request.
        if _live_cache["features"] is not None:
            return
        raise

    if raw.empty:
        raise RuntimeError("No AQI/weather rows found in the database.")

    raw["date"] = pd.to_datetime(raw["date"])

    raw = (
        raw
        .drop_duplicates(subset=["city_id", "date"], keep="last")
        .sort_values(["city_id", "date"])
        .reset_index(drop=True)
    )

    _live_cache["raw"] = raw
    _live_cache["features"] = rebuild_features(raw)
    _live_cache["key"] = key
    _forecast_cache.clear()


def _read_live_parquet():

    if not os.path.exists(LIVE_FILE):
        raise FileNotFoundError(
            f"Neither {DATABASE_FILE} nor {LIVE_FILE} exists."
        )

    df = pd.read_parquet(LIVE_FILE)
    df["date"] = pd.to_datetime(df["date"])

    return df


def load_live_features():
    """
    Return V4 serving features for every city/date.

    Features are built straight from the AQI/weather tables in
    database/aqi.duckdb (kept current by the hourly pipeline) with the
    same V4 feature code used for training. The old
    data/feature_store/live_features.parquet is only a fallback, since
    no pipeline refreshes it any more.
    """

    _refresh_live_cache()

    df = _live_cache["features"]

    if df.empty:
        raise RuntimeError(
            "Live feature store is empty."
        )

    if "city_name" not in df.columns:
        raise RuntimeError(
            "Live features have no city_name column."
        )

    # This is a mandatory feature for the final candidate.
    if "pm25_pollution_ratio" not in df.columns:
        raise RuntimeError(
            "Live features are missing pm25_pollution_ratio."
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

    X = build_model_input(row, feature_columns)

    prediction = float(
        np.asarray(
            model.predict(X)
        ).reshape(-1)[0]
    )

    return prediction, row, X


def build_model_input(row, feature_columns):
    """
    Build the exact 108-feature V4 input for one feature row.
    """

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

    return X


_shap_explainer = None


def get_shap_explainer():

    global _shap_explainer

    if _shap_explainer is None:

        model, _ = load_v4_model()

        try:
            _shap_explainer = shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent",
            )
        except Exception as explainer_error:
            print("FORECAST SHAP EXPLAINER ERROR:", explainer_error)
            return None

    return _shap_explainer


# ============================================================
# RECURSIVE 3-DAY FORECAST
# ============================================================

FORECAST_HORIZONS = 3

# Longest V4 window is rolling(30) on a shift(1), so 60 days of
# history reproduces every windowed feature exactly.
FORECAST_HISTORY_DAYS = 60

WEATHER_COLUMNS = ["temperature", "humidity", "precipitation", "windspeed"]

_forecast_cache = {}


def _future_weather(city_id, date):
    """
    Weather for a future recursive step: observed weather if the
    database already has it, otherwise the latest stored forecast.
    Returns (values, source) or (None, None).
    """

    if not os.path.exists(DATABASE_FILE):
        return None, None

    day = pd.Timestamp(date).date()

    try:
        with duckdb.connect(DATABASE_FILE, read_only=True) as conn:

            observed = conn.execute(
                """
                SELECT temperature, humidity, precipitation, windspeed
                FROM weather
                WHERE city_id = ? AND date = ?
                LIMIT 1
                """,
                [int(city_id), day],
            ).fetchone()

            if observed is not None:
                return dict(zip(WEATHER_COLUMNS, observed)), "observed"

            forecast = conn.execute(
                """
                SELECT temperature, humidity, precipitation, windspeed
                FROM weather_forecasts
                WHERE city_id = ? AND date = ?
                ORDER BY forecast_origin DESC, created_at DESC
                LIMIT 1
                """,
                [int(city_id), day],
            ).fetchone()

    except duckdb.Error:
        return None, None

    if forecast is not None:
        return dict(zip(WEATHER_COLUMNS, forecast)), "forecast"

    return None, None


def recursive_forecast(city):
    """
    Forecast the next three days by feeding each predicted AQI back
    in as the latest observation (same recursion as
    src/ml/forecast_engine_v9.py, using the served V4 model).

    The V4 model predicts the next day's AQI from a day's features, so:

        features(D)                       -> AQI(D+1)   day 1
        features(D+1 with aqi=day 1)      -> AQI(D+2)   day 2
        features(D+2 with aqi=day 2)      -> AQI(D+3)   day 3

    Pollutants are held at the latest observed values; weather comes
    from the database (observed, then forecast), else is held too.

    Returns (steps, row, X) where row/X are the day-1 serving inputs.
    """

    model, feature_columns = load_v4_model()

    day1_aqi, row, X = predict_v4(city)

    cache_key = (str(row["city_name"]).lower(), row["date"], _live_cache["key"])

    if cache_key not in _forecast_cache:
        _forecast_cache[cache_key] = roll_forward(row, day1_aqi, X)

    return _forecast_cache[cache_key], row, X


def roll_forward(row, day1_aqi, X, overrides=None):
    """
    Turn a day-1 prediction into the full recursive horizon list.

    overrides: raw columns (e.g. pm25, windspeed) to hold at fixed
    values across every step, for what-if scenarios.
    """

    model, feature_columns = load_v4_model()

    overrides = overrides or {}

    raw = load_raw_history()

    history = raw[raw["city_id"] == row["city_id"]].sort_values("date").copy()

    for column, value in overrides.items():
        history.loc[history.index[-1], column] = value

    full_aqi = history["aqi"].tolist()
    full_pm25 = history["pm25"].tolist()

    working = history.tail(FORECAST_HISTORY_DAYS).copy()

    steps = [{
        "forecast_date": row["date"] + pd.Timedelta(days=1),
        "predicted_aqi": max(0.0, day1_aqi),
        "X": X,
        "weather_source": "observed",
    }]

    for _ in range(1, FORECAST_HORIZONS):

        previous = steps[-1]

        future = working.iloc[-1].copy()
        future["date"] = previous["forecast_date"]
        future["aqi"] = previous["predicted_aqi"]

        weather, weather_source = _future_weather(
            future["city_id"], future["date"]
        )

        if weather is None:
            weather_source = "held"
        else:
            for column, value in weather.items():
                if value is not None and column not in overrides:
                    future[column] = value

        working = pd.concat(
            [working, future.to_frame().T],
            ignore_index=True,
        )

        for column in RAW_COLUMNS:
            if column not in ("city_name", "date"):
                working[column] = pd.to_numeric(working[column])

        features = rebuild_features(working).iloc[-1].copy()

        # Expanding city statistics need the full history, not just
        # the recent window used for the rebuild.
        features["city_aqi_mean"] = float(np.nanmean(full_aqi))
        features["city_aqi_std"] = float(pd.Series(full_aqi).std())
        features["city_pm25_mean"] = float(np.nanmean(full_pm25))

        step_X = build_model_input(features, feature_columns)

        prediction = float(
            np.asarray(model.predict(step_X)).reshape(-1)[0]
        )

        steps.append({
            "forecast_date": future["date"] + pd.Timedelta(days=1),
            "predicted_aqi": max(0.0, prediction),
            "X": step_X,
            "weather_source": weather_source,
        })

        full_aqi.append(future["aqi"])
        full_pm25.append(future["pm25"])

    return steps


# ============================================================
# LIVE CONDITIONS (current hour)
# ============================================================
#
# The model works on daily averages, but "current AQI" on the dashboard
# should be the latest hour, like official monitors and Google show.
#
# Two sources:
#   1. Open-Meteo current values (CAMS model estimate, no key needed).
#   2. Optional ground-station readings from the World Air Quality Index
#      project (aqicn.org) when AQICN_TOKEN is set in .env. Stations
#      capture local smog the ~40 km CAMS grid smooths out.

load_dotenv(os.path.join(BASE_DIR, ".env"))

OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WAQI_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/"

LIVE_TTL_SECONDS = 600
HOURLY_TTL_SECONDS = 1800

# A station farther than this from the city centre does not describe
# that city (the WAQI demo token, for example, always returns Shanghai).
MAX_STATION_DISTANCE_KM = 40

_live_lock = threading.Lock()
_live_state = {"at": 0.0, "data": {}}
_hourly_cache = {}
_city_locations = {}


def city_locations():
    """city_name -> (city_id, latitude, longitude), from the database."""

    if not _city_locations:
        with duckdb.connect(DATABASE_FILE, read_only=True) as conn:
            for city_id, name, lat, lon in conn.execute(
                "SELECT city_id, city_name, latitude, longitude "
                "FROM cities ORDER BY city_id"
            ).fetchall():
                _city_locations[name] = (city_id, lat, lon)

    return _city_locations


def _distance_km(lat1, lon1, lat2, lon2):

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2

    return 6371 * 2 * math.asin(math.sqrt(a))


def _fetch_open_meteo_current(locations):
    """One request per API for all cities (Open-Meteo accepts lists)."""

    names = list(locations)
    lats = ",".join(str(locations[n][1]) for n in names)
    lons = ",".join(str(locations[n][2]) for n in names)

    air = requests.get(
        OPEN_METEO_AIR_URL,
        params={
            "latitude": lats,
            "longitude": lons,
            "current": "us_aqi,pm2_5,pm10,nitrogen_dioxide,"
                       "sulphur_dioxide,carbon_monoxide,ozone",
            "timezone": "Asia/Karachi",
        },
        timeout=15,
    )
    air.raise_for_status()

    weather = requests.get(
        OPEN_METEO_WEATHER_URL,
        params={
            "latitude": lats,
            "longitude": lons,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "Asia/Karachi",
        },
        timeout=15,
    )
    weather.raise_for_status()

    air_rows = air.json()
    weather_rows = weather.json()

    # A single location comes back as an object, several as a list.
    if isinstance(air_rows, dict):
        air_rows = [air_rows]
    if isinstance(weather_rows, dict):
        weather_rows = [weather_rows]

    result = {}

    for name, air_row, weather_row in zip(names, air_rows, weather_rows):
        current = air_row.get("current", {})
        current_weather = weather_row.get("current", {})

        result[name] = {
            "aqi": current.get("us_aqi"),
            "time": current.get("time"),
            "source": "open-meteo",
            "pollutants": {
                "pm25": current.get("pm2_5"),
                "pm10": current.get("pm10"),
                "no2": current.get("nitrogen_dioxide"),
                "so2": current.get("sulphur_dioxide"),
                "co": current.get("carbon_monoxide"),
                "o3": current.get("ozone"),
            },
            "weather": {
                "temperature": current_weather.get("temperature_2m"),
                "humidity": current_weather.get("relative_humidity_2m"),
                "wind_speed": current_weather.get("wind_speed_10m"),
            },
        }

    return result


def _latest_hour_from_database(locations):
    """Fallback when Open-Meteo is unreachable: newest stored hour."""

    result = {}

    with duckdb.connect(DATABASE_FILE, read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT a.city_name, a.timestamp, a.aqi,
                   a.pm25, a.pm10, a.no2, a.so2, a.co, a.o3
            FROM aqi_hourly a
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY a.city_id ORDER BY a.timestamp DESC
            ) = 1
            """
        ).fetchall()

    for name, ts, aqi, pm25, pm10, no2, so2, co, o3 in rows:

        if name not in locations:
            continue

        result[name] = {
            "aqi": aqi,
            "time": ts.isoformat(timespec="minutes") if ts else None,
            "source": "database",
            "pollutants": {
                "pm25": pm25, "pm10": pm10, "no2": no2,
                "so2": so2, "co": co, "o3": o3,
            },
            "weather": {},
        }

    return result


def _fetch_station(token, lat, lon):

    try:
        response = requests.get(
            WAQI_URL.format(lat=lat, lon=lon),
            params={"token": token},
            timeout=10,
        )
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    if payload.get("status") != "ok":
        return None

    data = payload.get("data") or {}
    city = data.get("city") or {}
    aqi = data.get("aqi")
    geo = city.get("geo") or []

    if not isinstance(aqi, (int, float)) or len(geo) != 2:
        return None

    distance = _distance_km(lat, lon, float(geo[0]), float(geo[1]))

    if distance > MAX_STATION_DISTANCE_KM:
        return None

    return {
        "aqi": float(aqi),
        "station": city.get("name"),
        "url": city.get("url"),
        "time": (data.get("time") or {}).get("s"),
        "distance_km": round(distance, 1),
        "dominant_pollutant": data.get("dominentpol"),
    }


def live_conditions():
    """
    Current-hour conditions for every city, refreshed at most every
    LIVE_TTL_SECONDS. Never raises: an empty dict means "no live data".
    """

    with _live_lock:

        if time.time() - _live_state["at"] < LIVE_TTL_SECONDS:
            return _live_state["data"]

        locations = city_locations()

        try:
            data = _fetch_open_meteo_current(locations)
        except (requests.RequestException, ValueError, KeyError) as error:
            print("LIVE OPEN-METEO ERROR:", error)
            try:
                data = _latest_hour_from_database(locations)
            except duckdb.Error as db_error:
                print("LIVE DATABASE FALLBACK ERROR:", db_error)
                data = {}

        token = os.environ.get("AQICN_TOKEN", "").strip()

        if token:
            with ThreadPoolExecutor(max_workers=6) as pool:
                stations = dict(zip(
                    locations,
                    pool.map(
                        lambda loc: _fetch_station(token, loc[1], loc[2]),
                        locations.values(),
                    ),
                ))

            for name, station in stations.items():
                if station:
                    data.setdefault(name, {})["station"] = station

        _live_state.update(at=time.time(), data=data)

        return data


def live_for_city(city):

    for name, values in live_conditions().items():
        if name.lower() == city.strip().lower():
            return values

    return {}


def hourly_outlook(city):
    """
    Hourly US AQI for the last 12 hours and the next 36 hours from
    Open-Meteo (CAMS forecast), for the "next hours" chart.
    """

    location = next(
        (value for name, value in city_locations().items()
         if name.lower() == city.strip().lower()),
        None,
    )

    if location is None:
        raise ValueError(f"Unknown city: {city}")

    cached = _hourly_cache.get(location[0])

    if cached and time.time() - cached[0] < HOURLY_TTL_SECONDS:
        return cached[1]

    response = requests.get(
        OPEN_METEO_AIR_URL,
        params={
            "latitude": location[1],
            "longitude": location[2],
            "hourly": "us_aqi,pm2_5",
            "past_days": 1,
            "forecast_days": 3,
            "timezone": "Asia/Karachi",
        },
        timeout=15,
    )
    response.raise_for_status()

    hourly = response.json().get("hourly", {})

    now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).floor("h")
    start, end = now - pd.Timedelta(hours=12), now + pd.Timedelta(hours=36)

    points = [
        {
            "time": stamp,
            "aqi": aqi,
            "pm25": pm25,
            "is_forecast": pd.Timestamp(stamp) > now,
        }
        for stamp, aqi, pm25 in zip(
            hourly.get("time", []),
            hourly.get("us_aqi", []),
            hourly.get("pm2_5", []),
        )
        if aqi is not None and start <= pd.Timestamp(stamp) <= end
    ]

    _hourly_cache[location[0]] = (time.time(), points)

    return points


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


def pick(*values):
    """First value that is a real number."""

    for value in values:
        value = native(value)
        if isinstance(value, (int, float)) and not math.isnan(value):
            return value

    return None


def current_conditions(row, live):
    """
    Headline conditions: a ground-station reading when one is available,
    else the current-hour model estimate, else today's daily average.
    """

    station = live.get("station") or {}
    weather = live.get("weather") or {}

    return {
        "current_aqi": pick(station.get("aqi"), live.get("aqi"), row.get("aqi")),
        "temperature": pick(weather.get("temperature"), row.get("temperature")),
        "humidity": pick(weather.get("humidity"), row.get("humidity")),
        "wind_speed": pick(weather.get("wind_speed"), row.get("windspeed")),
    }


def live_summary(row, live):
    """Where the headline AQI came from, for the dashboard to label it."""

    station = live.get("station")

    if station:
        source, observed_at = "station", station.get("time")
    elif pick(live.get("aqi")) is not None:
        source, observed_at = "model_hourly", live.get("time")
    else:
        source, observed_at = "model_daily", str(row.get("date", ""))[:10]

    return {
        "source": source,
        "observed_at": observed_at,
        "station": station,
        "model_hourly_aqi": pick(live.get("aqi")),
        "model_hourly_time": live.get("time"),
        "daily_average_aqi": pick(row.get("aqi")),
        "station_data_enabled": bool(os.environ.get("AQICN_TOKEN", "").strip()),
    }


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
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(
        os.path.join(FRONTEND_DIST, "assets"),
        filename,
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
@app.route("/api/predict/<city>")
def predict(city):

    try:

        steps, row, X = recursive_forecast(city)

        # ============================================================
        # SHAP DRIVERS FOR EACH RECURSIVE HORIZON
        # ============================================================
        model, feature_columns = load_v4_model()

        forecast_explainer = get_shap_explainer()

        def forecast_shap(input_X):
            try:
                if forecast_explainer is None:
                    return []
                values = np.asarray(forecast_explainer.shap_values(input_X))
                if values.ndim == 2:
                    values = values[0]

                ranked = sorted(
                    zip(feature_columns, values),
                    key=lambda x: abs(float(x[1])),
                    reverse=True,
                )
                return [
                    {
                        "feature": feature,
                        "impact": round(float(impact), 3),
                    }
                    for feature, impact in ranked[:108]
                ]
            except Exception as shap_error:
                print("FORECAST SHAP ERROR:", shap_error)
                return []

        # Each horizon is explained with the exact input it was
        # predicted from.
        for step in steps:
            step["explanation"] = forecast_shap(step["X"])

        shap_explanation = steps[0]["explanation"]

        live = live_for_city(city)

        return jsonify({

            "city": city,

            "as_of": str(
                row.get("date", "")
            ),

            "current_conditions": current_conditions(row, live),

            "pollutants": live.get("pollutants") or pollutant_data(row),

            "live": live_summary(row, live),

            # Exact raw values the model used (daily averages); the
            # scenario simulator starts from these so an untouched run
            # reproduces the baseline.
            "model_inputs": {
                **pollutant_data(row),
                "temperature": native(row.get("temperature")),
                "humidity": native(row.get("humidity")),
                "wind_speed": native(row.get("windspeed")),
            },

            "shap_explanation": shap_explanation,

            "forecast": [
                {
                    "horizon": horizon,
                    "forecast_date": step["forecast_date"].strftime("%Y-%m-%d"),
                    "predicted_aqi": round(step["predicted_aqi"], 1),
                    "model_used": "XGBoost V4 Final 108",
                    "method": "recursive" if horizon > 1 else "direct",
                    "weather_source": step["weather_source"],
                    "alert": get_alert(step["predicted_aqi"]),
                    "explanation": step["explanation"],
                }
                for horizon, step in enumerate(steps, start=1)
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

@app.route("/api/hourly/<city>")
def hourly(city):

    try:
        points = hourly_outlook(city)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except requests.RequestException as e:
        return jsonify({"error": f"Hourly forecast unavailable: {e}"}), 503

    return jsonify({
        "city": city,
        "source": "Open-Meteo (CAMS) hourly US AQI",
        "hours": points,
    }), 200


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
# CUSTOM SCENARIO SIMULATOR — NEW ONLY
# ============================================================

@app.route("/api/scenario-predict", methods=["POST"])
def scenario_predict():
    """Run a what-if scenario using the existing V4 model."""
    try:
        payload = request.get_json(silent=True) or {}
        city = str(payload.get("city", "")).strip()
        scenario = payload.get("scenario") or {}

        if not city:
            return jsonify({"success": False, "error": "City is required."}), 400

        if not isinstance(scenario, dict):
            return jsonify({"success": False, "error": "Scenario must be an object."}), 400

        model, feature_columns = load_v4_model()
        row = get_city_row(city)

        # Exact same 108-feature contract/order as normal V4 inference.
        baseline_values = {}
        for feature in feature_columns:
            value = row.get(feature, 0)
            try:
                if pd.isna(value):
                    value = 0
            except Exception:
                pass
            baseline_values[feature] = value

        # Only override raw features that actually exist in the V4 schema.
        aliases = {
            "pm25": ["pm25"],
            "pm10": ["pm10"],
            "o3": ["o3"],
            "no2": ["no2"],
            "so2": ["so2"],
            "co": ["co"],
            "temperature": ["temperature"],
            "humidity": ["humidity"],
            "wind_speed": ["windspeed", "wind_speed"],
        }

        changed_features = []

        for field, aliases_for_field in aliases.items():
            if field not in scenario:
                continue

            try:
                new_value = float(scenario[field])
            except (TypeError, ValueError):
                continue

            feature_name = next(
                (name for name in aliases_for_field if name in feature_columns),
                None,
            )

            if feature_name is None:
                continue

            old_value = baseline_values.get(feature_name, 0)
            baseline_values[feature_name] = new_value

            # The simulator sends every input; only report real edits
            # (smallest slider step is 0.1).
            if abs(new_value - float(old_value or 0)) < 0.05:
                continue

            changed_features.append({
                "input": field,
                "feature": feature_name,
                "before": float(old_value) if old_value is not None else 0.0,
                "after": new_value,
            })

        X_baseline = pd.DataFrame(
            [[baseline_values[feature] for feature in feature_columns]],
            columns=feature_columns,
        )
        X_scenario = pd.DataFrame(
            [[baseline_values[feature] for feature in feature_columns]],
            columns=feature_columns,
        )

        # Rebuild true baseline separately from the modified scenario vector.
        X_baseline = pd.DataFrame(
            [[
                0 if pd.isna(row.get(feature, 0)) else row.get(feature, 0)
                for feature in feature_columns
            ]],
            columns=feature_columns,
        )

        X_baseline = X_baseline.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scenario = X_scenario.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        if X_baseline.shape != (1, 108) or X_scenario.shape != (1, 108):
            raise RuntimeError(
                f"Invalid V4 scenario input shape: baseline={X_baseline.shape}, scenario={X_scenario.shape}; expected (1, 108)."
            )

        baseline_aqi = max(0.0, float(np.asarray(model.predict(X_baseline)).reshape(-1)[0]))
        scenario_aqi = max(0.0, float(np.asarray(model.predict(X_scenario)).reshape(-1)[0]))
        # SHAP explanation for the scenario prediction
        shap_explanation = explain_prediction(
            model,
            X_scenario,
            X_baseline,
            top_n=len(feature_columns),
        )

        explanation = []
        if shap_explanation:
            for feature, impact in shap_explanation:
                explanation.append({
                    "feature": feature,
                    "impact": round(float(impact), 3),
                })

        # 3-day scenario forecast: the changed conditions are held
        # for all three recursive steps.
        scenario_steps = roll_forward(
            row,
            scenario_aqi,
            X_scenario,
            overrides={
                change["feature"]: change["after"]
                for change in changed_features
            },
        )

        scenario_forecast = [
            {
                "horizon": horizon,
                "forecast_date": step["forecast_date"].strftime("%Y-%m-%d"),
                "predicted_aqi": round(step["predicted_aqi"], 1),
                "model_used": "XGBoost V4 Final 108",
                "alert": get_alert(step["predicted_aqi"]),
            }
            for horizon, step in enumerate(scenario_steps, start=1)
        ]

        delta = scenario_aqi - baseline_aqi
        percentage = (delta / baseline_aqi * 100.0) if baseline_aqi > 0 else 0.0

        def category(aqi):
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

        if delta > 5:
            direction = "Worsened"
        elif delta < -5:
            direction = "Improved"
        else:
            direction = "Minimal Change"

        return jsonify({
            "success": True,
            "city": city,
            "baseline": {
                "aqi": round(baseline_aqi, 1),
                "category": category(baseline_aqi),
            },
            "scenario": {
                "aqi": round(scenario_aqi, 1),
                "category": category(scenario_aqi),
            },
            "impact": {
                "delta": round(delta, 1),
                "percentage": round(percentage, 1),
                "direction": direction,
            },
            "changed_features": changed_features,
            "forecast": scenario_forecast,
            "explanation": explanation,
            "model": {
                "name": "XGBoost V4 Final 108",
                "feature_count": len(feature_columns),
            },
        }), 200

    except Exception as e:
        print("SCENARIO PREDICTION ERROR")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/<path:path>")
def frontend_routes(path):
    requested_file = os.path.join(FRONTEND_DIST, path)

    if os.path.isfile(requested_file):
        return send_from_directory(FRONTEND_DIST, path)

    return send_from_directory(FRONTEND_DIST, "index.html")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PEARLSAQI V4 FLASK API")
    print("=" * 70)
    print("Model    : XGBoost V4 Final 108")
    print("Features : 108")
    print(f"Server   : http://127.0.0.1:{os.environ.get('PORT', 5000)}")
    print("=" * 70)

    try:

        # Startup validation
        model, feature_columns = load_v4_model()
        live = load_live_features()

        print()
        print("Startup validation successful.")
        print(f"Model features : {len(feature_columns)}")
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
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False,
    use_reloader=False,
)

    except Exception as e:

        print()
        print("=" * 70)
        print("FLASK STARTUP ERROR")
        print("=" * 70)
        print(repr(e))
        traceback.print_exc()
        print("=" * 70)