from pathlib import Path
from datetime import datetime
from typing import Optional

import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AQI Predictor API",
    description="AI-powered next-day AQI prediction API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

LATEST_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "predictions"
    / "latest_predictions.csv"
)

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

METRICS_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "final_metrics.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_latest_predictions():

    if not LATEST_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Latest predictions file not found."
        )

    df = pd.read_csv(LATEST_FILE)

    return df


def load_features():

    if not FEATURE_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Feature dataset not found."
        )

    return pd.read_parquet(FEATURE_FILE)


def get_category(aqi):

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
            "Health emergency conditions. Everyone is likely to be affected."
    }

    return messages.get(
        category,
        "Air quality information available."
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "AQI Predictor API",
        "model": "XGBoost",
        "model_type": "next-day AQI regression"
    }


# ============================================================
# ALL CURRENT PREDICTIONS
# ============================================================

@app.get("/predictions")
def predictions():

    df = load_latest_predictions()

    results = []

    for _, row in df.iterrows():

        prediction = float(row["prediction"])

        category = get_category(prediction)

        results.append({
            "city_id": int(row["city_id"]),
            "city_name": row["city_name"],
            "date": str(row["date"]),
            "prediction_date": str(row["prediction_date"]),
            "aqi": float(row["aqi"]),
            "prediction": prediction,
            "aqi_category": category,
            "health_message": get_health_message(category)
        })

    return {
        "updated_at": datetime.now().isoformat(),
        "cities": results
    }

# ============================================================
# AVAILABLE CITIES
# ============================================================

@app.get("/cities")
def cities():

    df = load_latest_predictions()

    city_list = []

    for _, row in df.iterrows():

        prediction = float(row["prediction"])
        category = get_category(prediction)

        city_list.append({
            "city_id": int(row["city_id"]),
            "city_name": row["city_name"],
            "current_aqi": float(row["aqi"]),
            "predicted_aqi": prediction,
            "category": category
        })

    return {
        "count": len(city_list),
        "cities": city_list,
        "updated_at": datetime.now().isoformat()
    }



# ============================================================
# SINGLE CITY
# ============================================================

@app.get("/prediction/{city_name}")
def prediction(city_name: str):

    df = load_latest_predictions()

    match = df[
        df["city_name"].str.lower() == city_name.lower()
    ]

    if match.empty:

        raise HTTPException(
            status_code=404,
            detail=f"City '{city_name}' not found."
        )

    row = match.iloc[0]

    prediction_value = float(row["prediction"])

    category = get_category(prediction_value)

    return {
        "city_id": int(row["city_id"]),
        "city_name": row["city_name"],
        "date": str(row["date"]),
        "prediction_date": str(row["prediction_date"]),
        "aqi": float(row["aqi"]),
        "prediction": prediction_value,
        "aqi_category": category,
        "health_message": get_health_message(category)
    }


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.get("/stats")
def stats():

    df = load_latest_predictions()

    predictions = df["prediction"].astype(float)

    average_aqi = float(predictions.mean())

    highest = df.loc[
        df["prediction"].idxmax()
    ]

    lowest = df.loc[
        df["prediction"].idxmin()
    ]

    categories = [
        get_category(value)
        for value in predictions
    ]

    category_counts = pd.Series(categories).value_counts()

    return {

        "average_aqi": round(average_aqi, 2),

        "highest_city": {
            "city": highest["city_name"],
            "aqi": round(float(highest["prediction"]), 2)
        },

        "lowest_city": {
            "city": lowest["city_name"],
            "aqi": round(float(lowest["prediction"]), 2)
        },

        "cities_monitored": len(df),

        "unhealthy_cities": int(
            sum(
                value > 150
                for value in predictions
            )
        ),

        "very_unhealthy_cities": int(
            sum(
                value > 200
                for value in predictions
            )
        ),

        "hazardous_cities": int(
            sum(
                value > 300
                for value in predictions
            )
        ),

        "category_distribution": {
            str(key): int(value)
            for key, value in category_counts.items()
        },

        "updated_at": datetime.now().isoformat()
    }


# ============================================================
# CITY HISTORY
# ============================================================

@app.get("/history/{city_name}")
def city_history(
    city_name: str,
    days: int = 30
):

    df = load_features()

    if "city_name" not in df.columns:

        raise HTTPException(
            status_code=500,
            detail="city_name column not found in feature dataset."
        )

    match = df[
        df["city_name"].str.lower()
        == city_name.lower()
    ].copy()

    if match.empty:

        raise HTTPException(
            status_code=404,
            detail=f"City '{city_name}' not found."
        )

    match["date"] = pd.to_datetime(match["date"])

    match = match.sort_values("date").tail(days)

    results = []

    for _, row in match.iterrows():

        aqi = float(row["aqi"])

        results.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "aqi": aqi,
            "category": get_category(aqi)
        })

    return {
        "city": city_name,
        "days": len(results),
        "history": results
    }


# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/model")
def model_metrics():

    if not METRICS_FILE.exists():

        return {
            "model": "XGBoost",
            "features": 107,
            "mae": 13.1605,
            "rmse": 18.1385,
            "r2": 0.8546
        }

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metrics = json.load(f)

    return metrics



# ============================================================
# PEARLSAQI ANALYTICS
# ============================================================

ANALYTICS_DIR = (
    ROOT / "analytics"
)


def load_analytics_csv(filename: str):
    path = ANALYTICS_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Analytics dataset not found: {filename}"
        )

    return pd.read_csv(path)


@app.get("/analytics/cities")
def analytics_cities():

    df = load_analytics_csv(
        "city_statistics.csv"
    )

    return {
        "updated_at": datetime.now().isoformat(),
        "cities": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/monthly")
def analytics_monthly():

    df = load_analytics_csv(
        "monthly_aqi.csv"
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/pollutants")
def analytics_pollutants():

    df = load_analytics_csv(
        "pollutant_statistics.csv"
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/categories")
def analytics_categories():

    df = load_analytics_csv(
        "aqi_categories.csv"
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/comparison")
def analytics_comparison():

    df = load_analytics_csv(
        "city_comparison.csv"
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/predictions")
def analytics_predictions():

    df = load_analytics_csv(
        "prediction_analysis.csv"
    )

    # Convert dates to strings for JSON
    if "date" in df.columns:
        df["date"] = (
            pd.to_datetime(
                df["date"],
                errors="coerce"
            )
            .dt.strftime("%Y-%m-%d")
        )

    # Replace NaN with None
    df = df.where(
        pd.notnull(df),
        None
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/model")
def analytics_model():

    df = load_analytics_csv(
        "model_performance.csv"
    )

    df = df.where(
        pd.notnull(df),
        None
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/summary")
def analytics_summary():

    summary_path = (
        ANALYTICS_DIR
        / "dashboard_summary.json"
    )

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard summary not found."
        )

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as f:

        summary = json.load(f)

    return summary