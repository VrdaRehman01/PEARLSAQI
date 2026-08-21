# AQI Predictor

End-to-end machine learning pipeline forecasting Air Quality Index (AQI)
1, 2, and 3 days ahead, using Hopsworks as the feature store and model
registry, Open-Meteo for weather (no API key needed), and AQICN for
pollutant data.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with:
- `AQICN_API_KEY` — from https://aqicn.org/data-platform/token/
- `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` — from your Hopsworks account, Project Settings → Api Keys

Open-Meteo (both live and historical weather) needs no key at all.

## Project structure

```
src/
  config.py                    # cities for live ingestion
  config_locations.py          # lat/lon per city
  ingestion/                   # API clients + live data collection
  validation/                  # historical data cleaning
  preprocessing/               # merging historical AQI + weather
  features/                    # feature engineering + Hopsworks feature store
  pipelines/                   # orchestration wrappers (called by main.py)
  models/                      # data loading, Random Forest, Ridge, XGBoost,
                                # TensorFlow, and the Hopsworks model registry
  alerts/                      # AQI severity thresholds and messages
  explainability/               # SHAP-based prediction explanations

templates/index.html           # website page shell
static/css/style.css           # light-mode UI styling
static/js/app.js               # frontend logic, calls the Flask API

main.py                        # runs live ingestion + feature pipeline
download_historical_data.py    # one-off: pulls historical AQI CSVs from HuggingFace
download_weather_history.py    # one-off: pulls matching historical weather
backfill.py                    # merges both, adds 3-day targets, pushes to Hopsworks
train_models.py                # trains + registers RF/Ridge/XGBoost/DeepLearning x 3 horizons
app.py                         # Flask website + prediction API
notebooks/eda.ipynb            # exploratory data analysis
```

## Run order (first time)

1. `python download_historical_data.py` — pulls historical pollution data (4 years, hourly, ~34k rows/city)
2. `python download_weather_history.py` — pulls matching historical weather from Open-Meteo
3. `python backfill.py` — merges both, adds `aqi_h1`/`aqi_h2`/`aqi_h3` targets, pushes to the Hopsworks `aqi_historical_features` feature group
4. `python train_models.py` — trains Random Forest, Ridge, XGBoost, and a TensorFlow model for **each** of the 3 forecast horizons (12 models total), registering each in the Hopsworks Model Registry with its RMSE/MAE/R²
5. `python main.py` — runs one live ingestion + feature pipeline cycle, pushing a row into the Hopsworks `aqi_live_features` feature group
6. `python app.py` — starts the website at **http://localhost:5000**

## How the 3-day forecast works

Three separate targets are trained per model type: `aqi_h1`, `aqi_h2`,
`aqi_h3` (AQI value 1/2/3 days ahead), rather than one model recursively
predicting further out (which compounds errors). At serving time,
`app.py` asks the Hopsworks Model Registry for whichever registered
model has the lowest RMSE per horizon — so day+1/+2/+3 might each come
from a different model type.

## Feature store & model registry

Both are real Hopsworks, not local files:
- **Live features**: `aqi_live_features` feature group (hourly, keyed by city + timestamp)
- **Historical features**: `aqi_historical_features` feature group (daily, keyed by city + date, with the 3 forecast targets attached)
- **Models**: registered under `{model_type}_h{horizon}` (e.g. `random_forest_h1`) with metrics attached; `get_best_model(horizon)` compares all 4 types per horizon and downloads the winner

Without `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT` set, every Hopsworks call
fails with a clear error message rather than crashing silently.

## Automation

Two GitHub Actions workflows under `.github/workflows/` handle scheduling:
- `feature_pipeline.yml` — runs `main.py` every hour
- `training_pipeline.yml` — runs `train_models.py` daily

Because Hopsworks persists everything externally, these workflows don't
need to commit anything back to the repo. Add `AQICN_API_KEY`,
`HOPSWORKS_API_KEY`, and `HOPSWORKS_PROJECT` as repository secrets
(Settings → Secrets and variables → Actions) before enabling them.

## The website

Open `http://localhost:5000`. Shows a city picker, current conditions
(temperature, humidity, wind, current AQI, weather icon), three forecast
cards (day +1/+2/+3) with color-coded severity badges, a trend chart, and
a SHAP explanation panel for the day+1 prediction. Light color palette —
see `static/css/style.css`, CSS custom properties at the top control all
colors in one place.

`dashboard.py` (Streamlit) is included as an optional alternative
interface but isn't needed to run the main website.

## Explainability and alerts

- `src/explainability/shap_explain.py` — SHAP explanation for the t+1 prediction
- `src/alerts/aqi_alerts.py` — standard AQI severity thresholds, shared by the API and dashboard

## Known caveats to verify yourself

- `src/preprocessing/merge_historical_data.py`'s `POLLUTANT_CANDIDATES`
  assumes specific column names from the HuggingFace dataset — verified
  working against 4 years of hourly data per city as of this build
- The TensorFlow model in `train_models.py` takes noticeably longer to
  train than the sklearn models — expect several minutes per horizon
- First run of any Hopsworks call may take longer while the feature
  group / model registry entries are created

## New in this version

- **Open-Meteo for everything** — pollutant data now comes from Open-Meteo's Air Quality API (`src/ingestion/aqi_api.py`) instead of AQICN. No API key is required anywhere in the entire pipeline anymore, only Hopsworks credentials.
- **WHO Air Quality Guidelines comparison** (`src/alerts/who_standards.py`) — every prediction response includes each pollutant compared against WHO's 2021 24-hour guideline limits, shown as a ratio (e.g. "3.7x WHO guideline").
- **Pollutant composition chart** — doughnut chart of PM2.5/PM10/NO2/SO2/O3/CO breakdown for the selected city, rendered from the live feature store reading.
- **Historical trends** (`/api/trends/<city>`) — average AQI by month and by day-of-week, computed live from the Hopsworks historical feature group.
- **Model Stats & Logs** (`/api/model-stats`) — every registered model across all horizons, with RMSE/MAE/R², shown directly on the site.
- **What-If Predictor** (`/api/manual-predict`) — simulate a forecast from hypothetical weather conditions instead of live city data.

## Phase 1 of the extended plan: 12 cities + lag/rolling/calendar features

- **12 cities** now configured (`src/config.py`, `src/config_locations.py`) for live data collection. Only the original 4 (`HISTORICAL_DATA_CITIES`) have verified 4-year historical data and get trained per-city models today — the other 8 start accumulating their own real history from their first `main.py` run onward. No fabricated historical data.
- **Lag features**: `aqi_lag_1`, `aqi_lag_3`, `aqi_lag_7` (AQI N days ago per city)
- **Rolling features**: 3-day and 7-day rolling mean + std of AQI (properly shifted to avoid lookahead leakage — verified against hand-calculated values)
- **Calendar features**: `is_weekend`, `season` (adjusted for Pakistan's actual AQI-relevant seasons: winter/smog, spring, monsoon/summer, autumn)
- At serving time, lag/rolling features are computed from each city's recent live history (`read_recent_daily_aqi` in `feature_store.py`); a city with little or no history yet falls back to its current AQI reading rather than crashing.
- The What-If predictor gained one new field, **"Recent AQI baseline"**, standing in for a week of hypothetical history.

**Still to build** (see conversation): FastAPI + multi-page Streamlit migration, LSTM, compare-before-promote model deployment logic, drift monitoring.
