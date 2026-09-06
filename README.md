@'
# PEARLSAQI — Pakistan AQI Prediction & Forecasting System

## Overview

PEARLSAQI is an end-to-end Air Quality Index (AQI) prediction and forecasting system developed for 12 major cities in Pakistan.

The system combines historical air-quality data, weather observations, forecast weather data, feature engineering, machine learning, multi-horizon forecasting, explainable AI, automated data pipelines, monitoring, and a web-based dashboard.

The current architecture is designed around a local DuckDB database and Parquet feature store, with XGBoost as the primary production model family.

---

## Supported Cities

PEARLSAQI currently covers:

1. Karachi
2. Lahore
3. Islamabad
4. Peshawar
5. Rawalpindi
6. Faisalabad
7. Multan
8. Quetta
9. Hyderabad
10. Gujranwala
11. Sialkot
12. Bahawalpur

---

## Key Features

- Historical AQI data collection
- Weather and air-quality data integration
- Open-Meteo data source
- Local DuckDB database
- Parquet-based feature store
- 100+ engineered predictive features
- XGBoost machine learning
- Model evaluation and versioning
- 24-hour, 48-hour and 72-hour AQI forecasting
- Recursive multi-step forecasting
- Future weather integration
- SHAP explainability
- LIME explainability
- Forecast error monitoring
- AQI alerts
- Flask REST API
- React + Vite frontend
- Pakistan city map visualization
- AQI classification and visualization
- Automated GitHub Actions workflows
- Docker deployment support

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │      Open-Meteo      │
                         │ Air Quality + Weather│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Data Ingestion     │
                         │      Pipelines       │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐             ┌──────────────────┐
          │     DuckDB       │             │ Parquet Feature  │
          │   aqi.duckdb     │             │      Store       │
          └────────┬─────────┘             └────────┬─────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         │ Lag / Rolling /      │
                         │ Trend / Weather etc. │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     XGBoost Model    │
                         │ Training & Evaluation│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Forecast Engine      │
                         │       24 / 48 / 72h  │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                ┌─────────────────┐    ┌─────────────────┐
                │ Flask REST API  │    │ SHAP / LIME     │
                └────────┬────────┘    │ Explainability  │
                         │             └─────────────────┘
                         ▼
                ┌─────────────────────┐
                │ React + Vite        │
                │ AQI Dashboard       │
                └─────────────────────┘