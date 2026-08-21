"""
PEARLSAQI ANALYTICS ENGINE

Generates dashboard-ready datasets from the real AQI project data.

Outputs:
    analytics/
        city_statistics.csv
        daily_aqi.csv
        monthly_aqi.csv
        pollutant_statistics.csv
        aqi_categories.csv
        city_comparison.csv
        prediction_analysis.csv
        model_performance.csv
        dashboard_dataset.parquet
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.parquet"

TEST_CANDIDATES = [
    ROOT / "data" / "analysis" / "test_error_analysis.parquet",
    ROOT / "data" / "analysis" / "v3" / "test_predictions.parquet",
]

PRODUCTION_PREDICTIONS = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "predictions"
    / "latest_predictions.csv"
)

CITY_METRICS = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "city_metrics.csv"
)

ANALYTICS_DIR = ROOT / "analytics"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def find_test_file():
    for path in TEST_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find test dataset. Checked:\n"
        + "\n".join(str(p) for p in TEST_CANDIDATES)
    )


def get_category(aqi):
    if pd.isna(aqi):
        return "Unknown"

    aqi = float(aqi)

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def save_csv(df, filename):
    path = ANALYTICS_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


# ============================================================
# LOAD DATA
# ============================================================

print_header("PEARLSAQI ANALYTICS ENGINE")

print("\nProject root:")
print(ROOT)

print("\nLoading training data...")
train = pd.read_parquet(TRAIN_PATH)

print(f"Training rows: {len(train):,}")

print("\nLoading validation data...")
validation = pd.read_parquet(VALIDATION_PATH)

print(f"Validation rows: {len(validation):,}")

test_path = find_test_file()

print("\nLoading test data...")
test = pd.read_parquet(test_path)

print(f"Test rows: {len(test):,}")
print(f"Test file: {test_path}")


# ============================================================
# COMBINE HISTORICAL DATA
# ============================================================

print_header("BUILDING HISTORICAL DATASET")

frames = [
    train.copy(),
    validation.copy(),
]

historical = pd.concat(
    frames,
    ignore_index=True,
)

# Remove accidental duplicate city/date rows.
if "city_name" in historical.columns and "date" in historical.columns:
    historical["date"] = pd.to_datetime(
        historical["date"],
        errors="coerce",
    )

    historical = historical.drop_duplicates(
        subset=["city_name", "date"],
        keep="last",
    )

print(f"Historical rows: {len(historical):,}")

print(
    f"Date range: "
    f"{historical['date'].min()} → "
    f"{historical['date'].max()}"
)

print(
    f"Cities: "
    f"{historical['city_name'].nunique()}"
)

print(
    "\nCities:"
)

for city in sorted(historical["city_name"].dropna().unique()):
    print(f"  - {city}")


# ============================================================
# ADD AQI CATEGORIES
# ============================================================

historical["aqi_category"] = historical["aqi"].apply(
    get_category
)


# ============================================================
# DAILY AQI DATASET
# ============================================================

print_header("1. DAILY AQI DATASET")

daily_columns = [
    column
    for column in [
        "city_id",
        "city_name",
        "date",
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
        "aqi_category",
    ]
    if column in historical.columns
]

daily_aqi = historical[daily_columns].copy()

daily_aqi = daily_aqi.sort_values(
    ["city_name", "date"]
)

save_csv(
    daily_aqi,
    "daily_aqi.csv",
)


# ============================================================
# CITY STATISTICS
# ============================================================

print_header("2. CITY STATISTICS")

grouped = historical.groupby(
    "city_name",
    dropna=False,
)

city_statistics = grouped["aqi"].agg(
    rows="count",
    average_aqi="mean",
    median_aqi="median",
    minimum_aqi="min",
    maximum_aqi="max",
    std_aqi="std",
).reset_index()

city_statistics["average_aqi"] = city_statistics[
    "average_aqi"
].round(2)

city_statistics["median_aqi"] = city_statistics[
    "median_aqi"
].round(2)

city_statistics["minimum_aqi"] = city_statistics[
    "minimum_aqi"
].round(2)

city_statistics["maximum_aqi"] = city_statistics[
    "maximum_aqi"
].round(2)

city_statistics["std_aqi"] = city_statistics[
    "std_aqi"
].round(2)

city_statistics["unhealthy_days"] = grouped.apply(
    lambda x: int((x["aqi"] > 150).sum()),
    include_groups=False,
).values

city_statistics["very_unhealthy_days"] = grouped.apply(
    lambda x: int((x["aqi"] > 200).sum()),
    include_groups=False,
).values

city_statistics["hazardous_days"] = grouped.apply(
    lambda x: int((x["aqi"] > 300).sum()),
    include_groups=False,
).values

city_statistics = city_statistics.sort_values(
    "average_aqi",
    ascending=False,
)

city_statistics["rank"] = range(
    1,
    len(city_statistics) + 1,
)

save_csv(
    city_statistics,
    "city_statistics.csv",
)


# ============================================================
# MONTHLY AQI
# ============================================================

print_header("3. MONTHLY AQI")

monthly = historical.copy()

monthly["year"] = monthly["date"].dt.year
monthly["month"] = monthly["date"].dt.month
monthly["month_name"] = monthly["date"].dt.month_name()

monthly_aqi = (
    monthly
    .groupby(
        [
            "city_name",
            "year",
            "month",
            "month_name",
        ],
        as_index=False,
    )
    .agg(
        average_aqi=("aqi", "mean"),
        minimum_aqi=("aqi", "min"),
        maximum_aqi=("aqi", "max"),
        days=("aqi", "count"),
    )
)

monthly_aqi[
    [
        "average_aqi",
        "minimum_aqi",
        "maximum_aqi",
    ]
] = monthly_aqi[
    [
        "average_aqi",
        "minimum_aqi",
        "maximum_aqi",
    ]
].round(2)

monthly_aqi = monthly_aqi.sort_values(
    [
        "city_name",
        "year",
        "month",
    ]
)

save_csv(
    monthly_aqi,
    "monthly_aqi.csv",
)


# ============================================================
# POLLUTANT STATISTICS
# ============================================================

print_header("4. POLLUTANT STATISTICS")

pollutants = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]

available_pollutants = [
    p
    for p in pollutants
    if p in historical.columns
]

pollutant_rows = []

for pollutant in available_pollutants:

    series = pd.to_numeric(
        historical[pollutant],
        errors="coerce",
    )

    pollutant_rows.append(
        {
            "pollutant": pollutant,
            "average": series.mean(),
            "median": series.median(),
            "minimum": series.min(),
            "maximum": series.max(),
            "std": series.std(),
        }
    )

pollutant_statistics = pd.DataFrame(
    pollutant_rows
)

if not pollutant_statistics.empty:
    numeric_columns = [
        "average",
        "median",
        "minimum",
        "maximum",
        "std",
    ]

    pollutant_statistics[numeric_columns] = (
        pollutant_statistics[numeric_columns].round(2)
    )

save_csv(
    pollutant_statistics,
    "pollutant_statistics.csv",
)


# ============================================================
# AQI CATEGORY DISTRIBUTION
# ============================================================

print_header("5. AQI CATEGORY DISTRIBUTION")

category_counts = (
    historical["aqi_category"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="days")
)

category_counts["percentage"] = (
    category_counts["days"]
    / category_counts["days"].sum()
    * 100
).round(2)

save_csv(
    category_counts,
    "aqi_categories.csv",
)


# ============================================================
# CITY COMPARISON
# ============================================================

print_header("6. CITY COMPARISON")

city_comparison = (
    historical
    .groupby("city_name", as_index=False)
    .agg(
        average_aqi=("aqi", "mean"),
        max_aqi=("aqi", "max"),
        min_aqi=("aqi", "min"),
        days_recorded=("aqi", "count"),
    )
)

city_comparison["unhealthy_percentage"] = (
    historical.assign(
        unhealthy=historical["aqi"] > 150
    )
    .groupby("city_name")["unhealthy"]
    .mean()
    .reindex(city_comparison["city_name"])
    .values
    * 100
)

city_comparison["very_unhealthy_percentage"] = (
    historical.assign(
        very_unhealthy=historical["aqi"] > 200
    )
    .groupby("city_name")["very_unhealthy"]
    .mean()
    .reindex(city_comparison["city_name"])
    .values
    * 100
)

city_comparison["hazardous_percentage"] = (
    historical.assign(
        hazardous=historical["aqi"] > 300
    )
    .groupby("city_name")["hazardous"]
    .mean()
    .reindex(city_comparison["city_name"])
    .values
    * 100
)

numeric_columns = [
    "average_aqi",
    "max_aqi",
    "min_aqi",
    "unhealthy_percentage",
    "very_unhealthy_percentage",
    "hazardous_percentage",
]

city_comparison[numeric_columns] = (
    city_comparison[numeric_columns].round(2)
)

city_comparison = city_comparison.sort_values(
    "average_aqi",
    ascending=False,
)

save_csv(
    city_comparison,
    "city_comparison.csv",
)


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

print_header("7. PREDICTION ANALYSIS")

prediction_columns = [
    column
    for column in [
        "city_name",
        "date",
        "aqi",
        "target_aqi",
        "prediction",
        "absolute_error",
        "error",
    ]
    if column in test.columns
]

prediction_analysis = test[
    prediction_columns
].copy()

if "date" in prediction_analysis.columns:
    prediction_analysis["date"] = pd.to_datetime(
        prediction_analysis["date"],
        errors="coerce",
    )

if "prediction" in prediction_analysis.columns:

    prediction_analysis["prediction_category"] = (
        prediction_analysis["prediction"]
        .apply(get_category)
    )

if "target_aqi" in prediction_analysis.columns:

    prediction_analysis["actual_category"] = (
        prediction_analysis["target_aqi"]
        .apply(get_category)
    )

if "aqi" in prediction_analysis.columns:

    prediction_analysis["actual_change"] = (
        prediction_analysis["target_aqi"]
        - prediction_analysis["aqi"]
    )

    prediction_analysis["predicted_change"] = (
        prediction_analysis["prediction"]
        - prediction_analysis["aqi"]
    )

    prediction_analysis["change_error"] = (
        prediction_analysis["predicted_change"]
        - prediction_analysis["actual_change"]
    )

prediction_analysis = prediction_analysis.sort_values(
    "date"
)

save_csv(
    prediction_analysis,
    "prediction_analysis.csv",
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print_header("8. MODEL PERFORMANCE")

model_rows = []

if "prediction" in test.columns:

    actual = test["target_aqi"].astype(float)
    predicted = test["prediction"].astype(float)

    error = predicted - actual

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot != 0
        else np.nan
    )

    model_rows.append(
        {
            "model": "Production XGBoost",
            "rows": len(test),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "within_10": round(
                float(
                    (np.abs(error) <= 10).mean() * 100
                ),
                2,
            ),
            "within_20": round(
                float(
                    (np.abs(error) <= 20).mean() * 100
                ),
                2,
            ),
            "within_30": round(
                float(
                    (np.abs(error) <= 30).mean() * 100
                ),
                2,
            ),
        }
    )


# Try to include V6 experiment if available.

V6_RESULTS = (
    ROOT
    / "models"
    / "delta_v6"
    / "v6_strategy_results.csv"
)

if V6_RESULTS.exists():

    try:

        v6 = pd.read_csv(V6_RESULTS)

        for _, row in v6.iterrows():

            model_rows.append(
                {
                    "model": str(
                        row.get(
                            "strategy",
                            "V6 Delta",
                        )
                    ),
                    "rows": len(test),
                    "mae": safe_float(
                        row.get("test_mae")
                    ),
                    "rmse": safe_float(
                        row.get("test_rmse")
                    ),
                    "r2": safe_float(
                        row.get("test_r2")
                    ),
                    "within_10": None,
                    "within_20": None,
                    "within_30": None,
                }
            )

    except Exception as exc:
        print(
            f"Warning: could not read V6 results: {exc}"
        )


model_performance = pd.DataFrame(
    model_rows
)

save_csv(
    model_performance,
    "model_performance.csv",
)


# ============================================================
# DASHBOARD DATASET
# ============================================================

print_header("9. DASHBOARD DATASET")

dashboard = daily_aqi.copy()

dashboard["year"] = dashboard["date"].dt.year
dashboard["month"] = dashboard["date"].dt.month
dashboard["month_name"] = dashboard[
    "date"
].dt.month_name()
dashboard["day_of_week"] = dashboard[
    "date"
].dt.day_name()

dashboard["aqi_category"] = dashboard[
    "aqi"
].apply(get_category)

# Useful rolling statistics for charts.

dashboard = dashboard.sort_values(
    ["city_name", "date"]
)

dashboard["aqi_7d_average"] = (
    dashboard
    .groupby("city_name")["aqi"]
    .transform(
        lambda x: x.rolling(
            7,
            min_periods=1,
        ).mean()
    )
)

dashboard["aqi_30d_average"] = (
    dashboard
    .groupby("city_name")["aqi"]
    .transform(
        lambda x: x.rolling(
            30,
            min_periods=1,
        ).mean()
    )
)

dashboard["aqi_change_1d"] = (
    dashboard
    .groupby("city_name")["aqi"]
    .diff()
)

dashboard["aqi_change_1d"] = (
    dashboard["aqi_change_1d"]
    .round(2)
)

dashboard["aqi_7d_average"] = (
    dashboard["aqi_7d_average"]
    .round(2)
)

dashboard["aqi_30d_average"] = (
    dashboard["aqi_30d_average"]
    .round(2)
)

dashboard_path = (
    ANALYTICS_DIR
    / "dashboard_dataset.parquet"
)

dashboard.to_parquet(
    dashboard_path,
    index=False,
)

print(
    f"Saved: {dashboard_path}"
)


# ============================================================
# SUMMARY JSON
# ============================================================

print_header("10. DASHBOARD SUMMARY")

latest_date = historical["date"].max()

latest_rows = historical[
    historical["date"] == latest_date
]

summary = {
    "project": "PearlsAQI",
    "generated_at": pd.Timestamp.now().isoformat(),
    "historical_rows": int(len(historical)),
    "cities": int(
        historical["city_name"].nunique()
    ),
    "start_date": str(
        historical["date"].min().date()
    ),
    "end_date": str(
        historical["date"].max().date()
    ),
    "latest_date": str(
        latest_date.date()
    ),
    "latest_average_aqi": round(
        float(latest_rows["aqi"].mean()),
        2,
    ),
    "latest_highest_city": (
        latest_rows
        .loc[latest_rows["aqi"].idxmax()]
        ["city_name"]
    ),
    "latest_highest_aqi": round(
        float(latest_rows["aqi"].max()),
        2,
    ),
    "latest_lowest_city": (
        latest_rows
        .loc[latest_rows["aqi"].idxmin()]
        ["city_name"]
    ),
    "latest_lowest_aqi": round(
        float(latest_rows["aqi"].min()),
        2,
    ),
}

summary_path = (
    ANALYTICS_DIR
    / "dashboard_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )

print(
    f"Saved: {summary_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("PEARLSAQI ANALYTICS COMPLETE")

print(
    f"""
Cities analyzed       : {historical["city_name"].nunique()}
Historical rows        : {len(historical):,}
Date range             : {historical["date"].min().date()}
                         → {historical["date"].max().date()}

Analytics directory:
{ANALYTICS_DIR}

Generated datasets:
    ✓ daily_aqi.csv
    ✓ city_statistics.csv
    ✓ monthly_aqi.csv
    ✓ pollutant_statistics.csv
    ✓ aqi_categories.csv
    ✓ city_comparison.csv
    ✓ prediction_analysis.csv
    ✓ model_performance.csv
    ✓ dashboard_dataset.parquet
    ✓ dashboard_summary.json
"""
)

print("PearlsAQI analytics pipeline completed successfully.")