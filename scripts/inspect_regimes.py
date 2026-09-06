import pandas as pd

file = "models/forecast/validation/forecast_validation_results.csv"

df = pd.read_csv(file)

df["origin_aqi"] = pd.to_numeric(
    df["origin_aqi"],
    errors="coerce"
)

df["actual_aqi"] = pd.to_numeric(
    df["actual_aqi"],
    errors="coerce"
)

df["predicted_aqi"] = pd.to_numeric(
    df["predicted_aqi"],
    errors="coerce"
)

df["error"] = (
    df["predicted_aqi"]
    - df["actual_aqi"]
)

def regime(aqi):
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

df["regime"] = df["origin_aqi"].apply(regime)

print("=== REGIME COUNTS ===")
print(
    df["regime"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n=== HAZARDOUS DETAILS ===")

hazardous = df[
    df["regime"] == "Hazardous"
].copy()

print(
    hazardous[
        [
            "city_name",
            "horizon",
            "origin_aqi",
            "predicted_aqi",
            "actual_aqi",
            "error",
        ]
    ].to_string(index=False)
)

print(
    "\nHazardous rows:",
    len(hazardous)
)

if len(hazardous) > 0:
    print(
        "Mean error:",
        hazardous["error"].mean()
    )

    print(
        "Median error:",
        hazardous["error"].median()
    )
