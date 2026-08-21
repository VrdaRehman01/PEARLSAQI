from src.database.connection import get_connection

c = get_connection()

print("=" * 80)
print("V9 FORECAST + FUTURE WEATHER AUDIT")
print("=" * 80)

df = c.execute("""
SELECT
    f.city_name,
    f.origin_date,
    f.forecast_date,
    f.horizon,
    f.predicted_aqi,

    w.temperature,
    w.humidity,
    w.precipitation,
    w.windspeed,
    w.forecast_origin

FROM forecast_predictions f

LEFT JOIN weather_forecasts w
    ON f.city_id = w.city_id
    AND f.forecast_date = w.date

WHERE f.origin_date = '2026-08-13'

ORDER BY
    f.city_id,
    f.horizon
""").fetchdf()

print(df.to_string(index=False))

print()
print("=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)

print("Rows:", len(df))
print("Cities:", df["city_name"].nunique())
print("Horizons:", sorted(df["horizon"].dropna().unique().tolist()))

missing_weather = df[
    df["temperature"].isna()
]

print()
print("Forecast rows missing future weather:", len(missing_weather))

if len(missing_weather) > 0:
    print(missing_weather.to_string(index=False))

print()
print("Weather origins:")
print(
    df["forecast_origin"]
    .dropna()
    .astype(str)
    .value_counts()
    .to_string()
)

c.close()