from src.ingestion.weather_history import fetch_weather_history

df = fetch_weather_history(
    city="Karachi",
    start_date="2022-01-01",
    end_date="2022-01-10"
)

print(df.head())
print(df.shape)
