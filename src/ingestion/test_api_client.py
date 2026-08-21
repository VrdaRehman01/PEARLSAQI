from src.ingestion.api_client import APIClient


client = APIClient()

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "current": "temperature_2m"
}

data = client.get(url, params)

print("\nAPI connection successful!")
print("Karachi temperature:", data["current"]["temperature_2m"])