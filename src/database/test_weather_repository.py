from src.database.repositories.weather_repository import WeatherRepository


repo = WeatherRepository()

print("Weather records:", repo.count())

repo.close()