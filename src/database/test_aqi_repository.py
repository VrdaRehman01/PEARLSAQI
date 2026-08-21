from src.database.repositories.aqi_repository import AQIRepository


repo = AQIRepository()

print("AQI records:", repo.count())

repo.close()