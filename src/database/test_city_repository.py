from src.database.repositories.city_repository import CityRepository


repo = CityRepository()

cities = repo.get_all_cities()

print(cities)

repo.close()