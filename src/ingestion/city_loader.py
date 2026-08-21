from src.database.repositories.city_repository import CityRepository


def load_cities():

    repo = CityRepository()

    cities = repo.get_all_cities()

    repo.close()

    return cities