from src.database.repositories.base_repository import BaseRepository


class CityRepository(BaseRepository):

    def get_all_cities(self):

        query = """
        SELECT
            city_id,
            city_name,
            province,
            latitude,
            longitude
        FROM cities
        ORDER BY city_name
        """

        return self.fetchdf(query)