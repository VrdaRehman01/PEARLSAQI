from src.database.repositories.base_repository import BaseRepository


class WeatherRepository(BaseRepository):

    def clear(self):
        """Remove all existing weather records."""
        self.execute(
            "DELETE FROM weather"
        )

    def get_next_id(self):
        """
        Return the next safe weather row ID.

        Uses MAX(id), not COUNT(id), because rows may have
        been deleted or replaced.
        """

        result = self.execute(
            """
            SELECT COALESCE(
                MAX(id),
                0
            ) + 1
            FROM weather
            """
        ).fetchone()

        return int(result[0])

    def insert_many(self, records):
        """Insert multiple weather records."""

        query = """
        INSERT INTO weather (
            id,
            city_id,
            date,
            temperature,
            humidity,
            precipitation,
            windspeed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        self.executemany(
            query,
            records
        )

    def get_all(self):
        """Return all weather records."""

        query = """
        SELECT *
        FROM weather
        ORDER BY city_id, date
        """

        return self.fetchdf(query)

    def get_by_city(self, city_id):
        """Return weather data for one city."""

        query = """
        SELECT *
        FROM weather
        WHERE city_id = ?
        ORDER BY date
        """

        return self.fetchdf(
            query,
            [city_id]
        )

    def count(self):
        """Return number of weather records."""

        result = self.execute(
            """
            SELECT COUNT(*)
            FROM weather
            """
        ).fetchone()

        return int(result[0])