from src.database.repositories.base_repository import BaseRepository


class AQIRepository(BaseRepository):

    def clear(self):
        """Remove all AQI records."""
        self.execute("DELETE FROM aqi")

    def insert_many(self, records):

        query = """
        INSERT INTO aqi (
            id,
            city_id,
            date,
            aqi,
            pm25,
            pm10,
            no2,
            so2,
            co,
            o3,
            aqi_source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.executemany(query, records)

    def get_all(self):
        """Return all AQI records."""

        query = """
        SELECT *
        FROM aqi
        ORDER BY city_id, date
        """

        return self.fetchdf(query)

    def get_by_city(self, city_id):
        """Return AQI data for one city."""

        query = """
        SELECT *
        FROM aqi
        WHERE city_id = ?
        ORDER BY date
        """

        return self.fetchdf(query, [city_id])

    def count(self):
        """Return total number of AQI records."""

        result = self.execute(
            "SELECT COUNT(*) FROM aqi"
        ).fetchone()

        return result[0]

    def get_latest_date(self, city_id):

        result = self.execute(
            """
            SELECT MAX(date)
            FROM aqi
            WHERE city_id = ?
            """,
            [city_id]
        ).fetchone()

        return result[0]