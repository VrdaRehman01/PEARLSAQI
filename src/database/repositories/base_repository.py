from src.database.connection import get_connection


class BaseRepository:

    def __init__(self):
        self.conn = get_connection()

    def execute(self, query, params=None):

        if params is None:
            return self.conn.execute(query)

        return self.conn.execute(query, params)

    def executemany(self, query, params):

        return self.conn.executemany(query, params)

    def fetchall(self, query):

        return self.conn.execute(query).fetchall()

    def fetchdf(self, query):

        return self.conn.execute(query).df()

    def close(self):

        self.conn.close()