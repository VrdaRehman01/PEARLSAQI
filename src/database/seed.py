from src.database.connection import get_connection


PAKISTAN_CITIES = [

    (1, "Karachi", "Sindh", 24.8607, 67.0011),

    (2, "Lahore", "Punjab", 31.5204, 74.3587),

    (3, "Islamabad", "ICT", 33.6844, 73.0479),

    (4, "Rawalpindi", "Punjab", 33.5651, 73.0169),

    (5, "Faisalabad", "Punjab", 31.4504, 73.1350),

    (6, "Multan", "Punjab", 30.1575, 71.5249),

    (7, "Peshawar", "KPK", 34.0151, 71.5249),

    (8, "Quetta", "Balochistan", 30.1798, 66.9750),

    (9, "Hyderabad", "Sindh", 25.3960, 68.3578),

    (10, "Gujranwala", "Punjab", 32.1877, 74.1945),

    (11, "Sialkot", "Punjab", 32.4945, 74.5229),

    (12, "Bahawalpur", "Punjab", 29.3956, 71.6836)

]


def seed_cities():

    conn = get_connection()

    conn.execute("DELETE FROM cities")

    conn.executemany(

        """
        INSERT INTO cities
        VALUES (?, ?, ?, ?, ?)
        """,

        PAKISTAN_CITIES

    )

    conn.close()

    print("Cities inserted successfully!")