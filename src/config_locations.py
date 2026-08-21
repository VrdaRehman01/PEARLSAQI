# Lat/lon per city, used by both live weather fetch (weather_api.py)
# and historical weather fetch (weather_history.py) so both sources
# query the exact same physical location.
#
# The original 4 cities (Karachi, Lahore, Islamabad, Peshawar) have
# verified 4-year hourly historical data (via download_historical_data.py)
# and are safe to train per-city models on. The 8 cities added below have
# NO historical dataset backing them yet -- live ingestion works
# immediately, but see HISTORICAL_DATA_CITIES below before training.

CITY_COORDINATES = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Rawalpindi": (33.5651, 73.0169),
    "Faisalabad": (31.4180, 73.0790),
    "Multan": (30.1575, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Hyderabad": (25.3960, 68.3578),
    "Gujranwala": (32.1877, 74.1945),
    "Sialkot": (32.4945, 74.5229),
    "Bahawalpur": (29.4000, 71.6833),
}

# Cities with a verified multi-year historical dataset (see
# download_historical_data.py) -- these are safe to backfill and train
# per-city models on. The remaining cities in CITY_COORDINATES only have
# live data until they accumulate their own history.
HISTORICAL_DATA_CITIES = ["Karachi", "Lahore", "Islamabad", "Peshawar"]
