# List of cities the live ingestion pipeline fetches every run.
# Keep this in sync with CITY_COORDINATES in config_locations.py --
# every city here must also have coordinates there.
#
# All 12 cities collect live data. Only the 4 in HISTORICAL_DATA_CITIES
# (config_locations.py) have historical data for training -- see there
# for why the other 8 aren't backfilled yet.

CITIES = [
    "Karachi", "Lahore", "Islamabad", "Peshawar",
    "Rawalpindi", "Faisalabad", "Multan", "Quetta",
    "Hyderabad", "Gujranwala", "Sialkot", "Bahawalpur",
]
