from datetime import date, timedelta
import time

from src.ingestion.api_client import APIClient
from src.ingestion.city_loader import load_cities
from src.database.repositories.weather_repository import WeatherRepository


class WeatherService:

    BASE_URL = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    CHUNK_DAYS = 7

    # Number of recent days to re-check.
    # This allows the service to repair missing/revised
    # historical weather without rebuilding the entire database.
    LOOKBACK_DAYS = 7

    def __init__(self):

        self.client = APIClient(
            timeout=60,
            max_retries=5,
            retry_delay=10,
        )

        self.repository = WeatherRepository()

    # ==========================================================
    # DATE CHUNKS
    # ==========================================================

    def generate_date_chunks(
        self,
        start_date,
        end_date,
    ):

        start = date.fromisoformat(
            str(start_date)
        )

        end = date.fromisoformat(
            str(end_date)
        )

        current = start

        while current <= end:

            chunk_end = min(
                current + timedelta(
                    days=self.CHUNK_DAYS - 1
                ),
                end,
            )

            yield (
                current.isoformat(),
                chunk_end.isoformat(),
            )

            current = (
                chunk_end
                + timedelta(days=1)
            )

    # ==========================================================
    # LATEST WEATHER DATE
    # ==========================================================

    def get_latest_date(
        self,
        city_id,
    ):

        result = self.repository.execute(
            """
            SELECT MAX(date)
            FROM weather
            WHERE city_id = ?
            """,
            [city_id],
        ).fetchone()

        if result is None:
            return None

        if result[0] is None:
            return None

        return date.fromisoformat(
            str(result[0])
        )

    # ==========================================================
    # CHECK WHETHER CITY/DATE EXISTS
    # ==========================================================

    def date_exists(
        self,
        city_id,
        weather_date,
    ):

        result = self.repository.execute(
            """
            SELECT COUNT(*)
            FROM weather
            WHERE city_id = ?
              AND date = ?
            """,
            [
                city_id,
                weather_date,
            ],
        ).fetchone()

        return int(result[0]) > 0

    # ==========================================================
    # DELETE ONE CITY/DATE
    # ==========================================================

    def delete_date(
        self,
        city_id,
        weather_date,
    ):

        self.repository.execute(
            """
            DELETE FROM weather
            WHERE city_id = ?
              AND date = ?
            """,
            [
                city_id,
                weather_date,
            ],
        )

    # ==========================================================
    # NEXT SAFE ID
    #
    # We calculate the next ID ourselves instead of depending
    # on WeatherRepository.get_next_id().
    # ==========================================================

    def get_next_id(self):

        result = self.repository.execute(
            """
            SELECT COALESCE(
                MAX(id),
                0
            ) + 1
            FROM weather
            """
        ).fetchone()

        return int(result[0])

    # ==========================================================
    # DOWNLOAD INCREMENTAL WEATHER
    # ==========================================================

    def download_incremental(
        self,
        end_date=None,
    ):

        print("\n" + "=" * 70)
        print("PEARLSAQI HISTORICAL WEATHER UPDATE")
        print("=" * 70)

        # ======================================================
        # COUNTERS
        # ======================================================

        total_inserted = 0
        total_replaced = 0
        total_skipped = 0
        total_failed = 0

        # ======================================================
        # DETERMINE SAFE END DATE
        # ======================================================

        if end_date is None:

            requested_end = (
                date.today()
                - timedelta(days=1)
            )

        else:

            requested_end = date.fromisoformat(
                str(end_date)
            )

        # Historical weather must never contain
        # today's date or a future date.

        safe_end = (
            date.today()
            - timedelta(days=1)
        )

        if requested_end > safe_end:

            requested_end = safe_end

        print(
            f"Requested weather end date: "
            f"{requested_end}"
        )

        print(
            f"Historical weather safety cutoff: "
            f"{safe_end}"
        )

        # ======================================================
        # LOAD CITIES
        # ======================================================

        cities = load_cities()

        if cities is None or cities.empty:

            raise RuntimeError(
                "No cities were loaded."
            )

        # ======================================================
        # REMOVE ILLEGAL FUTURE/TODAY WEATHER
        #
        # Historical weather table must only contain dates
        # strictly before today.
        # ======================================================

        self.repository.execute(
            """
            DELETE FROM weather
            WHERE date > ?
            """,
            [
                safe_end.isoformat()
            ],
        )

        # ======================================================
        # PROCESS CITIES
        # ======================================================

        for _, city in cities.iterrows():

            city_id = int(
                city["city_id"]
            )

            city_name = str(
                city["city_name"]
            )

            latest = self.get_latest_date(
                city_id
            )

            # ==================================================
            # DETERMINE DOWNLOAD START
            # ==================================================

            if latest is not None:

                normal_start = (
                    latest
                    + timedelta(days=1)
                )

                repair_start = (
                    requested_end
                    - timedelta(
                        days=self.LOOKBACK_DAYS
                    )
                )

                # We want to repair recent gaps while also
                # downloading anything beyond the current latest.
                start = min(
                    normal_start,
                    repair_start,
                )

            else:

                print()
                print(
                    f"{city_name}: "
                    "no existing weather data."
                )

                print(
                    "Skipping automatic full historical rebuild."
                )

                total_skipped += 1

                continue

            # ==================================================
            # NOTHING TO DOWNLOAD
            # ==================================================

            if start > requested_end:

                print()
                print(
                    f"{city_name}: "
                    f"already up to date "
                    f"({latest})"
                )

                continue

            print()
            print("-" * 70)

            print(
                f"{city_name}: "
                f"{start} -> {requested_end}"
            )

            records = []

            # ==================================================
            # DOWNLOAD CHUNKS
            # ==================================================

            for (
                chunk_start,
                chunk_end,
            ) in self.generate_date_chunks(
                start,
                requested_end,
            ):

                print(
                    f"Downloading "
                    f"{chunk_start} -> {chunk_end}"
                )

                params = {

                    "latitude":
                        city["latitude"],

                    "longitude":
                        city["longitude"],

                    "start_date":
                        chunk_start,

                    "end_date":
                        chunk_end,

                    "daily": (
                        "temperature_2m_mean,"
                        "relative_humidity_2m_mean,"
                        "precipitation_sum,"
                        "windspeed_10m_max"
                    ),

                    "timezone":
                        "Asia/Karachi",
                }

                # ==================================================
                # API REQUEST
                # ==================================================

                try:

                    data = self.client.get(
                        self.BASE_URL,
                        params,
                    )

                except RuntimeError as error:

                    total_failed += 1

                    print()
                    print(
                        f"WARNING: Weather request failed "
                        f"for {city_name}"
                    )

                    print(
                        f"Range: "
                        f"{chunk_start} -> {chunk_end}"
                    )

                    print(
                        f"Reason: {error}"
                    )

                    print(
                        "This range will be retried "
                        "on a future run."
                    )

                    continue

                # ==================================================
                # VALIDATE RESPONSE
                # ==================================================

                daily = data.get(
                    "daily"
                )

                if not daily:

                    total_failed += 1

                    print(
                        "WARNING: "
                        "No daily weather data returned."
                    )

                    continue

                dates = daily.get(
                    "time",
                    [],
                )

                temperature = daily.get(
                    "temperature_2m_mean",
                    [],
                )

                humidity = daily.get(
                    "relative_humidity_2m_mean",
                    [],
                )

                precipitation = daily.get(
                    "precipitation_sum",
                    [],
                )

                windspeed = daily.get(
                    "windspeed_10m_max",
                    [],
                )

                if not dates:

                    total_failed += 1

                    print(
                        "WARNING: "
                        "Weather API returned no dates."
                    )

                    continue

                # ==================================================
                # BUILD RECORDS
                # ==================================================

                for i, current_date in enumerate(
                    dates
                ):

                    current_date_obj = (
                        date.fromisoformat(
                            str(current_date)
                        )
                    )

                    # Never allow today/future weather
                    # into historical table.

                    if current_date_obj > safe_end:

                        print(
                            "Skipping non-historical "
                            f"weather date: {current_date}"
                        )

                        continue

                    temp_value = (
                        temperature[i]
                        if i < len(temperature)
                        else None
                    )

                    humidity_value = (
                        humidity[i]
                        if i < len(humidity)
                        else None
                    )

                    precipitation_value = (
                        precipitation[i]
                        if i < len(precipitation)
                        else None
                    )

                    windspeed_value = (
                        windspeed[i]
                        if i < len(windspeed)
                        else None
                    )

                    # ==================================================
                    # REJECT INCOMPLETE ROWS
                    # ==================================================

                    if (
                        temp_value is None
                        or humidity_value is None
                        or precipitation_value is None
                        or windspeed_value is None
                    ):

                        print(
                            "Skipping incomplete "
                            f"weather row "
                            f"{city_name} "
                            f"{current_date}"
                        )

                        continue

                    records.append(
                        (
                            city_id,
                            current_date_obj,
                            float(temp_value),
                            float(humidity_value),
                            float(precipitation_value),
                            float(windspeed_value),
                        )
                    )

                time.sleep(1)

            # ==================================================
            # NO USABLE RECORDS
            # ==================================================

            if not records:

                print(
                    f"No usable weather records "
                    f"for {city_name}"
                )

                continue

            # ==================================================
            # INSERT / REPLACE
            # ==================================================

            for record in records:

                (
                    record_city_id,
                    record_date,
                    temperature_value,
                    humidity_value,
                    precipitation_value,
                    windspeed_value,
                ) = record

                # ------------------------------------------------
                # Check whether this city/date already exists.
                # ------------------------------------------------

                existed = self.date_exists(
                    record_city_id,
                    record_date,
                )

                if existed:

                    self.delete_date(
                        record_city_id,
                        record_date,
                    )

                    total_replaced += 1

                # ------------------------------------------------
                # Generate ID safely.
                # ------------------------------------------------

                next_id = self.get_next_id()

                # ------------------------------------------------
                # Insert exactly one clean row.
                # ------------------------------------------------

                self.repository.insert_many(
                    [
                        (
                            next_id,
                            record_city_id,
                            record_date,
                            temperature_value,
                            humidity_value,
                            precipitation_value,
                            windspeed_value,
                        )
                    ]
                )

                total_inserted += 1

            print(
                f"Processed weather rows: "
                f"{len(records)}"
            )

        # ======================================================
        # FINAL COUNT
        # ======================================================

        total = self.repository.count()

        # ======================================================
        # CLOSE DATABASE CONNECTION
        # ======================================================

        self.repository.close()

        # ======================================================
        # FINAL SUMMARY
        # ======================================================

        print()
        print("=" * 70)
        print("HISTORICAL WEATHER UPDATE COMPLETE")
        print("=" * 70)

        print(
            f"Inserted / refreshed : "
            f"{total_inserted}"
        )

        print(
            f"Replaced existing    : "
            f"{total_replaced}"
        )

        print(
            f"Skipped              : "
            f"{total_skipped}"
        )

        print(
            f"Failed API chunks    : "
            f"{total_failed}"
        )

        print(
            f"Total weather rows   : "
            f"{total}"
        )

        print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    service = WeatherService()

    service.download_incremental()