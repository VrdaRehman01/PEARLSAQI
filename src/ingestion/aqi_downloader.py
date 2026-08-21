from datetime import date, timedelta
import time
import numpy as np

from src.ingestion.api_client import APIClient
from src.ingestion.city_loader import load_cities
from src.database.repositories.aqi_repository import AQIRepository


class AQIDownloader:

    BASE_URL = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    # Small chunks are safer for the daily production updater.
    CHUNK_DAYS = 7

    def __init__(self):

        self.client = APIClient(
            timeout=60,
            max_retries=5,
            retry_delay=10
        )

        self.repository = AQIRepository()

    # ==========================================================
    # Generate date chunks
    # ==========================================================

    def generate_date_chunks(
        self,
        start_date,
        end_date
    ):

        start = date.fromisoformat(
            start_date
        )

        end = date.fromisoformat(
            end_date
        )

        current = start

        while current <= end:

            chunk_end = min(
                current + timedelta(
                    days=self.CHUNK_DAYS - 1
                ),
                end
            )

            yield (
                current.isoformat(),
                chunk_end.isoformat()
            )

            current = (
                chunk_end
                + timedelta(days=1)
            )

    # ==========================================================
    # Safe average
    # ==========================================================

    @staticmethod
    def average(values):

        if not values:
            return None

        return float(
            np.mean(values)
        )

    # ==========================================================
    # Latest AQI date for one city
    # ==========================================================

    def get_latest_date(
        self,
        city_id
    ):

        result = self.repository.execute(
            """
            SELECT MAX(date)
            FROM aqi
            WHERE city_id = ?
            """,
            [city_id]
        ).fetchone()

        if result is None:
            return None

        return result[0]

    # ==========================================================
    # Delete overlapping records
    #
    # This is NOT deleting history.
    # It only replaces the exact date range being refreshed.
    # ==========================================================

    def delete_existing_dates(
        self,
        city_id,
        start_date,
        end_date
    ):

        self.repository.execute(
            """
            DELETE FROM aqi
            WHERE city_id = ?
              AND date BETWEEN ? AND ?
            """,
            [
                city_id,
                start_date,
                end_date
            ]
        )

    # ==========================================================
    # Download incremental AQI
    #
    # Default:
    # download through YESTERDAY.
    #
    # This prevents using a partially completed current day
    # as the "actual" AQI.
    # ==========================================================

    def download_incremental(
        self,
        end_date=None
    ):

        print("\n" + "=" * 70)
        print("PEARLSAQI INCREMENTAL AQI UPDATE")
        print("=" * 70)

        # ------------------------------------------------------
        # Default = yesterday
        # ------------------------------------------------------

        if end_date is None:

            end_date = (
                date.today()
                - timedelta(days=1)
            ).isoformat()

        requested_end = date.fromisoformat(
            end_date
        )

        print(
            f"Requested completed-data end date: "
            f"{requested_end}"
        )

        cities = load_cities()

        total_inserted = 0
        total_skipped = 0

        # ======================================================
        # Process all 12 cities
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

            # --------------------------------------------------
            # No historical data
            # --------------------------------------------------

            if latest is None:

                print()
                print(
                    f"{city_name}: "
                    "no existing AQI data."
                )

                print(
                    "Skipping automatic historical rebuild."
                )

                print(
                    "Run your historical AQI downloader/importer "
                    "separately if required."
                )

                total_skipped += 1

                continue

            # --------------------------------------------------
            # Convert latest database date
            # --------------------------------------------------

            latest = date.fromisoformat(
                str(latest)
            )

            start = (
                latest
                + timedelta(days=1)
            )

            # --------------------------------------------------
            # Already current
            # --------------------------------------------------

            if start > requested_end:

                print()
                print(
                    f"{city_name}: "
                    f"already up to date "
                    f"({latest})"
                )

                continue

            print("\n" + "-" * 70)

            print(
                f"{city_name}: "
                f"{latest} -> {requested_end}"
            )

            records = []

            # ==================================================
            # Download missing dates
            # ==================================================

            for (
                chunk_start,
                chunk_end
            ) in self.generate_date_chunks(
                start.isoformat(),
                requested_end.isoformat()
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

                    "hourly": (
                        "pm2_5,"
                        "pm10,"
                        "carbon_monoxide,"
                        "nitrogen_dioxide,"
                        "sulphur_dioxide,"
                        "ozone,"
                        "us_aqi"
                    ),

                    "timezone":
                        "Asia/Karachi"
                }

                # ------------------------------------------------
                # API failure must NOT kill the whole pipeline
                # ------------------------------------------------

                try:

                    data = self.client.get(
                        self.BASE_URL,
                        params
                    )

                except RuntimeError as error:

                    print(
                        f"WARNING: AQI request failed "
                        f"for {city_name}"
                    )

                    print(
                        f"Range: "
                        f"{chunk_start} -> {chunk_end}"
                    )

                    print(
                        f"Reason: {error}"
                    )

                    continue

                hourly = data.get(
                    "hourly"
                )

                if not hourly:

                    print(
                        "WARNING: "
                        "No hourly AQI data returned."
                    )

                    continue

                # ==================================================
                # Extract hourly arrays
                # ==================================================

                times = hourly.get(
                    "time",
                    []
                )

                pm25 = hourly.get(
                    "pm2_5",
                    []
                )

                pm10 = hourly.get(
                    "pm10",
                    []
                )

                co = hourly.get(
                    "carbon_monoxide",
                    []
                )

                no2 = hourly.get(
                    "nitrogen_dioxide",
                    []
                )

                so2 = hourly.get(
                    "sulphur_dioxide",
                    []
                )

                o3 = hourly.get(
                    "ozone",
                    []
                )

                us_aqi = hourly.get(
                    "us_aqi",
                    []
                )

                # ==================================================
                # Hourly -> daily
                # ==================================================

                daily = {}

                for i, timestamp in enumerate(
                    times
                ):

                    current_date = (
                        timestamp[:10]
                    )

                    if current_date not in daily:

                        daily[current_date] = {

                            "aqi": [],
                            "pm25": [],
                            "pm10": [],
                            "co": [],
                            "no2": [],
                            "so2": [],
                            "o3": []

                        }

                    # ----------------------------------------------
                    # AQI
                    # ----------------------------------------------

                    if i < len(us_aqi):

                        value = us_aqi[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "aqi"
                            ].append(
                                float(value)
                            )

                    # ----------------------------------------------
                    # PM2.5
                    # ----------------------------------------------

                    if i < len(pm25):

                        value = pm25[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "pm25"
                            ].append(
                                float(value)
                            )

                    # ----------------------------------------------
                    # PM10
                    # ----------------------------------------------

                    if i < len(pm10):

                        value = pm10[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "pm10"
                            ].append(
                                float(value)
                            )

                    # ----------------------------------------------
                    # CO
                    # ----------------------------------------------

                    if i < len(co):

                        value = co[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "co"
                            ].append(
                                float(value)
                            )

                    # ----------------------------------------------
                    # NO2
                    # ----------------------------------------------

                    if i < len(no2):

                        value = no2[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "no2"
                            ].append(
                                float(value)
                            )

                    # ----------------------------------------------
                    # SO2
                    # ----------------------------------------------

                    if i < len(so2):

                        value = so2[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "so2"
                            ].append(
                                float(value)
                            )

                    # ----------------------------------------------
                    # O3
                    # ----------------------------------------------

                    if i < len(o3):

                        value = o3[i]

                        if value is not None:

                            daily[
                                current_date
                            ][
                                "o3"
                            ].append(
                                float(value)
                            )

                # ==================================================
                # Build daily records
                # ==================================================

                for (
                    current_date,
                    values
                ) in daily.items():

                    aqi_values = (
                        values["aqi"]
                    )

                    if not aqi_values:

                        continue

                    # Primary AQI definition:
                    # median of hourly US-AQI
                    daily_aqi = float(
                        np.median(
                            aqi_values
                        )
                    )

                    records.append(
                        (
                            None,
                            city_id,
                            current_date,
                            daily_aqi,

                            self.average(
                                values["pm25"]
                            ),

                            self.average(
                                values["pm10"]
                            ),

                            self.average(
                                values["no2"]
                            ),

                            self.average(
                                values["so2"]
                            ),

                            self.average(
                                values["co"]
                            ),

                            self.average(
                                values["o3"]
                            ),

                            "Open-Meteo"
                        )
                    )

                time.sleep(1)

            # ==================================================
            # Nothing downloaded
            # ==================================================

            if not records:

                print(
                    f"No new AQI records "
                    f"for {city_name}"
                )

                continue

            # ==================================================
            # Remove overlapping dates
            #
            # Normally this will affect only the downloaded
            # range. Historical dates before it remain untouched.
            # ==================================================

            first_date = records[0][2]
            last_date = records[-1][2]

            self.delete_existing_dates(
                city_id,
                first_date,
                last_date
            )

            # ==================================================
            # Generate IDs
            # ==================================================

            current_count = (
                self.repository.count()
            )

            final_records = []

            for index, record in enumerate(
                records,
                start=current_count + 1
            ):

                final_records.append(
                    (
                        index,
                        *record[1:]
                    )
                )

            # ==================================================
            # Insert
            # ==================================================

            self.repository.insert_many(
                final_records
            )

            total_inserted += len(
                final_records
            )

            print(
                f"Inserted: "
                f"{len(final_records)}"
            )

        # ======================================================
        # Close
        # ======================================================

        total = self.repository.count()

        self.repository.close()

        # ======================================================
        # Summary
        # ======================================================

        print("\n" + "=" * 70)
        print("INCREMENTAL AQI UPDATE COMPLETE")
        print("=" * 70)

        print(
            f"New records : {total_inserted}"
        )

        print(
            f"Skipped     : {total_skipped}"
        )

        print(
            f"Total AQI   : {total}"
        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    downloader = AQIDownloader()

    downloader.download_incremental()