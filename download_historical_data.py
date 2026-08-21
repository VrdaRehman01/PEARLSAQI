from huggingface_hub import hf_hub_download
import shutil
import os

FILES = [
    "historical_air_pollution_all_karachi.csv",
    "historical_air_pollution_all_lahore.csv",
    "historical_air_pollution_all_islamabad.csv",
    "historical_air_pollution_all_peshawar.csv",
]

SAVE_DIR = "data/historical"


def download_historical_data():

    os.makedirs(SAVE_DIR, exist_ok=True)

    for filename in FILES:
        print(f"Downloading {filename}...")

        downloaded_file = hf_hub_download(
            repo_id="mk12rule/pakistan_air_quality_dataset",
            filename=filename,
            repo_type="dataset",
        )

        destination = os.path.join(SAVE_DIR, filename)
        shutil.copy(downloaded_file, destination)

        print(f"Saved -> {destination}")

    print("\nAll historical datasets downloaded successfully!")


if __name__ == "__main__":
    download_historical_data()
