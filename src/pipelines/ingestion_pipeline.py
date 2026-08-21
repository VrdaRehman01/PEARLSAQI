from src.ingestion.fetch_data import main as fetch_data


def run_ingestion_pipeline():

    print("\n" + "=" * 60)
    print("STEP 1 : DATA INGESTION")
    print("=" * 60)

    fetch_data()

    print("\nData Ingestion Completed.")
