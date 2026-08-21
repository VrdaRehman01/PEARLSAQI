from pathlib import Path
import duckdb

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Database folder
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(exist_ok=True)

# Database file
DATABASE_PATH = DATABASE_DIR / "aqi.duckdb"


def get_connection():
    """
    Returns a DuckDB connection.
    Creates the database automatically if it doesn't exist.
    """
    return duckdb.connect(str(DATABASE_PATH))