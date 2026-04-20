from pathlib import Path
import pandas as pd


# Return the project root directory.
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Load a raw dataset from the data/raw folder.
def get_raw_data(filename: str = "Consumer_Shopping_Trends_2026.csv") -> pd.DataFrame:
    data_path = get_project_root() / "data" / "raw" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")

    df = pd.read_csv(data_path)
    return df


# save processed data to the data/processed folder.
def save_processed_data(
    df: pd.DataFrame,
    filename: str = "consumer_shopping_trends_processed.csv"
) -> Path:

    processed_dir = get_project_root() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    save_path = processed_dir / filename
    df.to_csv(save_path, index=False)

    return save_path