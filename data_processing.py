"""
Data Processing & Dataset Preprocessing Module
==============================================
Project: Air Quality Analysis and Pollution Prediction using Machine Learning
Dataset: UCI Air Quality Dataset (AirQualityUCI.csv)
URL: https://archive.ics.uci.edu/dataset/360/air+quality

Requirements handled:
1. Semicolon-separated CSV reading with decimal commas.
2. Handling empty trailing rows and columns.
3. Converting -200 sentinel values to missing values (NaN).
4. Datetime parsing combining 'Date' (%d/%m/%Y) and 'Time' (%H.%M.%S).
5. Safe missing value handling (retaining values for downstream imputation).
6. Extraction of hour, month, day_of_week temporal features.
7. Saving clean output to `outputs/cleaned_air_quality_data.csv`.
8. Reporting dataset shape, missing value stats, and data types.
9. Clear error guidance if AirQualityUCI.csv is missing.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "AirQualityUCI.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs"

UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/dataset/360/air+quality"

def get_data_filepath() -> Path:
    """Finds AirQualityUCI.csv in project folder or subfolders."""
    possible_paths = [
        ROOT_DIR / DATA_FILENAME,
        ROOT_DIR / "data" / DATA_FILENAME,
        Path.cwd() / DATA_FILENAME
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

def print_missing_dataset_instructions():
    """Prints clear instructions when dataset file is not found."""
    print("\n" + "=" * 75)
    print(" [ERROR] DATASET NOT FOUND: AirQualityUCI.csv")
    print("=" * 75)
    print("Please follow these steps to place the dataset file:")
    print(f"1. Download the dataset ZIP from UCI repository:")
    print(f"   {UCI_DOWNLOAD_URL}")
    print("2. Extract AirQualityUCI.csv from the downloaded archive.")
    print(f"3. Place 'AirQualityUCI.csv' in the project directory:")
    print(f"   {ROOT_DIR / DATA_FILENAME}")
    print("=" * 75 + "\n")

def load_and_preprocess_data(data_path: Path = None, verbose: bool = True) -> pd.DataFrame:
    """
    Main preprocessing function:
    - Loads raw semicolon-delimited CSV with comma decimals.
    - Drops empty rows and columns.
    - Replaces -200 missing markers with NaN.
    - Parses Date and Time columns into a unified datetime feature.
    - Engineers hour, month, and day_of_week features.
    - Exports cleaned dataset to outputs/cleaned_air_quality_data.csv.
    """
    if data_path is None:
        data_path = get_data_filepath()
        
    if data_path is None or not Path(data_path).exists():
        print_missing_dataset_instructions()
        raise FileNotFoundError("AirQualityUCI.csv is missing. See instructions above.")

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: DATASET LOADING AND PREPROCESSING")
        print("=" * 70)
        print(f"Loading raw dataset from: {data_path}")

    # 1. Load semicolon-separated CSV with ',' decimal separator
    raw_df = pd.read_csv(data_path, sep=";", decimal=",", encoding="utf-8")

    # 2. Handle empty trailing columns and completely empty rows
    cleaned_df = raw_df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()
    cleaned_df.columns = [str(col).strip() for col in cleaned_df.columns]

    raw_rows, raw_cols = cleaned_df.shape

    # 3. Convert numeric columns and replace -200 values with NaN
    for col in cleaned_df.columns:
        if col not in ["Date", "Time"]:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
            cleaned_df.loc[cleaned_df[col] == -200, col] = np.nan

    # 4. Parse Date (%d/%m/%Y) and Time (%H.%M.%S)
    datetime_str = cleaned_df["Date"].astype(str).str.strip() + " " + cleaned_df["Time"].astype(str).str.strip()
    cleaned_df["datetime"] = pd.to_datetime(datetime_str, format="%d/%m/%Y %H.%M.%S", errors="coerce")

    # Drop rows where datetime couldn't be parsed and sort chronologically
    cleaned_df = cleaned_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # 5. Handle missing values safely (retaining invalid pollutant readings as NaN for downstream pipeline imputation)
    
    # 6. Engineer temporal features
    cleaned_df["hour"] = cleaned_df["datetime"].dt.hour
    cleaned_df["month"] = cleaned_df["datetime"].dt.month
    cleaned_df["day_of_week"] = cleaned_df["datetime"].dt.dayofweek

    # 7. Save cleaned dataset to outputs folder
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUTS_DIR / "cleaned_air_quality_data.csv"
    cleaned_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nSuccessfully cleaned and saved dataset to: {output_file}")
        print("\n--- DATASET SUMMARY ---")
        print(f"Raw shape (non-empty): {raw_rows} rows, {raw_cols} columns")
        print(f"Processed shape:       {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
        
        print("\n--- DATA TYPES ---")
        for col, dtype in cleaned_df.dtypes.items():
            print(f"  {col:<20} : {dtype}")

        print("\n--- MISSING VALUES SUMMARY (NaN count per column) ---")
        missing = cleaned_df.isna().sum()
        for col, count in missing.items():
            pct = (count / len(cleaned_df)) * 100
            print(f"  {col:<20} : {count:5d} missing ({pct:5.1f}%)")

        print("=" * 70 + "\n")

    return cleaned_df

if __name__ == "__main__":
    try:
        df = load_and_preprocess_data(verbose=True)
    except Exception as err:
        print(f"Execution stopped: {err}")
