"""
Main Script: Air Quality Analysis & Pollution Prediction
========================================================
Run from VS Code terminal:
    python main.py

This script runs the complete pipeline:
1. Data Cleaning & Feature Engineering (data_processing.py)
2. Exploratory Data Analysis & Plot Generation (eda.py)
3. Model Training & Metric Evaluation (model_training.py)
4. Exports outputs/model.joblib for the Streamlit dashboard (app.py)
"""

import sys
from pathlib import Path

# Add project root directory to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_processing import load_and_preprocess_data, print_missing_dataset_instructions
from eda import generate_all_eda_plots
from model_training import train_and_evaluate_model

def run_pipeline():
    """Executes end-to-end air quality analysis & prediction workflow."""
    print("\n" + "#" * 75)
    print(" COLLEGE MAJOR PROJECT: AIR QUALITY ANALYSIS & POLLUTION PREDICTION ")
    print("#" * 75)
    
    # Step 1: Preprocess Data
    try:
        df = load_and_preprocess_data(verbose=True)
    except FileNotFoundError:
        return
    except Exception as err:
        print(f"[ERROR] Failed during data processing: {err}")
        return

    # Step 2: Exploratory Data Analysis & Plot Generation
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING EXPLORATORY DATA ANALYSIS (EDA) CHARTS")
    print("=" * 70)
    generate_all_eda_plots(df)

    # Step 3: Model Training, Evaluation & Saving Artifacts
    results = train_and_evaluate_model(df)

    # Completion Summary
    print("\n" + "#" * 75)
    print(" PIPELINE COMPLETED SUCCESSFULLY! ")
    print("#" * 75)
    print("\nAll generated charts and model artifacts are saved in the 'outputs' directory:")
    print("  - outputs/cleaned_air_quality_data.csv")
    print("  - outputs/01_co_distribution.png")
    print("  - outputs/02_hourly_co_trend.png")
    print("  - outputs/03_monthly_co_trend.png")
    print("  - outputs/04_correlation_matrix.png")
    print("  - outputs/05_actual_vs_predicted.png")
    print("  - outputs/06_feature_importance.png")
    print("  - outputs/model.joblib (Trained ML Model)")
    print("\nTo launch the interactive Web Dashboard, run:")
    print("    streamlit run app.py\n")

if __name__ == "__main__":
    run_pipeline()
