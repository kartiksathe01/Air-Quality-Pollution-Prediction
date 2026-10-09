# Air Quality Analysis and Pollution Prediction

## Project summary
This project explores historical air-quality measurements and trains a Random Forest regression model to estimate hourly carbon monoxide concentration (`CO(GT)`, mg/m³) from other pollutant, sensor, and environmental measurements.

**Important:** The model predicts CO concentration, not official Air Quality Index (AQI). The UCI data were collected in an Italian city during 2004–2005, so they should not be presented as current air quality in India.

## Dataset
Download the original UCI Air Quality dataset:
https://archive.ics.uci.edu/dataset/360/air+quality

Download `AirQualityUCI.csv` and place it beside `main.py`.

The dataset contains 9,358 hourly records and 15 attributes. The original data uses `-200` to indicate missing readings; the program converts these markers to missing values before analysis.

## Run in VS Code
1. Extract this project ZIP.
2. Download `AirQualityUCI.csv` from the UCI link above.
3. Place `AirQualityUCI.csv` beside `main.py`.
4. Open the project folder in VS Code.
5. Open Terminal > New Terminal.
6. Optional virtual environment on Windows:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
7. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
8. Run:
   ```bash
   python main.py
   ```

## Output files
The `outputs` folder is generated automatically:
- `01_co_distribution.png`
- `02_hourly_co_trend.png`
- `03_monthly_co_trend.png`
- `04_correlation_matrix.png`
- `05_actual_vs_predicted.png`
- `06_feature_importance.png`
- `cleaned_air_quality_data.csv`
- `model_results.txt`
- `dataset_summary.txt`
- `feature_importance.csv`

## Methodology
- Data loading with pandas
- Convert `-200` readings to missing values
- Parse date and time and create hour, month, and day-of-week features
- Visualize distributions, temporal trends, and correlations
- Use a chronological 80/20 train-test split
- Median imputation inside a scikit-learn pipeline
- Train Random Forest Regressor
- Evaluate with MAE, RMSE, and R²

## Dataset citation
Vito, S. (2008). *Air quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F

## Limitations
- Measurements are historical (2004–2005) and from an Italian city.
- The model estimates CO concentration; it does not compute official AQI.
- Historical patterns may not generalize to other places or current conditions.
- This project is educational and not an operational pollution alert system.
