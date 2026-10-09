"""
Air Quality Analysis and Pollution Prediction
Run from VS Code: python main.py

Dataset: UCI Air Quality dataset (AirQualityUCI.csv)
This project predicts hourly CO concentration (regression), not AQI.
The source data are historical measurements from an Italian city (2004–2005).
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "AirQualityUCI.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

def main():
    if not DATA_PATH.exists():
        print("\nCould not find AirQualityUCI.csv.")
        print("Download AirQualityUCI.csv from the UCI link in README.md")
        print("and place it in the same folder as main.py.")
        return

    # UCI file may have a trailing empty column; remove fully empty columns.
    df = pd.read_csv(DATA_PATH, sep=";", decimal=",", encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    # Drop rows that are entirely empty, which can occur due to trailing CSV lines.
    df = df.dropna(how="all").copy()

    required = {"Date", "Time", "CO(GT)"}
    if not required.issubset(set(df.columns)):
        print("Unexpected CSV format. Expected columns include Date, Time, CO(GT).")
        print("Columns found:", df.columns.tolist())
        print("Check that you downloaded the original UCI AirQualityUCI.csv file.")
        return

    print("=" * 68)
    print("AIR QUALITY ANALYSIS AND POLLUTION PREDICTION")
    print("=" * 68)
    print(f"Raw shape after removing empty rows/columns: {df.shape}")
    print("\nFirst five rows:\n", df.head().to_string(index=False))

    # Convert numeric-looking columns; the UCI file uses -200 for missing readings.
    for col in df.columns:
        if col not in ["Date", "Time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == -200, col] = np.nan

    # Combine date and time into a timestamp. Source uses DD/MM/YYYY and HH.MM.SS.
    dt = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " +
        df["Time"].astype(str).str.strip(),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # Save clean copy; missing pollutant values are retained as NaN for transparency.
    df.to_csv(OUT / "cleaned_air_quality_data.csv", index=False)
    print("\nMissing values after converting -200 markers to NaN:\n")
    print(df.isna().sum().sort_values(ascending=False).head(12).to_string())

    # Plot 1: CO concentration distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["CO(GT)"].dropna(), bins=35, kde=True)
    plt.title("Distribution of Hourly CO Concentration")
    plt.xlabel("CO concentration (mg/m³)")
    plt.ylabel("Number of hourly readings")
    plt.tight_layout()
    plt.savefig(OUT / "01_co_distribution.png", dpi=160)
    plt.close()

    # Plot 2: hourly average CO
    hourly = df.groupby("hour", as_index=False)["CO(GT)"].mean()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=hourly, x="hour", y="CO(GT)", marker="o")
    plt.title("Average CO Concentration by Hour of Day")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean CO concentration (mg/m³)")
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(OUT / "02_hourly_co_trend.png", dpi=160)
    plt.close()

    # Plot 3: monthly mean CO
    monthly = df.groupby("month", as_index=False)["CO(GT)"].mean()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=monthly, x="month", y="CO(GT)")
    plt.title("Average CO Concentration by Month")
    plt.xlabel("Month number")
    plt.ylabel("Mean CO concentration (mg/m³)")
    plt.tight_layout()
    plt.savefig(OUT / "03_monthly_co_trend.png", dpi=160)
    plt.close()

    # Plot 4: correlation matrix of pollutant/environmental measures
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in ["hour", "month", "day_of_week"]]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Matrix of Numeric Measurements")
    plt.tight_layout()
    plt.savefig(OUT / "04_correlation_matrix.png", dpi=160)
    plt.close()

    # Regression task: estimate CO(GT) from other measured pollutant/sensor/weather values.
    # Exclude date/time and target; use date-derived features and sensor/other pollutant columns.
    target = "CO(GT)"
    excluded = {"Date", "Time", "datetime", target}
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Remove rows where target is missing. Features are imputed inside the training pipeline.
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    # Chronological split: first 80% for training, last 20% for testing.
    split_at = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    print("\nMODEL RESULTS (chronological 80/20 split)")
    print("-" * 68)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"MAE:  {mae:.4f} mg/m³")
    print(f"RMSE: {rmse:.4f} mg/m³")
    print(f"R²:   {r2:.4f}")

    results_text = (
        "Air Quality Analysis and Pollution Prediction\n"
        "Target: hourly CO(GT), mg/m³\n"
        "Split: chronological first 80% train, final 20% test\n"
        f"Training rows: {len(X_train)}\nTesting rows: {len(X_test)}\n"
        f"MAE: {mae:.6f} mg/m³\nRMSE: {rmse:.6f} mg/m³\nR2: {r2:.6f}\n\n"
        "Note: This predicts CO concentration, not official AQI. "
        "Results are based on historical Italian city data from 2004–2005."
    )
    (OUT / "model_results.txt").write_text(results_text, encoding="utf-8")

    # Plot 5: actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.5)
    lo = min(float(y_test.min()), float(np.min(pred)))
    hi = max(float(y_test.max()), float(np.max(pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title("Actual vs Predicted CO Concentration")
    plt.xlabel("Actual CO (mg/m³)")
    plt.ylabel("Predicted CO (mg/m³)")
    plt.tight_layout()
    plt.savefig(OUT / "05_actual_vs_predicted.png", dpi=160)
    plt.close()

    # Plot 6: feature importance
    importances = model.named_steps["regressor"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(OUT / "feature_importance.csv", index=False)
    top = importance_df.head(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature")
    plt.title("Top 15 Features for CO Concentration Prediction")
    plt.xlabel("Random Forest feature importance")
    plt.ylabel("Input feature")
    plt.tight_layout()
    plt.savefig(OUT / "06_feature_importance.png", dpi=160)
    plt.close()

    summary = [
        f"Rows after timestamp parsing: {len(df)}",
        f"Columns after adding time features: {len(df.columns)}",
        f"Valid target rows used: {len(X)}",
        f"Mean CO concentration (non-missing): {df[target].mean():.3f} mg/m³",
        f"Missing CO target rows: {int(df[target].isna().sum())}",
        f"MAE: {mae:.4f} mg/m³",
        f"RMSE: {rmse:.4f} mg/m³",
        f"R²: {r2:.4f}",
    ]
    (OUT / "dataset_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\nSaved results and six charts in:", OUT)
    print("Open outputs/model_results.txt and outputs/dataset_summary.txt for report values.")
    print("\nCaution: this is a historical research dataset, not live local AQI data.")

if __name__ == "__main__":
    main()
