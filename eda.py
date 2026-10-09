"""
Exploratory Data Analysis (EDA) & Visualization Module
======================================================
Project: Air Quality Analysis and Pollution Prediction using Machine Learning

Generates data visualizations:
1. Carbon Monoxide (CO) concentration distribution.
2. Hourly CO concentration trend (0-23 hours).
3. Monthly CO concentration trend (1-12 months).
4. Feature correlation matrix heatmap.
5. Temperature & Relative Humidity vs CO relationship plots.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Use clean visualization theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def plot_co_distribution(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Plot distribution (histogram + KDE) of CO(GT) concentration."""
    fig, ax = plt.subplots(figsize=(9, 5))
    co_data = df["CO(GT)"].dropna()
    sns.histplot(co_data, bins=35, kde=True, color="#2b5c8f", ax=ax)
    
    mean_val = co_data.mean()
    median_val = co_data.median()
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f} mg/m³")
    ax.axvline(median_val, color="green", linestyle="-.", label=f"Median: {median_val:.2f} mg/m³")
    
    ax.set_title("Distribution of Hourly Carbon Monoxide (CO) Concentration", fontsize=14, pad=12)
    ax.set_xlabel("CO(GT) Concentration (mg/m³)")
    ax.set_ylabel("Reading Frequency (Hours)")
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig

def plot_hourly_trend(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Plot average CO concentration by hour of day (0 to 23)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    hourly_df = df.groupby("hour", as_index=False)["CO(GT)"].mean()
    
    sns.lineplot(data=hourly_df, x="hour", y="CO(GT)", marker="o", color="#d95f02", linewidth=2.5, markersize=7, ax=ax)
    ax.set_title("Average CO Concentration by Hour of Day", fontsize=14, pad=12)
    ax.set_xlabel("Hour of Day (0:00 - 23:00)")
    ax.set_ylabel("Mean CO(GT) (mg/m³)")
    ax.set_xticks(range(0, 24, 2))
    
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig

def plot_monthly_trend(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Plot average CO concentration by month (1 to 12)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    monthly_df = df.groupby("month", as_index=False)["CO(GT)"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_df["month_name"] = monthly_df["month"].apply(lambda m: month_names[m-1] if 1 <= m <= 12 else str(m))
    
    sns.barplot(data=monthly_df, x="month_name", y="CO(GT)", hue="month_name", palette="viridis", legend=False, ax=ax)
    ax.set_title("Average CO Concentration by Month", fontsize=14, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean CO(GT) (mg/m³)")
    
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig

def plot_correlation_matrix(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Plot correlation matrix heatmap of numerical measurement columns."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ["hour", "month", "day_of_week"]]
    corr = df[num_cols].corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation Coefficient"})
    ax.set_title("Correlation Heatmap of Pollutants & Environmental Factors", fontsize=14, pad=12)
    
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig

def generate_all_eda_plots(df: pd.DataFrame, output_dir: Path = None) -> dict:
    """Generates and saves all EDA charts to output folder."""
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    figures["co_distribution"] = plot_co_distribution(df, save_path=output_dir / "01_co_distribution.png")
    figures["hourly_trend"] = plot_hourly_trend(df, save_path=output_dir / "02_hourly_co_trend.png")
    figures["monthly_trend"] = plot_monthly_trend(df, save_path=output_dir / "03_monthly_co_trend.png")
    figures["correlation_matrix"] = plot_correlation_matrix(df, save_path=output_dir / "04_correlation_matrix.png")
    
    plt.close("all")
    print(f"All EDA plots generated and saved to: {output_dir}")
    return figures

if __name__ == "__main__":
    from data_processing import load_and_preprocess_data
    df = load_and_preprocess_data(verbose=False)
    generate_all_eda_plots(df)
