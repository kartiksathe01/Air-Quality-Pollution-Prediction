"""
Model Training & Evaluation Module
===================================
Project: Air Quality Analysis and Pollution Prediction using Machine Learning

Handles:
1. Feature matrix (X) and target vector (y = CO(GT)) preparation.
2. Chronological Train/Test Split (80% Train, 20% Test).
3. Scikit-learn Pipeline with Median Imputation + RandomForestRegressor.
4. Metric Evaluation: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score.
5. Plotting Actual vs Predicted values & Feature Importances.
6. Saving trained model artifact (outputs/model.joblib) for live deployment in Streamlit.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

def prepare_features_and_target(df: pd.DataFrame, target_col: str = "CO(GT)"):
    """
    Extracts features (X) and target (y).
    Excludes datetime and non-numeric identifiers.
    Drops rows where target y is missing (NaN).
    """
    excluded_cols = {"Date", "Time", "datetime", target_col}
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    # Filter out rows where target variable is NaN
    valid_df = df.dropna(subset=[target_col]).copy()
    
    X = valid_df[feature_cols]
    y = valid_df[target_col]
    
    return X, y, feature_cols

def train_and_evaluate_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Trains Random Forest model using chronological split.
    Evaluates metrics (MAE, RMSE, R2) and saves trained model joblib artifact.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    X, y, feature_cols = prepare_features_and_target(df)
    
    # Chronological Split (first 80% train, final 20% test)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    print(f"Target variable: CO(GT) concentration (mg/m³)")
    print(f"Total valid samples: {len(X)}")
    print(f"Training split:      {len(X_train)} samples (first {int((1-test_size)*100)}%)")
    print(f"Testing split:       {len(X_test)} samples (last {int(test_size*100)}%)")
    print(f"Number of features:  {len(feature_cols)}")
    
    # Scikit-learn Pipeline: Median Imputation + Random Forest Regressor
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])
    
    # Fit pipeline on training data
    print("\nTraining Random Forest Regressor...")
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate performance metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    
    print("\n--- MODEL TEST METRICS ---")
    print(f"Mean Absolute Error (MAE):  {mae:.4f} mg/m³")
    print(f"Root Mean Sq Error (RMSE):  {rmse:.4f} mg/m³")
    print(f"Coefficient of Det (R²):    {r2:.4f}")
    
    # Plot Actual vs Predicted
    fig_act_pred, ax_ap = plt.subplots(figsize=(8, 6))
    ax_ap.scatter(y_test, y_pred, alpha=0.5, color="#1f77b4", label="Predictions")
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    ax_ap.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Match")
    ax_ap.set_title("Actual vs Predicted CO Concentration (Test Set)", fontsize=13)
    ax_ap.set_xlabel("Actual CO(GT) (mg/m³)")
    ax_ap.set_ylabel("Predicted CO(GT) (mg/m³)")
    ax_ap.legend()
    fig_act_pred.savefig(OUTPUTS_DIR / "05_actual_vs_predicted.png", dpi=160)
    plt.close(fig_act_pred)
    
    # Feature Importances
    rf_model = pipeline.named_steps["regressor"]
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(9, 6))
    sns.barplot(data=importance_df.head(12), x="importance", y="feature", hue="feature", palette="Blues_r", legend=False, ax=ax_imp)
    ax_imp.set_title("Top Feature Importances (Random Forest)", fontsize=13)
    ax_imp.set_xlabel("Importance Score")
    ax_imp.set_ylabel("Feature")
    fig_imp.savefig(OUTPUTS_DIR / "06_feature_importance.png", dpi=160)
    plt.close(fig_imp)
    
    # Save trained model pipeline for Streamlit application
    model_path = OUTPUTS_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nTrained model pipeline successfully saved to: {model_path}")
    
    # Save summary report text
    results_summary = (
        "Air Quality Analysis and Pollution Prediction Results\n"
        "======================================================\n"
        f"Target Column: CO(GT)\n"
        f"Train Rows: {len(X_train)} | Test Rows: {len(X_test)}\n"
        f"MAE:  {mae:.4f} mg/m³\n"
        f"RMSE: {rmse:.4f} mg/m³\n"
        f"R2:   {r2:.4f}\n"
    )
    (OUTPUTS_DIR / "model_results.txt").write_text(results_summary, encoding="utf-8")
    
    print("=" * 70 + "\n")
    
    return {
        "pipeline": pipeline,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "feature_importance": importance_df,
        "y_test": y_test,
        "y_pred": y_pred
    }

if __name__ == "__main__":
    from data_processing import load_and_preprocess_data
    df = load_and_preprocess_data(verbose=False)
    train_and_evaluate_model(df)
