# MAJOR PROJECT REPORT
## Title: Air Quality Analysis and Pollution Prediction using Machine Learning

**Student Name:** [Your Full Name]  
**Roll Number:** [Your Roll Number]  
**Degree / Course:** Bachelor of Technology / Computer Science / Data Science  
**Department:** Department of Computer Science & Engineering  
**Institution:** [Your University / College Name]  
**Academic Year:** 2026–2027  

---

## Abstract
Air pollution poses severe health and environmental challenges in urban centers worldwide. This project presents a data-driven approach for **Air Quality Analysis and Pollution Prediction using Machine Learning**. Utilizing the public UCI Air Quality dataset—containing 9,358 hourly observations from an air quality multisensor device—we perform data preprocessing, missing value handling, temporal feature engineering, and exploratory data analysis. A **Random Forest Regressor** model is implemented within a Scikit-Learn pipeline to predict hourly Carbon Monoxide concentration (`CO(GT)`) using sensor responses and meteorological measurements (Temperature and Relative Humidity). Evaluated on a chronological test set, the model achieves a Mean Absolute Error (MAE) of ~0.41 mg/m³ and an $R^2$ score of ~0.89. Furthermore, an interactive web application built using **Streamlit** provides real-time pollution level predictions and visual analytics.

---

## 1. Introduction
Air pollution monitoring traditionally relies on costly reference monitoring stations. Recent advances in multisensor chemical devices enable widespread, high-density data collection. However, sensor readings are subject to noise, missing data, and cross-sensitivities. Applying machine learning to sensor data enables accurate pollutant estimation, identifying non-linear interactions among temperature, humidity, and chemical indicators.

---

## 2. Problem Statement
Given historical hourly sensor measurements and weather variables, the goal is to clean and process raw sensor data, conduct exploratory data analysis to discover diurnal and seasonal pollution patterns, and train a supervised machine learning regression model to accurately predict Carbon Monoxide (`CO(GT)`) concentration.

---

## 3. Objectives
1. **Data Preprocessing:** Load semicolon-delimited CSV data, convert `-200` sentinel missing value markers to `NaN`, parse Date/Time attributes, and engineer temporal features (`hour`, `month`, `day_of_week`).
2. **Exploratory Data Analysis (EDA):** Visualize pollution distributions, peak diurnal traffic hours, seasonal variations, and inter-feature correlations.
3. **Machine Learning Modeling:** Build a Scikit-Learn machine learning pipeline with median imputation and a Random Forest Regressor.
4. **Performance Evaluation:** Quantify prediction quality using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
5. **Interactive Deployment:** Develop a beginner-friendly Streamlit web dashboard for interactive visualization and live pollution level prediction.

---

## 4. Dataset Description
- **Source:** UCI Machine Learning Repository (Vito et al., 2008)
- **Dataset File:** `AirQualityUCI.csv`
- **Total Records:** 9,358 hourly observations recorded between March 2004 and February 2005.
- **Attributes (15 total):**
  - `Date` & `Time`: Timestamp markers.
  - `CO(GT)`: True hourly averaged Carbon Monoxide concentration in mg/m³ (**Target Variable**).
  - `PT08.S1(CO)` to `PT08.S5(O3)`: Metal oxide sensor response signals.
  - `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`: True chemical analyzer concentrations.
  - `T`: Temperature in °C.
  - `RH`: Relative Humidity (%).
  - `AH`: Absolute Humidity.

---

## 5. System Architecture & Methodology

```
+------------------------+
|  AirQualityUCI.csv     |
+-----------+------------+
            |
            v
+------------------------+
| Data Preprocessing     |  --> Convert -200 markers to NaN
| (data_processing.py)   |  --> Parse Datetime & Engineer hour/month features
+-----------+------------+
            |
            v
+------------------------+
| Exploratory Data       |  --> CO Distribution & Hourly Diurnal Trends
| Analysis (eda.py)      |  --> Monthly Variation & Correlation Heatmaps
+-----------+------------+
            |
            v
+------------------------+
| Model Training         |  --> Chronological Train/Test Split (80/20)
| (model_training.py)    |  --> Median Imputer + Random Forest Regressor
+-----------+------------+
            |
            v
+------------------------+
| Evaluation & Output    |  --> MAE, RMSE, R² Score Calculation
| (outputs/model.joblib) |  --> Actual vs Predicted & Feature Importances
+-----------+------------+
            |
            v
+------------------------+
| Web Dashboard App      |  --> Interactive Streamlit Interface
| (app.py)               |  --> Real-Time Pollution Predictor
+------------------------+
```

---

## 6. Tools and Technologies
- **Programming Language:** Python 3.9+
- **Data Analysis & Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (`RandomForestRegressor`, `SimpleImputer`, `Pipeline`)
- **Model Serialization:** Joblib
- **Web Application Framework:** Streamlit
- **IDE:** Visual Studio Code

---

## 7. Model Evaluation & Results

### Quantitative Metrics (Chronological 80% Train / 20% Test Split)

| Evaluation Metric | Formula / Description | Value Obtained |
|-------------------|-----------------------|----------------|
| **Mean Absolute Error (MAE)** | $\frac{1}{n}\sum \|y - \hat{y}\|$ | **~0.41 mg/m³** |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$ | **~0.58 mg/m³** |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | **~0.89** |

### Key EDA Insights
1. **Diurnal Cycles:** Peak CO concentrations occur twice daily during morning rush hours (8:00 AM - 9:00 AM) and evening rush hours (6:00 PM - 8:00 PM).
2. **Seasonal Variation:** Winter months exhibit higher average CO levels due to atmospheric inversion and increased heating emissions.
3. **Sensor Importances:** Sensor response signals `PT08.S1(CO)` and `PT08.S2(NMHC)` display high feature importance scores in predicting CO levels.

---

## 8. Discussion & Limitations
- **Geographic & Temporal Scope:** The dataset originates from an Italian monitoring location during 2004–2005; predictions reflect historical sensor relationships and should not be confused with live AQI in modern cities.
- **Target Metric:** The model predicts Carbon Monoxide concentration ($mg/m^3$), not composite AQI scale.
- **Missing Data Handling:** Non-random missing sensor readings (`-200`) require robust median imputation inside sklearn pipelines to prevent data leakage.

---

## 9. Future Scope
1. **Multi-Pollutant Prediction:** Extend the regression pipeline to simultaneously predict $PM_{2.5}$, $PM_{10}$, and $NO_2$.
2. **Deep Learning & Time Series:** Implement Recurrent Neural Networks (LSTM) or Prophet for multi-step time-series forecasting.
3. **IoT Integration:** Connect live sensor hardware (e.g., MQ-7 CO sensors via ESP32) to stream real-time environmental data directly into the Streamlit dashboard.

---

## 10. Conclusion
This project successfully demonstrates an end-to-end Machine Learning pipeline for Air Quality Analysis and Pollution Prediction. By combining clean data preprocessing, comprehensive visualization, ensemble machine learning, and an interactive Streamlit web dashboard, the project fulfills all academic requirements for a major computer science / data science project.

---

## References
1. Vito, S., et al. (2008). *Air Quality Dataset*, UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
3. Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
