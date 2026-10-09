# MAJOR PROJECT REPORT
## Air Quality Analysis and Pollution Prediction Using Machine Learning

**Student Name:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**Department:** [Department]  
**Subject:** [Subject Name]  
**Academic Year:** 2026–2027

---

## Abstract
Air pollution is an important environmental concern. This project uses a public historical air-quality dataset to explore pollutant measurements, visualize temporal patterns, and demonstrate machine-learning regression. After cleaning missing-value markers and preparing time-related features, a Random Forest Regressor estimates hourly carbon monoxide concentration (`CO(GT)`) from other sensor, pollutant, and environmental variables. Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²). The project demonstrates a reproducible data-science workflow and highlights the limitations of historical sensor data.

## 1. Introduction
Air-quality measurements can be analyzed to understand how pollutant concentrations vary over time and how measured variables relate to one another. Data visualization helps summarize these patterns, while regression models can estimate a continuous measurement from other available features. This project applies these techniques to the UCI Air Quality dataset.

## 2. Problem Statement
To analyze historical air-quality data, identify temporal patterns and relationships among measured variables, and train a machine-learning model to estimate hourly CO concentration.

## 3. Objectives
1. Load and inspect the air-quality dataset.
2. Handle missing-value markers and prepare date/time features.
3. Visualize CO concentration distribution, hourly and monthly trends, and correlations.
4. Train a Random Forest regression model.
5. Evaluate the model using MAE, RMSE, and R².
6. Discuss the reliability, limitations, and possible extensions of the approach.

## 4. Dataset Description
The project uses `AirQualityUCI.csv` from the UCI Machine Learning Repository. The dataset contains 9,358 hourly records collected from March 2004 to February 2005 by an air-quality chemical multisensor device in a polluted area of an Italian city. It includes pollutant concentrations, sensor responses, temperature, and humidity-related variables. The value `-200` is used in the original data to represent missing readings.

**Source:** Vito, S. (2008). *Air quality*. UCI Machine Learning Repository.  
https://archive.ics.uci.edu/dataset/360/air+quality

## 5. Tools and Technologies
- Python
- Visual Studio Code
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## 6. Methodology
1. Load the semicolon-separated CSV file.
2. Remove empty columns and rows.
3. Convert numeric measurements and replace `-200` with missing values.
4. Parse date and time into a timestamp and derive hour, month, and day-of-week features.
5. Explore the data with six visualizations.
6. Select `CO(GT)` as the continuous prediction target.
7. Use a chronological split: the first 80% of valid target records for training and the final 20% for testing.
8. Impute missing input features using training-set medians through a pipeline.
9. Train a Random Forest Regressor.
10. Evaluate predictions using MAE, RMSE, and R².

## 7. System Architecture
CSV Dataset → Data Cleaning → Date/Time Processing → Exploratory Data Analysis → Feature Preparation → Chronological Train/Test Split → Random Forest Regression → Evaluation → Graphs and Report

## 8. Results
**Fill this section using the actual output after running `python main.py`. Do not invent metrics.**

- Rows after timestamp parsing: copy from `outputs/dataset_summary.txt`
- Mean CO concentration: copy from `outputs/dataset_summary.txt`
- MAE: copy from `outputs/model_results.txt`
- RMSE: copy from `outputs/model_results.txt`
- R²: copy from `outputs/model_results.txt`
- Attach all six PNG charts from the `outputs` folder.
- Explain what the hourly and monthly trend plots show in your run.
- Mention the top features shown in `06_feature_importance.png`.

## 9. Discussion
The distribution plot summarizes the range of measured CO values. Hourly and monthly plots help explore variation across time. The correlation matrix summarizes linear relationships among numeric measurements. Actual-versus-predicted values and the reported regression metrics describe model performance on the held-out later portion of the data. Feature importance describes which input variables the fitted Random Forest used most, but does not establish causation.

## 10. Limitations
- The dataset is historical and was collected in an Italian city during 2004–2005.
- This model predicts CO concentration, not official AQI.
- Sensor drift and missing data may affect model quality.
- Results may not generalize to other cities, years, or monitoring systems.
- The project is an educational demonstration, not a real-time public warning service.

## 11. Future Scope
- Evaluate additional regression algorithms.
- Tune hyperparameters with time-aware validation.
- Compare models using additional cities or newer datasets.
- Build a dashboard for interactive historical data exploration.
- If official AQI prediction is required, use a dataset with documented AQI labels and the relevant pollutant standards.

## 12. Conclusion
This project demonstrates how Python can be used to clean, explore, visualize, and model historical air-quality data. A Random Forest regression model is evaluated with standard regression metrics. The resulting graphs and metrics provide a practical example of data analysis and machine learning, while the dataset's age and geographic coverage limit its use for current local pollution assessment.

## References
1. Vito, S. (2008). *Air quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F
2. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/360/air+quality
3. Scikit-learn documentation: https://scikit-learn.org/stable/
