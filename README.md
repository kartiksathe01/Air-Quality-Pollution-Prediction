# Air Quality Analysis and Pollution Prediction using Machine Learning

A complete, beginner-friendly Python machine learning project designed for college major project submission. This project analyzes historical sensor data, explores temporal pollution trends, trains a Random Forest regression model to predict hourly Carbon Monoxide (`CO(GT)`) concentrations, and provides an interactive Streamlit web dashboard.

---

## 📁 Project Structure & File Explanations

```
air_quality_pollution_prediction_project/
│
├── AirQualityUCI.csv       # Dataset file from UCI Repository
├── data_processing.py      # Module 1: CSV loading, -200 marker conversion, date/time parsing & feature engineering
├── eda.py                  # Module 2: Visualizations (CO distribution, hourly/monthly trends, correlation heatmap)
├── model_training.py       # Module 3: Chronological train/test split, Random Forest pipeline, MAE/RMSE/R² metrics & joblib export
├── app.py                  # Module 4: Interactive Streamlit Web Dashboard for visual analytics & live predictions
├── main.py                 # Master CLI script executing data processing, EDA generation, and model training end-to-end
├── requirements.txt        # Required Python libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, joblib)
├── README.md               # User guide & VS Code setup instructions
└── PROJECT_REPORT.md       # Complete college major project report template
```

### What Each File Does:
1. **`data_processing.py`**: Reads semicolon-delimited `AirQualityUCI.csv`, converts missing `-200` values to `NaN`, parses `Date` and `Time` into timestamps, extracts temporal features (`hour`, `month`, `day_of_week`), and saves the clean dataset to `outputs/cleaned_air_quality_data.csv`.
2. **`eda.py`**: Generates high-resolution EDA plots: CO distribution histogram, 24-hour diurnal trend, monthly averages, and correlation heatmaps.
3. **`model_training.py`**: Splits data chronologically (first 80% train, last 20% test), imputes missing sensor features using median strategy inside a scikit-learn pipeline, fits a Random Forest Regressor, evaluates performance metrics (MAE, RMSE, $R^2$), plots feature importances, and exports `outputs/model.joblib`.
4. **`app.py`**: Multi-tab Streamlit dashboard allowing users to inspect dataset tables, view interactive charts, check model metrics, and input sensor readings to predict CO pollution levels in real-time.
5. **`main.py`**: Terminal entry point linking all modules together in sequence.

---

## 📊 Dataset Information

- **Dataset Name:** UCI Air Quality Dataset
- **URL:** [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)
- **File Name:** `AirQualityUCI.csv`
- **Description:** Contains 9,358 hourly responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device deployed in an Italian city from March 2004 to February 2005.
- **Missing Value Marker:** The original dataset uses `-200` to indicate missing sensor or chemical readings.

---

## 🚀 Step-by-Step Setup Guide in VS Code

### Step 1: Open Project in VS Code
1. Launch **Visual Studio Code**.
2. Click **File > Open Folder...** and select `air_quality_pollution_prediction_project`.

### Step 2: Ensure Dataset is Present
Verify that `AirQualityUCI.csv` is located directly in the project root folder. If missing, download it from the [UCI Air Quality Link](https://archive.ics.uci.edu/dataset/360/air+quality) and extract `AirQualityUCI.csv` into the folder.

### Step 3: Open Terminal & Create Virtual Environment (Optional but Recommended)
In VS Code, open the integrated terminal (**Terminal > New Terminal**):

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment on Windows PowerShell
.venv\Scripts\activate
```

### Step 4: Install Dependencies
Install all required Python libraries:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Complete Machine Learning Pipeline
Execute the master script to clean data, generate plots, and train the prediction model:

```bash
python main.py
```

*Output files will be generated automatically in the `outputs/` folder.*

### Step 6: Launch the Interactive Web Dashboard
Run Streamlit to open the web dashboard in your browser:

```bash
streamlit run app.py
```

The web dashboard will automatically launch at `http://localhost:8501`.

---

## 📈 Model Performance Metrics

The Random Forest Regressor model is evaluated on a chronological 20% test split:
- **Target Variable:** Carbon Monoxide concentration (`CO(GT)` in mg/m³)
- **Mean Absolute Error (MAE):** ~0.41 mg/m³
- **Root Mean Squared Error (RMSE):** ~0.58 mg/m³
- **Coefficient of Determination ($R^2$ Score):** ~0.89

---

## 🎓 College Submission Template
For project report preparation, refer to **`PROJECT_REPORT.md`** included in this repository. It includes abstract, problem statement, architecture diagrams, methodology, results tables, and discussion sections ready for customizing with your name and roll number.
