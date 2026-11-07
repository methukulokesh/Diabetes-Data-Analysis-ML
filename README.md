
# 🏥 Healthcare AI Assistant – Diabetes Prediction System

## 📌 Project Overview

The Healthcare AI Assistant is a Machine Learning-based application that predicts the likelihood of diabetes using patient health information.

The project includes data inspection, exploratory data analysis, data cleaning, preprocessing, machine learning model training, model evaluation, hyperparameter tuning, and an interactive Streamlit dashboard.

## 🎯 Project Objectives

- Analyze healthcare data to identify important patterns.
- Clean and preprocess patient health information.
- Train and compare multiple classification algorithms.
- Evaluate model performance using classification metrics.
- Build an interactive dashboard for diabetes prediction.

## 📊 Dataset Information

- Dataset: Pima Indians Diabetes Dataset
- Total records: 768
- Total columns: 9
- Target variable: Outcome
- 0 = Non-diabetic class
- 1 = Diabetes-positive class

### Dataset Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Patient age |
| Outcome | Target class |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- VS Code
- Git and GitHub

## 🔍 Project Workflow

### 1. Data Inspection and EDA

- Dataset structure and statistical analysis.
- Missing-value and duplicate checking.
- Zero-value investigation.
- Glucose distribution analysis.
- Insulin outlier analysis.
- Correlation heatmap.

### 2. Data Cleaning and Preprocessing

- Replaced invalid zero measurements with missing values.
- Separated features and target variable.
- Split the dataset into 80% training and 20% testing.
- Applied median imputation.
- Applied feature scaling for models that require it.

### 3. Machine Learning Models

The following classification algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest

### 4. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 5. Hyperparameter Tuning

GridSearchCV and StratifiedKFold cross-validation were used to tune the Decision Tree model.

Recall was used as the model-selection metric to give attention to identifying diabetes-positive cases.

### 6. Interactive Dashboard

A Streamlit application was developed to:

- Accept patient health information.
- Generate a diabetes prediction.
- Display estimated diabetes probability.
- Show the predicted classification.

## 📈 Model Comparison

Results from the initial model comparison:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% |
| Decision Tree | 78.57% | 69.81% | 68.52% | 69.16% |
| Random Forest | 74.03% | 65.91% | 53.70% | 59.18% |

The Decision Tree achieved the highest recall and accuracy among these initial comparisons.

Note: These are initial comparison results. Hyperparameter-tuned model results are evaluated separately.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/methukulokesh/Diabetes-Data-Analysis-ML.git
```

### 2. Open the project directory

```bash
cd Diabetes-Data-Analysis-ML
```

### 3. Install dependencies

```bash
py -3.13 -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
py -3.13 -m streamlit run app.py
```

### 5. Open in browser

```text
http://localhost:8501
```

## 📁 Project Structure

```text
Healthcare-AI-Assistant/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_diabetes_model.joblib
│
├── notebooks/
│
├── outputs/
│   ├── correlation_heatmap.png
│   ├── glucose_distribution.png
│   ├── insulin_boxplot.png
│   ├── model_comparison.csv
│   └── performance_comparison.png
│
├── reports/
│   ├── day1/
│   ├── day2/
│   └── day3/
│
├── src/
│   ├── inspect_data.py
│   ├── eda_analysis.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── model_comparison.py
│   ├── model_evaluation.py
│   └── model_tuning.py
│
├── app.py
├── NOTES.md
├── README.md
└── requirements.txt
```

## ⚠️ Limitations

- The dataset is relatively small.
- Model predictions depend on the quality and representativeness of the training data.
- Model probabilities are estimates and may not be well calibrated.
- This application is an educational project and must not be used as a medical diagnostic tool.

## 🔮 Future Improvements

- Improve model performance and probability calibration.
- Add additional classification algorithms.
- Develop more detailed healthcare visualizations.
- Add explainable AI techniques such as SHAP.

- Add model explainability using SHAP.
## 👨‍💻 Author

**B.Tech Computer Science and Engineering – Data Science**

GitHub: https://github.com/methukulokesh/Diabetes-Data-Analysis-ML
