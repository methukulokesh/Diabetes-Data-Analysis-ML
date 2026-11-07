
# Healthcare AI Assistant – Diabetes Prediction and Analysis

## 1. Project Overview

The Healthcare AI Assistant is a machine learning-based application developed to estimate the likelihood of diabetes using patient health information.

The project covers data analysis, preprocessing, classification model development, evaluation, hyperparameter tuning, probability calibration and an interactive Streamlit dashboard.

## 2. Project Objectives

- Analyze healthcare data using Exploratory Data Analysis.
- Identify and handle invalid zero measurements.
- Prepare data for machine learning.
- Train and compare classification algorithms.
- Improve model performance through hyperparameter tuning.
- Develop an interactive prediction dashboard.
- Deploy the application using Streamlit Community Cloud.

## 3. Dataset Description

Dataset: Pima Indians Diabetes Dataset

- Total records: 768
- Total columns: 9
- Input features: 8
- Target variable: Outcome

### Features

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

Outcome:
- 0: Diabetes-negative class
- 1: Diabetes-positive class

## 4. Exploratory Data Analysis

EDA was performed using Pandas and Matplotlib.

Activities:
- Dataset inspection.
- Missing-value analysis.
- Duplicate detection.
- Statistical summary.
- Glucose distribution histogram.
- Insulin boxplot.
- Correlation analysis and heatmap.

### Findings

- Records: 768
- Columns: 9
- Conventional missing values: 0
- Duplicate records: 0

Zero values in selected medical measurement columns were investigated as potentially missing measurements.

## 5. Data Cleaning and Preprocessing

The following operations were performed:

- Identified suspicious zero values.
- Replaced selected invalid zeros with missing values.
- Split data into training and testing sets.
- Used an 80:20 train-test split.
- Applied median imputation.
- Applied StandardScaler for the scaled model datasets.
- Avoided fitting preprocessing parameters on the test data.

Dataset split:
- Training records: 614
- Testing records: 154

## 6. Machine Learning Algorithms

Three classification algorithms were implemented.

### Logistic Regression

Accuracy: 70.78%

### Decision Tree

Accuracy: 78.57%

### Random Forest

Accuracy: 74.03%

## 7. Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% |
| Decision Tree | 78.57% | 69.81% | 68.52% | 69.16% |
| Random Forest | 74.03% | 65.91% | 53.70% | 59.18% |

Decision Tree performed best across the four reported metrics in the initial comparison.

## 8. Hyperparameter Tuning

GridSearchCV and StratifiedKFold were used to tune the Decision Tree classifier.

- Cross-validation folds: 5
- Selection metric: Recall
- Parameters investigated:
  - Maximum tree depth
  - Minimum samples split
  - Minimum samples leaf
  - Splitting criterion

The tuned model was evaluated on the held-out test dataset.

The tuned model's test accuracy was approximately 75%, with recall approximately 46%.

The tuning results did not outperform the initial Decision Tree on the reported test metrics.

## 9. Probability Calibration

CalibratedClassifierCV with sigmoid calibration was tested.

Brier score comparison:

| Model | Brier Score |
|---|---:|
| Original model | 0.1809 |
| Calibrated model | 0.1756 |

The calibrated model showed a small improvement in Brier score.

Its recall was 35.19%, so it was retained as an experimental model rather than automatically selected as the main model.

## 10. Streamlit Application

An interactive dashboard was developed using Streamlit.

Features:
- Patient health input fields.
- Diabetes classification.
- Estimated probability display.
- Input validation.
- Medical disclaimer.

The application was tested with positive, negative and invalid input cases.

## 11. Deployment

The application was deployed using Streamlit Community Cloud.

Live application:

https://healthcare-diabetes-assistant.streamlit.app/

GitHub repository:

https://github.com/methukulokesh/Diabetes-Data-Analysis-ML

## 12. Limitations

- The dataset contains only 768 records.
- Model performance is limited by the available training data.
- Decision Tree probabilities may be extreme.
- Calibration results require further validation.
- The application is not clinically validated.

## 13. Future Improvements

- Evaluate additional classification algorithms.
- Improve recall while controlling false positives.
- Perform more extensive model validation.
- Add SHAP-based model explainability.
- Improve probability calibration.
- Expand dashboard visualizations.

## 14. Conclusion

The Healthcare AI Assistant project demonstrates a complete machine learning workflow, from dataset inspection and preprocessing to model development, evaluation and cloud deployment.

Three classification algorithms were compared, and Decision Tree achieved the best initial test performance among them.

The final application provides an interactive demonstration of diabetes prediction using machine learning.

This project is intended for educational purposes and must not be used as a medical diagnostic system.
