
# Healthcare AI Assistant – Technical Learning Notes

These notes document the concepts, techniques, Python functions, dataset observations and interview questions learned while developing the Healthcare AI Assistant project.

# Table of Contents

| Day | Topic | Status |
|---|---|---|
| Day 1 | Dataset Inspection and Exploratory Data Analysis (EDA) | Completed |
| Day 2 | Data Cleaning and Preprocessing | Completed |
| Day 3 | Feature Engineering | Upcoming |
| Day 4 | Machine Learning Algorithms | Upcoming |
| Day 5 | Model Training and Evaluation | Upcoming |
| Day 6 | Model Comparison and Optimization | Upcoming |
| Day 7 | Healthcare AI Dashboard | Upcoming |

---

# Day 1 – Dataset Inspection and EDA

## 1. What is EDA?

Exploratory Data Analysis (EDA) is the process of examining a dataset to understand its structure, patterns, distributions and quality before applying machine learning.

## 2. Libraries Used

- Pandas: Data manipulation and analysis.
- Matplotlib: Data visualization.
- NumPy: Numerical computations.

## 3. Important Pandas Functions

| Function | Purpose |
|---|---|
| pd.read_csv() | Reads CSV data |
| df.shape | Returns rows and columns |
| df.head() | Displays first five records |
| df.columns | Returns column names |
| df.info() | Shows dataset structure |
| df.describe() | Generates statistical summary |
| df.isnull().sum() | Counts missing values |
| df.duplicated().sum() | Counts duplicate records |

## 4. Dataset Information

- Total rows: 768
- Total columns: 9
- Conventional missing values: 0
- Duplicate records: 0

## 5. Important Learning – Invalid Zero Values

A dataset can have zero missing values but still contain invalid values.

For example, zero Insulin or zero BMI readings may represent missing or incorrectly recorded medical measurements.

Therefore, missing-value analysis and invalid-zero analysis are different.

## 6. Interview Question

**Q: What is the purpose of EDA?**

Answer: EDA helps understand dataset structure, identify missing values, detect outliers, discover patterns and prepare data for machine learning.

## 7. Statistical Analysis

The df.describe() function generates descriptive statistics.

Important concepts:

- Mean: Average of all values.
- Median: Middle value in sorted data.
- Standard deviation: Measures data variation.
- Minimum: Smallest value.
- Maximum: Largest value.
- Quartiles: Divide data into four equal parts.

### Observations

- Insulin mean = 79.79
- Insulin median = 30.50
- BMI mean = 31.99
- Glucose mean = 120.89

The difference between mean and median can indicate skewness or the influence of extreme observations.

Statistical summaries help identify unusual values before data cleaning and model training.

## 8. Glucose Distribution – Histogram

**Concept:**

A histogram is a graphical representation used to understand the frequency distribution of numerical data.

**Python function:**

`plt.hist()`

**Parameters:**

- `df["Glucose"]`: Selects the Glucose column.
- `bins=20`: Divides the data into 20 intervals.
- `edgecolor="black"`: Adds borders to bars.

**Graph interpretation:**

1. Most glucose measurements are concentrated around 90–130.
2. A few records contain zero glucose values.
3. The distribution has a tail towards higher glucose values.

**Why is this useful?**

- Understand data distribution.
- Identify unusual values.
- Detect possible data-quality problems.
- Prepare data for machine learning.

**Interview question: Why do we use histograms in EDA?**

Answer: Histograms help us understand the frequency distribution, spread, concentration and shape of numerical variables.

## 9. Boxplot and Outlier Detection

### What is a Boxplot?

A boxplot is a statistical visualization used to identify the distribution, median, spread and potential outliers in numerical data.

### Important Concepts

1. Median: Middle value of the dataset.
2. Q1: 25th percentile.
3. Q3: 75th percentile.
4. IQR: Interquartile Range = Q3 - Q1.
5. Outliers: Values outside the usual range, identified using the 1.5 × IQR rule.

### Python Code

```python
plt.boxplot(df["Insulin"])
```

### Dataset Findings

- Insulin median: 30.5
- Q1: 0
- Q3: 127.25
- Maximum: 846
- Zero values: 374 (48.70%)
- Numerous potential high-value outliers.

### Conclusion

The Insulin column contains a large number of zero values and several unusually high observations. These require further investigation during data cleaning.

### Interview Question

**Q: What is an outlier?**

Answer: An outlier is a data point that differs significantly from other observations in a dataset. It can be detected using methods such as boxplots and the IQR method.

## 10. Correlation Analysis and Heatmap

### What is Correlation?

Correlation measures the strength and direction of the relationship between two numerical variables.

### Types of Correlation

1. Positive Correlation: Both variables tend to increase together.
2. Negative Correlation: One variable tends to increase when the other decreases.
3. Zero or Weak Correlation: There is little or no linear relationship.

### Python Functions

- df.corr(): Calculates correlation coefficients.
- plt.imshow(): Displays matrix data as an image.
- plt.colorbar(): Displays the correlation scale.
- plt.savefig(): Saves the visualization.

### Actual Dataset Observations

Correlation with the Outcome column:

- Glucose: 0.4666
- BMI: 0.2927
- Age: 0.2384
- Pregnancies: 0.2219

Glucose has the strongest positive linear correlation with the diabetes Outcome.

**Note:** These are preliminary correlations calculated on the original dataset, before invalid zero values were replaced.

### Important Learning

Correlation does not imply causation.

A strong correlation does not prove that one variable causes another.

### Interview Question

**Q: What is a correlation heatmap?**

Answer: A correlation heatmap is a graphical representation of relationships between numerical variables using different colours to represent correlation strength and direction.

---

# Day 2 – Data Cleaning and Preprocessing

## 1. What is Data Preprocessing?

Data preprocessing is the process of preparing raw data before training a Machine Learning model.

It includes:

- Handling missing values.
- Removing duplicate records.
- Separating features and target.
- Splitting data into training and testing sets.
- Feature scaling.

## 2. Handling Invalid Zero Values

In our diabetes dataset, zero values in certain medical measurement columns are conventionally treated as missing measurements.

Affected columns:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

We replaced these zero values with NaN (Not a Number).

Important:

- Pregnancies = 0 is valid.
- Outcome = 0 is valid.
- Original raw dataset was preserved.

**Python function:**

`replace(0, float("nan"))`

## 3. Missing Value Analysis

After replacing invalid zeros:

| Column | Missing Values |
|---|---:|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |
| Total | 652 |

The dataset contains 768 records and 9 columns.

### Important Learning

Missing-value percentage helps identify how much information is unavailable in each feature.

Insulin has the highest missing percentage, approximately 48.70%.

## 4. Train-Test Split

Train-test splitting divides a dataset into two parts:

- Training data: Used to train the machine learning model.
- Testing data: Used to evaluate model performance on unseen data.

We used an 80:20 split.

### Actual Dataset Results

| Dataset | Records |
|---|---:|
| Training data | 614 |
| Testing data | 154 |
| Total | 768 |

**Python function:**

`train_test_split()`

We used `stratify=y` to maintain approximately the same target-class proportions in both datasets.

## 5. Median Imputation

Median imputation replaces missing values with the median of the respective feature.

**Library:**

`sklearn.impute.SimpleImputer`

**Python code:**

```python
imputer = SimpleImputer(strategy="median")
```

### Why Use Median?

- Less sensitive to extreme values than the mean.
- Useful for skewed numerical data.
- Helps retain records instead of deleting them.

### Important Functions

- `fit_transform()`: Learns median values from training data and fills missing values.
- `transform()`: Applies the learned median values to testing data.

The imputer was fitted only on training data.

## 6. Feature Scaling

Feature scaling brings numerical features to comparable scales.

For example:

- Glucose has values up to 199.
- Insulin has values up to 846.
- BMI has values up to 67.1.

Different numerical ranges can affect distance-based and gradient-based algorithms.

**Library:**

`sklearn.preprocessing.StandardScaler`

**Python code:**

```python
scaler = StandardScaler()
```

### Standardization Formula

Z = (X - Mean) / Standard Deviation

After standardization:

- Mean is approximately 0.
- Standard deviation is approximately 1.

### Important Functions

- `fit_transform()`: Learns scaling parameters from training data and scales it.
- `transform()`: Applies the same scaling parameters to testing data.

The scaler was fitted only on training data.

### Algorithms That Benefit From Scaling

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Decision Trees and Random Forests generally do not require feature scaling.

## 7. Data Leakage

Data leakage occurs when information from testing data influences model training.

We prevented it by:

1. Fitting the imputer only on training data.
2. Fitting the scaler only on training data.
3. Applying learned transformations to testing data.

This ensures that the test dataset remains independent during preprocessing.

## 8. Generated Files

| File | Purpose |
|---|---|
| `diabetes_cleaned.csv` | Dataset with invalid zeros replaced by NaN |
| `train.csv` | Imputed training dataset |
| `test.csv` | Imputed testing dataset |
| `train_scaled.csv` | Scaled training dataset |
| `test_scaled.csv` | Scaled testing dataset |

## 9. Interview Questions

**Q1. What is data preprocessing?**

Answer: Data preprocessing is the process of cleaning, transforming and preparing raw data before training a machine learning model.

**Q2. Why use median imputation?**

Answer: The median is less affected by extreme values than the mean and is useful for skewed numerical data.

**Q3. What is StandardScaler?**

Answer: StandardScaler standardizes numerical features to approximately zero mean and unit variance.

**Q4. Why split data into training and testing sets?**

Answer: Training data is used to train a model, while testing data evaluates its performance on unseen data.

**Q5. What is data leakage?**

Answer: Data leakage occurs when information that should be unavailable during training influences model development.

**Q6. Why fit the imputer and scaler only on training data?**

Answer: To prevent information from the test dataset influencing the training process.

---

# Future Learning Sections

## Day 3 – Feature Engineering

Topics to be covered:

- Feature selection.
- Feature creation.
- Feature importance.
- Handling categorical features if applicable.

## Day 4 – Machine Learning Algorithms

Topics to be covered:

- Logistic Regression.
- Decision Tree.
- Random Forest.
- Support Vector Machine.
- K-Nearest Neighbors.

## Day 5 – Model Training and Evaluation

Topics to be covered:

- Model training.
- Predictions.
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.

## Day 6 – Model Comparison and Optimization

Topics to be covered:

- Comparing multiple algorithms.
- Cross-validation.
- Hyperparameter tuning.
- Selecting the best-performing model.

## Day 7 – Healthcare AI Dashboard

Topics to be covered:

- Streamlit application.
- User input interface.
- Diabetes prediction.
- Model results visualization.
- Dashboard deployment.
