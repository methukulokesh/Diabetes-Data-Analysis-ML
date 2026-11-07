
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Keep test data separate
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Pipeline: imputation + Decision Tree
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(random_state=42))
])

# Hyperparameter combinations
param_grid = {
    "model__max_depth": [3, 4, 5, 6, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__criterion": ["gini", "entropy"]
}

# Five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Search for the best model
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1
)

print("\n===== MODEL TUNING STARTED =====")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate on untouched test data
y_pred = best_model.predict(X_test)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-validation Recall:")
print(round(grid_search.best_score_, 4))

print("\nTest Accuracy:")
print(round(accuracy_score(y_test, y_pred), 4))

print("\nTest Recall:")
print(round(recall_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save best pipeline
joblib.dump(
    best_model,
    MODEL_DIR / "best_diabetes_model.joblib"
)

print("\nBest model saved successfully!")
print("Location: models/best_diabetes_model.joblib")
