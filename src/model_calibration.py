
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_diabetes_model.joblib"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "calibrated_diabetes_model.joblib"

# Load dataset and existing model
df = pd.read_csv(DATA_FILE)
base_model = joblib.load(MODEL_FILE)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Recreate the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== PROBABILITY CALIBRATION =====")

# Calibrate using training data only
calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

# Evaluate on untouched test data
y_pred = calibrated_model.predict(X_test)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("Brier Score:", round(brier_score_loss(y_test, y_prob), 4))

# Save calibrated model separately
joblib.dump(calibrated_model, OUTPUT_MODEL)

print("\nCalibrated model saved successfully!")
print("Location:", OUTPUT_MODEL)
# Compare original and calibrated probability quality

original_probabilities = base_model.predict_proba(X_test)[:, 1]

original_brier = brier_score_loss(
    y_test,
    original_probabilities
)

calibrated_brier = brier_score_loss(
    y_test,
    y_prob
)

print("\n===== BRIER SCORE COMPARISON =====")
print("Original model:", round(original_brier, 4))
print("Calibrated model:", round(calibrated_brier, 4))