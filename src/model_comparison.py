
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load datasets
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

train_scaled = pd.read_csv("data/processed/train_scaled.csv")
test_scaled = pd.read_csv("data/processed/test_scaled.csv")

# Separate features and target
X_train = train.drop("Outcome", axis=1)
y_train = train["Outcome"]

X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]

X_train_scaled = train_scaled.drop("Outcome", axis=1)
X_test_scaled = test_scaled.drop("Outcome", axis=1)

# Define three models
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        X_train_scaled,
        X_test_scaled
    ),

    "Decision Tree": (
        DecisionTreeClassifier(max_depth=4, random_state=42),
        X_train,
        X_test
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ),
        X_train,
        X_test
    )
}

results = []

print("\n===== THREE MODEL COMPARISON =====")

for name, (model, x_train, x_test) in models.items():

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    })

# Create comparison table
results_df = pd.DataFrame(results)

print("\n", results_df.round(4).to_string(index=False))

# Find best model based on recall
best_model = results_df.loc[results_df["Recall"].idxmax()]

print("\n===== BEST MODEL BASED ON RECALL =====")
print("Model:", best_model["Model"])
print("Recall:", round(best_model["Recall"], 4))

# Save comparison results
results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nComparison saved to outputs/model_comparison.csv")
