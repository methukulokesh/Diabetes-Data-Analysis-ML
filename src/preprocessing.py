
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("\n===== TRAIN-TEST SPLIT =====")
print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# Create median imputer
imputer = SimpleImputer(strategy="median")

# Fit only on training data
X_train_imputed = imputer.fit_transform(X_train)

# Apply training medians to test data
X_test_imputed = imputer.transform(X_test)

# Convert arrays back to DataFrames
X_train_imputed = pd.DataFrame(
    X_train_imputed,
    columns=X.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    X_test_imputed,
    columns=X.columns,
    index=X_test.index
)

# Reattach target column
train_data = X_train_imputed.copy()
train_data["Outcome"] = y_train

test_data = X_test_imputed.copy()
test_data["Outcome"] = y_test

# Save prepared datasets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\n===== IMPUTATION RESULTS =====")
print("Training missing values:", int(X_train_imputed.isnull().sum().sum()))
print("Testing missing values:", int(X_test_imputed.isnull().sum().sum()))

print("\n===== PREPROCESSING COMPLETED =====")
print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)
print("Saved: data/processed/train.csv")
print("Saved: data/processed/test.csv")


# ===== FEATURE SCALING =====

print("\n===== FEATURE SCALING =====")

# Create scaler
scaler = StandardScaler()

# Learn scaling parameters from training data only
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Apply the same scaling parameters to test data
X_test_scaled = scaler.transform(X_test_imputed)

# Convert scaled arrays into DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index
)

# Attach target column
train_scaled = X_train_scaled.copy()
train_scaled["Outcome"] = y_train

test_scaled = X_test_scaled.copy()
test_scaled["Outcome"] = y_test

# Save scaled datasets separately
train_scaled.to_csv(
    OUTPUT_DIR / "train_scaled.csv",
    index=False
)

test_scaled.to_csv(
    OUTPUT_DIR / "test_scaled.csv",
    index=False
)

print("Training scaled shape:", train_scaled.shape)
print("Testing scaled shape:", test_scaled.shape)

print("\nTraining feature means:")
print(X_train_scaled.mean().round(2))

print("\n===== FEATURE SCALING COMPLETED =====")
