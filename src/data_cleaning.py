
import pandas as pd
from pathlib import Path

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE)

print("\n===== DATA CLEANING STARTED =====")
print("Original dataset shape:", df.shape)

# Columns where zero is treated as a missing measurement
invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Check zero values before cleaning
print("\n===== ZERO VALUES BEFORE CLEANING =====")

for column in invalid_zero_columns:
    print(f"{column}: {(df[column] == 0).sum()}")

# Check and remove duplicate records
print("\nDuplicate records:", df.duplicated().sum())
df = df.drop_duplicates()

# Replace invalid zeros with NaN
df[invalid_zero_columns] = df[invalid_zero_columns].replace(
    0, float("nan")
)

# Display missing values after replacement
print("\n===== MISSING VALUES AFTER CLEANING =====")
print(df.isnull().sum())

# Missing value percentage
print("\n===== MISSING VALUE PERCENTAGE =====")

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": missing_percentage
})

print(missing_summary)

# Compare mean and median of affected columns
print("\n===== MEAN VS MEDIAN =====")

for column in invalid_zero_columns:
    print(f"\n{column}")
    print("Mean:", df[column].mean())
    print("Median:", df[column].median())

# Create processed folder if it doesn't exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n===== DATA CLEANING COMPLETED =====")
print("Final dataset shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())
print("Saved file:", OUTPUT_FILE)
