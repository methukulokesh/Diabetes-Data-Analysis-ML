# import pandas as pd
# df = pd.read_csv("data/raw/diabetes.csv")
# print(df.head())
# print(df.shape())
# print(df.columns.tolist())
import pandas as pd

# Load diabetes dataset
df = pd.read_csv("data/raw/diabetes.csv")

# Display first five records
print(df.head())

# Display number of rows and columns
print(df.shape)

# Display column names
print(df.columns.tolist())
# Dataset information
print("\n===== DATASET INFORMATION =====")
df.info()

# Check missing values
print("\n===== MISSING VALUE ANALYSIS =====")
print(df.isnull().sum())
# Check zero values in each column
print("\n===== ZERO VALUE ANALYSIS =====")
print((df == 0).sum())
# Check duplicate records
print("\n===== DUPLICATE ANALYSIS =====")
print("Duplicate rows:", df.duplicated().sum())

# Identify columns where zero may represent missing measurements

# Investigate suspicious zero values
zero_check_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\n===== INVALID ZERO VALUE INVESTIGATION =====")

for column in zero_check_columns:
    zero_count = (df[column] == 0).sum()
    percentage = (zero_count * 100.0) / df.shape[0]

    print("{}: {} ({:.2f}%)".format(
        column, zero_count, percentage
    ))

# Statistical summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
