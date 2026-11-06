
import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/raw/diabetes.csv")

# Create outputs directory if it does not exist
os.makedirs("outputs", exist_ok=True)


# ==========================================
# 2. DATASET OVERVIEW
# ==========================================

print("\n===== DATASET OVERVIEW =====")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()


# ==========================================
# 3. STATISTICAL SUMMARY
# ==========================================

print("\n===== STATISTICAL SUMMARY =====")

print(df.describe())


# ==========================================
# 4. GLUCOSE DISTRIBUTION
# ==========================================

print("\n===== GLUCOSE DISTRIBUTION =====")

plt.figure(figsize=(8, 5))

plt.hist(
    df["Glucose"],
    bins=20,
    edgecolor="black"
)

plt.title("Glucose Distribution")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig(
    "outputs/glucose_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 5. INSULIN OUTLIER ANALYSIS
# ==========================================

print("\n===== INSULIN BOXPLOT =====")

plt.figure(figsize=(8, 5))

plt.boxplot(df["Insulin"])

plt.title("Insulin Outlier Detection")
plt.ylabel("Insulin Level")

plt.tight_layout()

plt.savefig(
    "outputs/insulin_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 6. CORRELATION ANALYSIS
# ==========================================

print("\n===== CORRELATION ANALYSIS =====")

# Calculate correlation between numerical columns
correlation = df.corr()

# Display correlation values
print("\nCorrelation Matrix:")
print(correlation)


# ==========================================
# 7. CORRELATION HEATMAP
# ==========================================

print("\n===== CORRELATION HEATMAP =====")

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Healthcare Dataset Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 8. EDA COMPLETION
# ==========================================

print("\n===== EDA COMPLETED SUCCESSFULLY =====")

print("Total Records:", df.shape[0])
print("Total Features:", df.shape[1])

print("\nGenerated Visualizations:")

print("1. outputs/glucose_distribution.png")
print("2. outputs/insulin_boxplot.png")
print("3. outputs/correlation_heatmap.png")

print("\nAll EDA visualizations saved successfully!")
