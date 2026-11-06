
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "outputs" / "model_comparison.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model comparison results
df = pd.read_csv(INPUT_FILE)

print("\n===== MODEL EVALUATION =====")
print(df)

# 1. Compare model accuracy
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Accuracy"])

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300)
plt.show()

# 2. Compare Precision, Recall and F1-score
metrics = ["Precision", "Recall", "F1 Score"]

df.plot(
    x="Model",
    y=metrics,
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Metrics Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "performance_comparison.png", dpi=300)
plt.show()

print("\nEvaluation graphs saved successfully!")
