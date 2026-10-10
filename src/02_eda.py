from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_default_raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ==========================================
# TARGET VARIABLE ANALYSIS
# ==========================================

print("\n--- Target Distribution ---")

target_counts = df["Y"].value_counts().sort_index()
target_percentages = (
    df["Y"].value_counts(normalize=True).sort_index() * 100
)

print(target_counts)
print("\nTarget percentages:")
print(target_percentages.round(2))


# ==========================================
# TARGET DISTRIBUTION CHART
# ==========================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="Y",
    hue="Y",
    order=[0, 1],
    palette="Set2",
    legend=False
)

ax.set_title("Distribution of Credit Card Default")
ax.set_xlabel("Default Next Month (0 = No, 1 = Yes)")
ax.set_ylabel("Number of Customers")

for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f"{int(height):,}",
        (bar.get_x() + bar.get_width() / 2, height),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points"
    )

plt.tight_layout()

chart_path = OUTPUT_DIR / "target_distribution.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nChart saved to: {chart_path}")