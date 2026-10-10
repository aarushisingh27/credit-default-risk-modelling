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
# DESCRIPTIVE STATISTICS
# ==========================================

numerical_columns = [
    "X1", "X5",
    "X12", "X13", "X14", "X15", "X16", "X17",
    "X18", "X19", "X20", "X21", "X22", "X23"
]

print("\n--- Descriptive Statistics ---")

statistics = df[numerical_columns].describe().T
statistics["skewness"] = df[numerical_columns].skew()

print(statistics.round(2))

statistics_path = OUTPUT_DIR / "numerical_descriptive_statistics.csv"
statistics.to_csv(statistics_path)

print(f"\nDescriptive statistics saved to: {statistics_path}")


# ==========================================
# CREDIT LIMIT AND AGE DISTRIBUTIONS
# ==========================================

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(
    data=df,
    x="X1",
    bins=30,
    kde=True,
    ax=axes[0]
)
axes[0].set_title("Distribution of Credit Limit")
axes[0].set_xlabel("Credit Limit")
axes[0].set_ylabel("Number of Customers")

sns.histplot(
    data=df,
    x="X5",
    bins=30,
    kde=True,
    ax=axes[1]
)
axes[1].set_title("Distribution of Customer Age")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Number of Customers")

plt.tight_layout()

distribution_path = OUTPUT_DIR / "credit_limit_age_distributions.png"
plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nCredit limit and age plots saved to: {distribution_path}")


# ==========================================
# BILL AMOUNT DISTRIBUTIONS
# ==========================================

bill_columns = ["X12", "X13", "X14", "X15", "X16", "X17"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for col, ax in zip(bill_columns, axes):
    sns.histplot(
        data=df,
        x=col,
        bins=40,
        ax=ax
    )
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel("Bill Amount")
    ax.set_ylabel("Number of Customers")

plt.tight_layout()

bill_path = OUTPUT_DIR / "bill_amount_distributions.png"
plt.savefig(bill_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Bill amount plots saved to: {bill_path}")


# ==========================================
# PREVIOUS PAYMENT DISTRIBUTIONS
# ==========================================

payment_columns = ["X18", "X19", "X20", "X21", "X22", "X23"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for col, ax in zip(payment_columns, axes):
    sns.histplot(
        data=df,
        x=col,
        bins=40,
        ax=ax
    )
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel("Payment Amount")
    ax.set_ylabel("Number of Customers")

plt.tight_layout()

payment_path = OUTPUT_DIR / "payment_amount_distributions.png"
plt.savefig(payment_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Payment amount plots saved to: {payment_path}")


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\nNumerical analysis completed successfully.")