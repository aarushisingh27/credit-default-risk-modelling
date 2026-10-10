from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DOWNLOAD DATASET
# ==========================================

print("Downloading the UCI credit-card dataset...")

dataset = fetch_ucirepo(id=350)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

# Save the original dataset for reuse
dataset_path = DATA_DIR / "credit_default_raw.csv"
df.to_csv(dataset_path, index=False)

print(f"\nDataset saved to: {dataset_path}")

print("\nDataset downloaded successfully!")
print(f"Feature dimensions: {X.shape}")
print(f"Target dimensions: {y.shape}")


# ==========================================
# BASIC DATASET INFORMATION
# ==========================================

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- First Five Rows ---")
print(df.head())


# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print("\n--- Target Distribution ---")
print(df["Y"].value_counts().sort_index())

print("\n--- Target Percentages ---")
print(
    (df["Y"].value_counts(normalize=True).sort_index() * 100).round(2)
)


# ==========================================
# DUPLICATE INVESTIGATION
# ==========================================

print("\n--- Duplicate Investigation ---")

exact_duplicates = df.duplicated().sum()
print(f"Exact duplicate rows: {exact_duplicates}")

X_only = df.drop(columns=["Y"])

repeated_predictors = X_only.duplicated(keep=False)
repeated_records = df.loc[repeated_predictors]

print(
    "Rows with repeated predictor values: "
    f"{repeated_predictors.sum()}"
)

target_counts = (
    repeated_records
    .groupby(list(X_only.columns), dropna=False)["Y"]
    .nunique()
)

conflicting_groups = (target_counts > 1).sum()

print(
    "Groups with conflicting target values: "
    f"{conflicting_groups}"
)


# ==========================================
# CATEGORICAL AND ORDINAL-LIKE VARIABLES
# ==========================================

print("\n--- Categorical and Ordinal-Like Variables ---")

for col in ["X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"]:
    print(f"\n{col}:")
    print(df[col].value_counts().sort_index().to_dict())


# ==========================================
# NUMERICAL RANGE CHECKS
# ==========================================

print("\n--- Numerical Range Checks ---")

numerical_columns = [
    "X1", "X5",
    "X12", "X13", "X14", "X15", "X16", "X17",
    "X18", "X19", "X20", "X21", "X22", "X23"
]

for col in numerical_columns:
    print(
        f"{col}: minimum = {df[col].min()}, "
        f"maximum = {df[col].max()}"
    )


# ==========================================
# FINAL AUDIT SUMMARY
# ==========================================

print("\n--- Final Audit Summary ---")
print(f"Records: {df.shape[0]}")
print(f"Predictor variables: {X.shape[1]}")
print(f"Missing values in dataset: {df.isnull().sum().sum()}")
print(f"Exact duplicate rows: {exact_duplicates}")
print(f"Conflicting predictor groups: {conflicting_groups}")

print("\nDataset audit completed.")