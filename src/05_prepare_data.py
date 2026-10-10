from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "credit_default_raw.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate predictors and target
X = df.drop(columns=["Y"])
y = df["Y"]


# --------------------------------------------------
# 2. Create a stratified train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Report split information
# --------------------------------------------------

print("Dataset preparation completed.")
print("-" * 45)

print(f"Total observations: {len(df)}")
print(f"Number of predictors: {X.shape[1]}")

print(f"\nTraining observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

print("\nTarget distribution in the full dataset:")
print(y.value_counts(normalize=True).sort_index().round(4))

print("\nTarget distribution in the training set:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTarget distribution in the test set:")
print(y_test.value_counts(normalize=True).sort_index().round(4))


# --------------------------------------------------
# 4. Define a preprocessing pipeline
# --------------------------------------------------

preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

print("\nPreprocessing pipeline defined:")
print(preprocessor)

print(
    "\nNote: The pipeline has been defined but not fitted yet. "
    "It will be fitted on training data only during model training."
)