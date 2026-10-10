from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Define paths and load the dataset
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "credit_default_raw.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Y"])
y = df["Y"]


# --------------------------------------------------
# 2. Create a reproducible train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Build the Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

print("Training Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 6. Calculate evaluation metrics
# --------------------------------------------------

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

print("\nLogistic Regression Evaluation")
print("=" * 40)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


# --------------------------------------------------
# 7. Print the classification report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Default", "Default"],
        zero_division=0
    )
)


# --------------------------------------------------
# 8. Calculate and save the confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"]
)

plt.title("Logistic Regression: Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. Save the evaluation results
# --------------------------------------------------

pd.DataFrame(
    [metrics]
).to_csv(
    OUTPUT_DIR / "logistic_regression_metrics.csv",
    index=False
)

print("\nSaved evaluation metrics to:")
print("outputs/logistic_regression_metrics.csv")

print("\nSaved confusion matrix to:")
print("outputs/logistic_regression_confusion_matrix.png")