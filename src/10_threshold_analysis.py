from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. Define project paths and load data
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "credit_default_raw.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Y"])
y = df["Y"]


# --------------------------------------------------
# 2. Create the original train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Split training data into training and validation
# --------------------------------------------------

X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("Data split completed.")
print(f"Subtraining observations: {len(X_subtrain)}")
print(f"Validation observations: {len(X_val)}")
print(f"Final test observations: {len(X_test)}")


# --------------------------------------------------
# 4. Train QDA using only the subtraining data
# --------------------------------------------------

qda_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            QuadraticDiscriminantAnalysis(reg_param=0.1)
        )
    ]
)

print("\nTraining QDA on subtraining data...")
qda_model.fit(X_subtrain, y_subtrain)

print("Training completed.")


# --------------------------------------------------
# 5. Obtain validation probabilities
# --------------------------------------------------

validation_probabilities = qda_model.predict_proba(X_val)[:, 1]


# --------------------------------------------------
# 6. Evaluate candidate thresholds on validation data
# --------------------------------------------------

thresholds = [
    0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80
]

validation_results = []

for threshold in thresholds:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    validation_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_val, validation_predictions
        ),
        "Precision": precision_score(
            y_val, validation_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, validation_predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_val, validation_predictions, zero_division=0
        )
    })

validation_results = pd.DataFrame(validation_results)

print("\nVALIDATION RESULTS")
print("=" * 75)
print(validation_results.round(4).to_string(index=False))


# --------------------------------------------------
# 7. Select a threshold using validation F1-score
# --------------------------------------------------

best_index = validation_results["F1-score"].idxmax()
best_threshold = validation_results.loc[best_index, "Threshold"]

print(f"\nSelected threshold: {best_threshold:.2f}")
print(
    "Selection criterion: highest F1-score on validation data."
)


# Save validation results
validation_results.to_csv(
    OUTPUT_DIR / "qda_threshold_validation.csv",
    index=False
)


# --------------------------------------------------
# 8. Plot validation precision and recall
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    validation_results["Threshold"],
    validation_results["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    validation_results["Threshold"],
    validation_results["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    validation_results["Threshold"],
    validation_results["F1-score"],
    marker="o",
    label="F1-score"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("QDA Threshold Analysis on Validation Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "qda_threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. Retrain QDA on the complete training set
# --------------------------------------------------

final_qda = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            QuadraticDiscriminantAnalysis(reg_param=0.1)
        )
    ]
)

print("\nRetraining QDA on the complete training set...")
final_qda.fit(X_train, y_train)

print("Retraining completed.")


# --------------------------------------------------
# 10. Evaluate once on the untouched test set
# --------------------------------------------------

test_probabilities = final_qda.predict_proba(X_test)[:, 1]

# Compare the default threshold with the selected threshold
test_predictions_default = (
    test_probabilities >= 0.50
).astype(int)

test_predictions_selected = (
    test_probabilities >= best_threshold
).astype(int)


def calculate_metrics(y_true, y_pred, probabilities):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities)
    }


test_results = pd.DataFrame([
    {
        "Model": "QDA threshold 0.50",
        **calculate_metrics(
            y_test,
            test_predictions_default,
            test_probabilities
        )
    },
    {
        "Model": f"QDA threshold {best_threshold:.2f}",
        **calculate_metrics(
            y_test,
            test_predictions_selected,
            test_probabilities
        )
    }
])

print("\nFINAL TEST-SET COMPARISON")
print("=" * 75)
print(test_results.round(4).to_string(index=False))

test_results.to_csv(
    OUTPUT_DIR / "qda_threshold_test_results.csv",
    index=False
)

print("\nSaved:")
print("outputs/qda_threshold_validation.csv")
print("outputs/qda_threshold_analysis.png")
print("outputs/qda_threshold_test_results.csv")