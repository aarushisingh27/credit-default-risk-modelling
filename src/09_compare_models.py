from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the saved model metrics
# --------------------------------------------------

logistic = pd.read_csv(
    OUTPUT_DIR / "logistic_regression_metrics.csv"
)
lda = pd.read_csv(
    OUTPUT_DIR / "lda_metrics.csv"
)
qda = pd.read_csv(
    OUTPUT_DIR / "qda_metrics.csv"
)

# Add model names where necessary
logistic["Model"] = "Logistic Regression"

# Keep the same metric columns for all models
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

results = pd.concat(
    [
        logistic[["Model"] + metric_columns],
        lda[["Model"] + metric_columns],
        qda[["Model"] + metric_columns]
    ],
    ignore_index=True
)


# --------------------------------------------------
# 3. Display and save the comparison table
# --------------------------------------------------

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 75)

print(
    results.to_string(
        index=False,
        formatters={
            column: "{:.4f}".format
            for column in metric_columns
        }
    )
)

results.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 4. Plot the evaluation metrics
# --------------------------------------------------

plot_data = results.melt(
    id_vars="Model",
    value_vars=metric_columns,
    var_name="Metric",
    value_name="Score"
)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))

sns.barplot(
    data=plot_data,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Comparison of Credit Default Classification Models")
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.ylim(0, 1)
plt.legend(title="Model")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved comparison table:")
print("outputs/model_comparison.csv")

print("\nSaved comparison chart:")
print("outputs/model_comparison.png")