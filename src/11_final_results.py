from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


# --------------------------------------------------
# 2. Load the existing experimental results
# --------------------------------------------------

baseline_results = pd.read_csv(
    OUTPUT_DIR / "model_comparison.csv"
)

threshold_results = pd.read_csv(
    OUTPUT_DIR / "qda_threshold_test_results.csv"
)

# Rename the baseline QDA entry for clarity
baseline_results.loc[
    baseline_results["Model"] == "QDA",
    "Model"
] = "QDA (threshold 0.50)"

# Remove the default-threshold QDA row because it
# already exists in the baseline comparison table.
threshold_results = threshold_results[
    threshold_results["Model"] != "QDA threshold 0.50"
]

# Combine the three baseline models with the
# validation-selected QDA threshold.
final_results = pd.concat(
    [baseline_results, threshold_results],
    ignore_index=True
)

# Keep the final table organised
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

final_results = final_results[
    ["Model"] + metric_columns
]

# Save the table
final_results.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 3. Print the comparison table
# --------------------------------------------------

print("\nFINAL MODEL COMPARISON")
print("=" * 85)

print(
    final_results.to_string(
        index=False,
        formatters={
            column: "{:.4f}".format
            for column in metric_columns
        }
    )
)


# --------------------------------------------------
# 4. Create a comparison chart
# --------------------------------------------------

plot_data = final_results.melt(
    id_vars="Model",
    value_vars=metric_columns,
    var_name="Metric",
    value_name="Score"
)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(13, 7))

sns.barplot(
    data=plot_data,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Final Comparison of Credit Default Models")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Model", loc="best")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved final comparison table:")
print("outputs/final_model_comparison.csv")

print("\nSaved final comparison chart:")
print("outputs/final_model_comparison.png")
