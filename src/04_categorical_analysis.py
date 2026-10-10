from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "credit_default_raw.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 1. Categorical variables: gender, education,
#    and marital status
# --------------------------------------------------

categorical_columns = ["X2", "X3", "X4"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col in zip(axes, categorical_columns):
    sns.countplot(data=df, x=col, hue="Y", ax=ax)
    ax.set_title(f"{col}: Distribution by Default Status")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of Clients")
    ax.legend(title="Default (Y)")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "categorical_distributions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 2. Repayment status across six months
# --------------------------------------------------

repayment_columns = ["X6", "X7", "X8", "X9", "X10", "X11"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, col in zip(axes, repayment_columns):
    sns.countplot(data=df, x=col, hue="Y", ax=ax)
    ax.set_title(f"{col}: Repayment Status")
    ax.set_xlabel("Recorded Status")
    ax.set_ylabel("Number of Clients")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Default (Y)")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "repayment_status_distributions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 3. Default rates by category
# --------------------------------------------------

default_rate_tables = []

for col in categorical_columns + repayment_columns:
    table = (
        df.groupby(col, observed=True)["Y"]
        .agg(["count", "mean"])
        .reset_index()
    )

    table.rename(
        columns={
            col: "category",
            "count": "number_of_clients",
            "mean": "default_rate"
        },
        inplace=True
    )

    table.insert(0, "variable", col)
    default_rate_tables.append(table)

default_rates = pd.concat(default_rate_tables, ignore_index=True)

default_rates.to_csv(
    OUTPUT_DIR / "categorical_default_rates.csv",
    index=False
)

# Display the results in the terminal
for col in categorical_columns + repayment_columns:
    print(f"\nDefault rates for {col}:")
    print(
        default_rates[default_rates["variable"] == col]
        .to_string(index=False)
    )

print("\nCategorical analysis completed.")
print("Saved: outputs/categorical_distributions.png")
print("Saved: outputs/repayment_status_distributions.png")
print("Saved: outputs/categorical_default_rates.csv")