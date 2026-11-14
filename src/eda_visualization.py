
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/processed/creditcard_cleaned.csv"
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    print("Creating fraud detection visualizations...")

    # 1. Class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Class")
    plt.title("Legitimate vs Fraudulent Transactions")
    plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    save_plot("class_distribution.png")

    # 2. Fraud percentage
    fraud_percentage = df["Class"].value_counts(normalize=True) * 100

    plt.figure(figsize=(7, 7))
    plt.pie(
        fraud_percentage,
        labels=["Legitimate", "Fraud"],
        autopct="%1.3f%%",
        startangle=90
    )
    plt.title("Transaction Class Percentage")
    save_plot("fraud_percentage.png")

    # 3. Transaction amount distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=50,
        log_scale=(False, True),
        element="step"
    )
    plt.title("Transaction Amount Distribution")
    save_plot("transaction_amount_distribution.png")

    # 4. Fraud amount boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Class", y="Amount")
    plt.yscale("symlog")
    plt.title("Transaction Amount by Class")
    save_plot("fraud_amount_boxplot.png")

    # 5. Correlation heatmap
    plt.figure(figsize=(18, 14))
    correlation = df.corr()
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1
    )
    plt.title("Feature Correlation Heatmap")
    save_plot("correlation_heatmap.png")

    # 6. Time distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=60,
        element="step",
        stat="density",
        common_norm=False
    )
    plt.title("Transaction Timing: Fraud vs Legitimate")
    plt.xlabel("Seconds elapsed from first transaction")
    save_plot("transaction_time_distribution.png")

    print("\nAll six visualizations created successfully.")
    print("Saved inside:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
