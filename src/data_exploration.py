
import pandas as pd

DATA_PATH = "data/raw/creditcard.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("FINANCIAL FRAUD DATA EXPLORATION")
    print("=" * 60)

    print("\n1. Dataset information")
    print(df.info())

    print("\n2. Statistical summary")
    print(df.describe().T)

    print("\n3. Fraud distribution")
    print(df["Class"].value_counts())

    print("\n4. Fraud percentage")
    print(df["Class"].value_counts(normalize=True) * 100)

    print("\n5. Duplicate analysis")
    print("Total duplicates:", df.duplicated().sum())

    print("\nDuplicates by class:")
    print(
        df.loc[df.duplicated(keep=False), "Class"]
        .value_counts()
    )

    print("\n6. Transaction amount by class")
    print(
        df.groupby("Class")["Amount"]
        .agg(["count", "mean", "median", "min", "max"])
    )

    print("\n7. Fraud amount statistics")
    print(df[df["Class"] == 1]["Amount"].describe())

    print("\n8. Legitimate amount statistics")
    print(df[df["Class"] == 0]["Amount"].describe())


if __name__ == "__main__":
    main()
