
import pandas as pd

DATA_PATH = "data/processed/creditcard_cleaned.csv"


def show_distribution(name, data):
    counts = data["Class"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    print(f"\n{name}")
    print("-" * 45)
    print(f"Transactions: {len(data):,}")
    print(f"Legitimate: {counts[0]:,}")
    print(f"Fraudulent: {counts[1]:,}")
    print(f"Fraud rate: {data['Class'].mean() * 100:.5f}%")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("Time").reset_index(drop=True)

    # Calculate chronological timestamp boundaries
    train_boundary = df["Time"].quantile(0.70)
    validation_boundary = df["Time"].quantile(0.85)

    # Keep every transaction with the same timestamp together
    train = df[df["Time"] <= train_boundary].copy()

    validation = df[
        (df["Time"] > train_boundary)
        & (df["Time"] <= validation_boundary)
    ].copy()

    test = df[df["Time"] > validation_boundary].copy()

    print("=" * 60)
    print("LEAKAGE-SAFE CHRONOLOGICAL SPLIT")
    print("=" * 60)

    show_distribution("TRAINING SET", train)
    show_distribution("VALIDATION SET", validation)
    show_distribution("TEST SET", test)

    print("\nTimestamp boundaries")
    print("-" * 45)

    print("Training maximum:", train["Time"].max())
    print("Validation minimum:", validation["Time"].min())
    print("Validation maximum:", validation["Time"].max())
    print("Test minimum:", test["Time"].min())

    # Validate no timestamp overlap
    train_times = set(train["Time"].unique())
    validation_times = set(validation["Time"].unique())
    test_times = set(test["Time"].unique())

    assert train_times.isdisjoint(validation_times)
    assert validation_times.isdisjoint(test_times)
    assert train_times.isdisjoint(test_times)

    # Validate chronological ordering
    assert train["Time"].max() < validation["Time"].min()
    assert validation["Time"].max() < test["Time"].min()

    print("\nSUCCESS: No timestamp overlap between partitions.")

    # Validate fraud representation
    for name, data in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        assert data["Class"].nunique() == 2
        print(f"{name}: {int(data['Class'].sum())} fraud cases")


if __name__ == "__main__":
    main()
