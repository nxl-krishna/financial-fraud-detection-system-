from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")

FILES = {
    "Training": DATA_DIR / "train_features.csv",
    "Validation": DATA_DIR / "validation_features.csv",
    "Test": DATA_DIR / "test_features.csv"
}


def main():

    print("=" * 65)
    print("ENGINEERED FEATURE VALIDATION")
    print("=" * 65)

    datasets = {}

    # 1. Load all datasets
    for name, path in FILES.items():

        df = pd.read_csv(path)

        datasets[name] = df

        print(f"\n{name} dataset")
        print("-" * 45)

        print("Shape:", df.shape)
        print("Missing values:", df.isnull().sum().sum())
        print("Duplicate rows:", df.duplicated().sum())
        print("Fraud cases:", int(df["Class"].sum()))
        print("Fraud rate:", round(df["Class"].mean() * 100, 5), "%")

        assert df.isnull().sum().sum() == 0
        assert df["Class"].nunique() == 2

    # 2. Verify feature consistency
    train_columns = set(datasets["Training"].columns)

    for name, df in datasets.items():

        assert set(df.columns) == train_columns

    print("\nSUCCESS: All datasets have identical columns.")

    # 3. Check chronological boundaries
    train = datasets["Training"]
    validation = datasets["Validation"]
    test = datasets["Test"]

    assert train["Time"].max() < validation["Time"].min()
    assert validation["Time"].max() < test["Time"].min()

    print("SUCCESS: Chronological ordering is valid.")

    # 4. Check historical activity features
    velocity_columns = [
        "txn_count_prev_60s",
        "txn_count_prev_300s",
        "txn_count_prev_3600s"
    ]

    for column in velocity_columns:

        assert (train[column] >= 0).all()
        assert (validation[column] >= 0).all()
        assert (test[column] >= 0).all()

    print("SUCCESS: Historical activity features are non-negative.")

    # 5. Check amount transformations
    for name, df in datasets.items():

        assert (df["log_amount"] >= 0).all()

    print("SUCCESS: Log amount transformations are valid.")

    # 6. Confirm test fraud count
    assert int(test["Class"].sum()) == 52

    print("SUCCESS: Test set remains consistent.")

    print("\n" + "=" * 65)
    print("ALL FEATURE VALIDATION CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()