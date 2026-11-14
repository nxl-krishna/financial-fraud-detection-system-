
import pandas as pd

DATA_PATH = "data/raw/creditcard.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # All rows that belong to duplicate groups
    duplicate_groups = df[df.duplicated(keep=False)]

    # Only additional copies that would be removed
    duplicate_mask = df.duplicated(keep="first")
    duplicate_copies = df[duplicate_mask]

    print("=" * 60)
    print("DUPLICATE INVESTIGATION")
    print("=" * 60)

    print("\n1. Duplicate group rows:")
    print(len(duplicate_groups))

    print("\n2. Additional duplicate copies:")
    print(len(duplicate_copies))

    print("\n3. Additional duplicate copies by class:")
    print(duplicate_copies["Class"].value_counts())

    print("\n4. Duplicate group rows by class:")
    print(duplicate_groups["Class"].value_counts())

    print("\n5. Original class distribution:")
    print(df["Class"].value_counts())

    cleaned = df.drop_duplicates()

    print("\n6. Class distribution after removing exact duplicates:")
    print(cleaned["Class"].value_counts())

    print("\n7. Dataset shape after duplicate removal:")
    print(cleaned.shape)


if __name__ == "__main__":
    main()
