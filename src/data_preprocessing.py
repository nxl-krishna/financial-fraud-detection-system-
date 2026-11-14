
from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/creditcard.csv")
OUTPUT_PATH = Path("data/processed/creditcard_cleaned.csv")


def main():
    print("=" * 60)
    print("FINANCIAL FRAUD DATA PREPROCESSING")
    print("=" * 60)

    # Load original dataset
    df = pd.read_csv(RAW_PATH)

    print("\nOriginal dataset:", df.shape)

    # Remove exact duplicate rows
    df_cleaned = df.drop_duplicates().copy()

    print("Cleaned dataset:", df_cleaned.shape)

    # Validate missing values
    print("\nMissing values:", df_cleaned.isnull().sum().sum())

    # Validate remaining duplicates
    print("Remaining duplicates:", df_cleaned.duplicated().sum())

    # Display class distribution
    print("\nClass distribution:")
    print(df_cleaned["Class"].value_counts())

    print("\nClass percentage:")
    print(df_cleaned["Class"].value_counts(normalize=True) * 100)

    # Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved cleaned dataset to:", OUTPUT_PATH)
    print("\nPREPROCESSING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
