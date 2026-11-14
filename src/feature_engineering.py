
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler


RAW_PATH = Path("data/processed/creditcard_cleaned.csv")

OUTPUT_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def create_time_features(df):
    df = df.copy()

    # Cyclical time features
    seconds_in_day = df["Time"] % 86400

    df["hour_sin"] = np.sin(
        2 * np.pi * seconds_in_day / 86400
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * seconds_in_day / 86400
    )

    # Elapsed dataset day
    df["elapsed_day"] = np.floor(
        df["Time"] / 86400
    ).astype(int)

    return df


def create_velocity_features(df):
    df = df.copy()

    # Aggregate all transactions sharing a timestamp.
    activity = (
        df.groupby("Time")
        .size()
        .sort_index()
    )

    time_index = pd.to_timedelta(
        activity.index,
        unit="s"
    )

    activity.index = time_index

    # Strictly historical activity:
    # closed="left" excludes the current timestamp.
    windows = {
        60: "txn_count_prev_60s",
        300: "txn_count_prev_300s",
        3600: "txn_count_prev_3600s"
    }

    for seconds, feature_name in windows.items():

        historical_counts = (
            activity
            .rolling(
                f"{seconds}s",
                closed="left"
            )
            .sum()
        )

        historical_counts.index = (
            historical_counts.index.total_seconds()
        )

        df[feature_name] = (
            df["Time"]
            .map(historical_counts.to_dict())
            .fillna(0)
        )

    return df


def main():

    print("=" * 65)
    print("ADVANCED FINANCIAL FRAUD FEATURE ENGINEERING")
    print("=" * 65)

    df = pd.read_csv(RAW_PATH)

    # Sort chronologically
    df = df.sort_values("Time").reset_index(drop=True)

    print("\nOriginal shape:", df.shape)

    # 1. Log transformation
    df["log_amount"] = np.log1p(df["Amount"])

    # 2. Time features
    df = create_time_features(df)

    # 3. Historical transaction activity
    df = create_velocity_features(df)

    # 4. Chronological split
    train_boundary = df["Time"].quantile(0.70)
    validation_boundary = df["Time"].quantile(0.85)

    train = df[df["Time"] <= train_boundary].copy()

    validation = df[
        (df["Time"] > train_boundary)
        & (df["Time"] <= validation_boundary)
    ].copy()

    test = df[
        df["Time"] > validation_boundary
    ].copy()

    # 5. Fit robust scaler on training data only
    scaler = RobustScaler()

    scale_columns = ["Amount", "log_amount"]

    train_scaled = scaler.fit_transform(
        train[scale_columns]
    )

    validation_scaled = scaler.transform(
        validation[scale_columns]
    )

    test_scaled = scaler.transform(
        test[scale_columns]
    )

    for data, scaled_values in [
        (train, train_scaled),
        (validation, validation_scaled),
        (test, test_scaled)
    ]:
        data["amount_robust"] = scaled_values[:, 0]
        data["log_amount_robust"] = scaled_values[:, 1]

    # 6. Validate chronological separation
    assert train["Time"].max() < validation["Time"].min()
    assert validation["Time"].max() < test["Time"].min()

    assert set(train["Time"]).isdisjoint(set(validation["Time"]))
    assert set(validation["Time"]).isdisjoint(set(test["Time"]))

    # 7. Save partitions
    train.to_csv(
        OUTPUT_DIR / "train_features.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "validation_features.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test_features.csv",
        index=False
    )

    # 8. Save fitted scaler
    joblib.dump(
        scaler,
        MODEL_DIR / "amount_scaler.joblib"
    )

    # 9. Save feature schema
    excluded_columns = {"Class"}

    feature_columns = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    schema = {
        "feature_columns": feature_columns,
        "target": "Class",
        "scaled_columns": scale_columns,
        "train_boundary": float(train_boundary),
        "validation_boundary": float(validation_boundary),
        "velocity_feature_scope": (
            "Global transaction activity, not customer-level velocity"
        )
    }

    with open(
        REPORT_DIR / "feature_schema.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(schema, file, indent=4)

    # 10. Final report
    print("\nFEATURE ENGINEERING SUMMARY")
    print("-" * 65)

    print("Training:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)

    print("\nFraud cases:")
    print("Training:", int(train["Class"].sum()))
    print("Validation:", int(validation["Class"].sum()))
    print("Test:", int(test["Class"].sum()))

    print("\nNew engineered features:")

    print([
        "log_amount",
        "amount_robust",
        "log_amount_robust",
        "hour_sin",
        "hour_cos",
        "elapsed_day",
        "txn_count_prev_60s",
        "txn_count_prev_300s",
        "txn_count_prev_3600s"
    ])

    print("\nFeature count:", len(feature_columns))

    print("\nSaved files:")
    print("train_features.csv")
    print("validation_features.csv")
    print("test_features.csv")
    print("amount_scaler.joblib")
    print("feature_schema.json")

    print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
