
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)


TRAIN_PATH = Path("data/processed/train_features.csv")
VALIDATION_PATH = Path("data/processed/validation_features.csv")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def load_data():

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    excluded = ["Class", "Amount", "log_amount"]

    feature_columns = [
        column for column in train.columns
        if column not in excluded
    ]

    X_train = train[feature_columns]
    y_train = train["Class"]

    X_validation = validation[feature_columns]
    y_validation = validation["Class"]

    return X_train, y_train, X_validation, y_validation


def get_operating_points(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # The final precision/recall entry has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    best_f1_index = int(np.argmax(f1))

    eligible = np.where(precision >= 0.90)[0]

    if len(eligible) > 0:
        best_precision_index = eligible[
            np.argmax(recall[eligible])
        ]
    else:
        best_precision_index = None

    return {
        "best_f1": {
            "threshold": float(thresholds[best_f1_index]),
            "precision": float(precision[best_f1_index]),
            "recall": float(recall[best_f1_index]),
            "f1": float(f1[best_f1_index])
        },
        "precision_90": (
            {
                "threshold": float(
                    thresholds[best_precision_index]
                ),
                "precision": float(
                    precision[best_precision_index]
                ),
                "recall": float(
                    recall[best_precision_index]
                ),
                "f1": float(
                    f1[best_precision_index]
                )
            }
            if best_precision_index is not None
            else None
        )
    }


def evaluate_strategy(
    name,
    model,
    X_validation,
    y_validation
):

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        probabilities
    )

    operating_points = get_operating_points(
        y_validation,
        probabilities
    )

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    print(f"PR-AUC: {pr_auc:.6f}")

    print("\nBest F1 operating point:")
    print(operating_points["best_f1"])

    print("\nBest recall with precision >= 90%:")
    print(operating_points["precision_90"])

    # Confusion matrix at best F1 threshold
    threshold = operating_points["best_f1"]["threshold"]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\nConfusion matrix at best F1 threshold:")
    print(
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1]
        )
    )

    return {
        "Strategy": name,
        "PR_AUC": pr_auc,
        "Best_F1": operating_points["best_f1"]["f1"],
        "Best_F1_Threshold": operating_points["best_f1"]["threshold"],
        "Best_F1_Precision": operating_points["best_f1"]["precision"],
        "Best_F1_Recall": operating_points["best_f1"]["recall"],
        "Precision_90_Recall": (
            operating_points["precision_90"]["recall"]
            if operating_points["precision_90"]
            else np.nan
        ),
        "Precision_90_Threshold": (
            operating_points["precision_90"]["threshold"]
            if operating_points["precision_90"]
            else np.nan
        )
    }


def main():

    print("=" * 65)
    print("ADVANCED CLASS IMBALANCE STRATEGY COMPARISON")
    print("=" * 65)

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = load_data()

    print("\nOriginal training distribution:")
    print(y_train.value_counts().to_string())

    # Strategy 1: Existing class-weighted Random Forest.
    print("\nLoading existing class-weighted Random Forest...")

    weighted_model = joblib.load(
        MODEL_DIR / "random_forest.joblib"
    )

    # Strategy 2: SMOTE applied only to training data.
    smote_model = ImbPipeline([
        (
            "sampler",
            SMOTE(
                sampling_strategy=0.5,
                random_state=42,
                k_neighbors=5
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                class_weight=None,
                n_jobs=-1,
                random_state=42
            )
        )
    ])

    # Strategy 3: Random undersampling applied only to training data.
    undersampling_model = ImbPipeline([
        (
            "sampler",
            RandomUnderSampler(
                sampling_strategy=0.5,
                random_state=42
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                class_weight=None,
                n_jobs=-1,
                random_state=42
            )
        )
    ])

    strategies = [
        ("Class Weighted Random Forest", weighted_model),
        ("SMOTE Random Forest", smote_model),
        ("Undersampled Random Forest", undersampling_model)
    ]

    all_results = []

    for name, model in strategies:

        if name != "Class Weighted Random Forest":

            print("\n" + "=" * 65)
            print("TRAINING:", name)
            print("=" * 65)

            model.fit(X_train, y_train)

            print("Training completed.")

            filename = (
                name.lower()
                .replace(" ", "_")
                + ".joblib"
            )

            joblib.dump(
                model,
                MODEL_DIR / filename
            )

            print("Saved:", filename)

        result = evaluate_strategy(
            name,
            model,
            X_validation,
            y_validation
        )

        all_results.append(result)

    comparison = pd.DataFrame(all_results)

    comparison = comparison.sort_values(
        "PR_AUC",
        ascending=False
    )

    comparison.to_csv(
        REPORT_DIR / "imbalance_strategy_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL IMBALANCE STRATEGY COMPARISON")
    print("=" * 65)

    print(comparison.to_string(index=False))

    print("\nSaved:")
    print("reports/imbalance_strategy_comparison.csv")

    print("\nIMBALANCE STRATEGY COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
