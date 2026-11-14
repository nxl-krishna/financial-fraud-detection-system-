
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


TEST_PATH = Path("data/processed/test_features.csv")
MODEL_PATH = Path("models/random_forest.joblib")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Selected using validation data only.
FINAL_THRESHOLD = 0.2441122105893745


def main():

    print("=" * 65)
    print("FINAL FRAUD DETECTION MODEL EVALUATION")
    print("=" * 65)

    print("\nLoading final candidate model...")

    model = joblib.load(MODEL_PATH)

    test = pd.read_csv(TEST_PATH)

    excluded = [
        "Class",
        "Amount",
        "log_amount"
    ]

    feature_columns = [
        column
        for column in test.columns
        if column not in excluded
    ]

    X_test = test[feature_columns]
    y_test = test["Class"]

    print("\nTest transactions:", len(test))
    print("Test fraud cases:", int(y_test.sum()))
    print("Selected threshold:", FINAL_THRESHOLD)

    print("\nGenerating predictions...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    # Calculate final metrics.
    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(f"\nPR-AUC: {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print(f"Specificity: {specificity:.6f}")
    print(f"False Positive Rate: {false_positive_rate:.6f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nDetailed Counts:")
    print("True Positives:", int(tp))
    print("False Positives:", int(fp))
    print("False Negatives:", int(fn))
    print("True Negatives:", int(tn))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    # Save final evaluation results.
    results = {
        "model": "Class-weighted Random Forest",
        "threshold": FINAL_THRESHOLD,
        "test_transactions": int(len(test)),
        "test_fraud_cases": int(y_test.sum()),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }

    with open(
        REPORT_DIR / "final_test_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # Save individual test predictions for later analysis.
    predictions_df = pd.DataFrame({
        "actual_class": y_test.values,
        "fraud_probability": probabilities,
        "predicted_class": predictions
    })

    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False
    )

    print("\nSaved:")
    print("reports/final_test_evaluation.json")
    print("reports/test_predictions.csv")

    print("\nFINAL TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
