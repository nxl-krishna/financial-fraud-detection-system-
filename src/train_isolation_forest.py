
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
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
        column
        for column in train.columns
        if column not in excluded
    ]

    X_train = train[feature_columns]
    y_train = train["Class"]

    X_validation = validation[feature_columns]
    y_validation = validation["Class"]

    return X_train, y_train, X_validation, y_validation


def main():

    print("=" * 65)
    print("UNSUPERVISED FRAUD DETECTION - ISOLATION FOREST")
    print("=" * 65)

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = load_data()

    # Train exclusively on legitimate transactions.
    X_legitimate = X_train[y_train == 0]

    print("\nTraining transactions:", len(X_legitimate))
    print("Validation transactions:", len(X_validation))
    print("Validation fraud cases:", int(y_validation.sum()))

    model = IsolationForest(
        n_estimators=300,
        max_samples=1024,
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining Isolation Forest...")

    model.fit(X_legitimate)

    print("Training completed.")

    # Lower decision_function values are more anomalous.
    # Negating the scores makes higher values indicate greater anomaly.
    anomaly_scores = -model.decision_function(
        X_validation
    )

    pr_auc = average_precision_score(
        y_validation,
        anomaly_scores
    )

    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        anomaly_scores
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    best_f1_index = int(np.argmax(f1))

    best_threshold = float(
        thresholds[best_f1_index]
    )

    predictions = (
        anomaly_scores >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    ).ravel()

    print("\n" + "=" * 65)
    print("ISOLATION FOREST RESULTS")
    print("=" * 65)

    print(f"\nPR-AUC: {pr_auc:.6f}")

    print("\nBest F1 threshold:", best_threshold)
    print(f"Precision: {precision[best_f1_index]:.6f}")
    print(f"Recall: {recall[best_f1_index]:.6f}")
    print(f"F1: {f1[best_f1_index]:.6f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_validation,
            predictions
        )
    )

    print("\nTrue Positives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Negatives:", tn)

    joblib.dump(
        model,
        MODEL_DIR / "isolation_forest.joblib"
    )

    results = {
        "model": "Isolation Forest",
        "pr_auc": float(pr_auc),
        "best_threshold": best_threshold,
        "precision": float(precision[best_f1_index]),
        "recall": float(recall[best_f1_index]),
        "f1": float(f1[best_f1_index]),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }

    pd.DataFrame([results]).to_csv(
        REPORT_DIR / "isolation_forest_results.csv",
        index=False
    )

    print("\nSaved:")
    print("models/isolation_forest.joblib")
    print("reports/isolation_forest_results.csv")

    print("\nISOLATION FOREST TRAINING COMPLETED")


if __name__ == "__main__":
    main()
