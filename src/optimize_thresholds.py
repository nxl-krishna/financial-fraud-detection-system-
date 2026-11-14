
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


VALIDATION_PATH = Path("data/processed/validation_features.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)


def load_validation_data():

    validation = pd.read_csv(VALIDATION_PATH)

    excluded = [
        "Class",
        "Amount",
        "log_amount"
    ]

    feature_columns = [
        column
        for column in validation.columns
        if column not in excluded
    ]

    X = validation[feature_columns]
    y = validation["Class"]

    return X, y


def evaluate_thresholds(name, model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    pr_auc = average_precision_score(y, probabilities)

    thresholds = np.arange(0.001, 1.000, 0.001)

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Threshold": threshold,
            "PR_AUC": pr_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True_Positive": tp,
            "False_Positive": fp,
            "False_Negative": fn,
            "True_Negative": tn
        })

    return pd.DataFrame(results)


def report_best_thresholds(name, results):

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    # Best F1 threshold
    best_f1 = results.loc[
        results["F1"].idxmax()
    ]

    print("\nBEST F1 THRESHOLD")
    print("-" * 45)

    print(best_f1.to_string())

    # Maximum recall while maintaining minimum precision
    for minimum_precision in [0.80, 0.90, 0.95]:

        eligible = results[
            results["Precision"] >= minimum_precision
        ]

        print(
            f"\nBEST RECALL WITH PRECISION >= "
            f"{minimum_precision:.0%}"
        )
        print("-" * 45)

        if eligible.empty:
            print("No threshold satisfies this requirement.")
            continue

        best_recall = eligible.loc[
            eligible["Recall"].idxmax()
        ]

        print(best_recall.to_string())

    # Threshold closest to default 0.5
    default_result = results.iloc[
        (results["Threshold"] - 0.5).abs().argmin()
    ]

    print("\nDEFAULT THRESHOLD COMPARISON")
    print("-" * 45)

    print(default_result.to_string())


def main():

    print("=" * 65)
    print("FRAUD DETECTION THRESHOLD OPTIMIZATION")
    print("=" * 65)

    X, y = load_validation_data()

    print("\nValidation transactions:", len(y))
    print("Validation fraud cases:", int(y.sum()))

    model_names = [
        "logistic_regression",
        "random_forest",
        "xgboost"
    ]

    all_results = []

    selected_thresholds = {}

    for model_name in model_names:

        model_path = MODEL_DIR / f"{model_name}.joblib"

        model = joblib.load(model_path)

        results = evaluate_thresholds(
            model_name,
            model,
            X,
            y
        )

        all_results.append(results)

        report_best_thresholds(
            model_name,
            results
        )

        best_f1 = results.loc[
            results["F1"].idxmax()
        ]

        selected_thresholds[model_name] = {
            "best_f1_threshold": float(
                best_f1["Threshold"]
            ),
            "best_f1": float(best_f1["F1"]),
            "precision": float(best_f1["Precision"]),
            "recall": float(best_f1["Recall"])
        }

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(
        REPORT_DIR / "threshold_analysis.csv",
        index=False
    )

    with open(
        REPORT_DIR / "candidate_thresholds.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            selected_thresholds,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    print("=" * 65)

    print("\nSaved:")
    print("reports/threshold_analysis.csv")
    print("reports/candidate_thresholds.json")


if __name__ == "__main__":
    main()
