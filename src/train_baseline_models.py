
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/train_features.csv")
VALIDATION_PATH = Path("data/processed/validation_features.csv")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def load_data():

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    # Remove raw Amount and log_amount because their
    # robust-scaled equivalents are already included.
    excluded = [
        "Class",
        "Amount",
        "log_amount"
    ]

    feature_columns = [
        column
        for column in train.columns
        if column not in excluded
    ]

    X_train = train[feature_columns]
    y_train = train["Class"]

    X_validation = validation[feature_columns]
    y_validation = validation["Class"]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns
    )


def evaluate_model(name, model, X_validation, y_validation):

    probabilities = model.predict_proba(X_validation)[:, 1]

    # Initial operating threshold.
    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "Model": name,
        "PR_AUC": average_precision_score(
            y_validation,
            probabilities
        ),
        "ROC_AUC": roc_auc_score(
            y_validation,
            probabilities
        ),
        "Precision": precision_score(
            y_validation,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_validation,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_validation,
            predictions,
            zero_division=0
        )
    }

    print(f"\n{name}")
    print("-" * 55)

    for metric, value in results.items():

        if metric != "Model":
            print(f"{metric}: {value:.6f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_validation,
            predictions
        )
    )

    return results


def main():

    print("=" * 65)
    print("ADVANCED FRAUD DETECTION - BASELINE MODEL TRAINING")
    print("=" * 65)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns
    ) = load_data()

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)

    fraud_count = int(y_train.sum())
    legitimate_count = int((y_train == 0).sum())

    scale_pos_weight = legitimate_count / fraud_count

    print("\nTraining class distribution:")
    print("Legitimate:", legitimate_count)
    print("Fraudulent:", fraud_count)
    print("Scale positive weight:", round(scale_pos_weight, 2))

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )
    }

    all_results = []

    for name, model in models.items():

        print("\n" + "=" * 65)
        print("Training:", name)
        print("=" * 65)

        model.fit(X_train, y_train)

        results = evaluate_model(
            name,
            model,
            X_validation,
            y_validation
        )

        all_results.append(results)

        model_filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            MODEL_DIR / model_filename
        )

        print("\nSaved model:", model_filename)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="PR_AUC",
        ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "baseline_model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("BASELINE MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved comparison:")
    print("reports/baseline_model_comparison.csv")

    print("\nBASELINE MODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
