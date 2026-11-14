
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


VALIDATION_PATH = Path("data/processed/validation_features.csv")
MODEL_PATH = Path("models/random_forest.joblib")

OUTPUT_DIR = Path("reports/explainability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_positive_class_shap_values(explainer, X):

    raw_values = explainer.shap_values(X)

    # Compatibility with different SHAP versions.
    if isinstance(raw_values, list):
        values = raw_values[1]

    elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
        values = raw_values[:, :, 1]

    else:
        values = raw_values

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[-1]

    return values, float(expected_value)


def main():

    print("=" * 65)
    print("FRAUD DETECTION MODEL EXPLAINABILITY - SHAP")
    print("=" * 65)

    # Load validation data and selected model.
    validation = pd.read_csv(VALIDATION_PATH)

    model = joblib.load(MODEL_PATH)

    excluded = ["Class", "Amount", "log_amount"]

    feature_columns = [
        column
        for column in validation.columns
        if column not in excluded
    ]

    X = validation[feature_columns]
    y = validation["Class"]

    # Select a manageable and representative explanation sample.
    legitimate_sample = X[y == 0].sample(
        n=500,
        random_state=42
    )

    fraud_sample = X[y == 1]

    X_sample = pd.concat(
        [legitimate_sample, fraud_sample]
    )

    print("\nExplanation sample:", X_sample.shape)
    print("Fraud examples:", int(y.loc[X_sample.index].sum()))

    # Select the fraud transaction with the highest predicted probability.
    fraud_indices = y[y == 1].index

    fraud_probabilities = model.predict_proba(
        X.loc[fraud_indices]
    )[:, 1]

    selected_index = fraud_indices[
        int(np.argmax(fraud_probabilities))
    ]

    local_transaction = X.loc[[selected_index]]

    print("\nSelected validation transaction index:", selected_index)
    print(
        "Predicted fraud probability:",
        round(
            float(model.predict_proba(local_transaction)[0, 1]),
            6
        )
    )

    # Explain the model.
    print("\nInitializing SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    shap_values, expected_value = get_positive_class_shap_values(
        explainer,
        X_sample
    )

    print("SHAP calculations completed.")

    # 1. Global summary plot.
    print("\nGenerating SHAP summary plot...")

    plt.figure(figsize=(12, 9))

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20
    )

    plt.title(
        "SHAP Summary - Fraud Detection",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # 2. Global importance bar chart.
    print("Generating SHAP feature importance plot...")

    mean_importance = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Mean_Absolute_SHAP": mean_importance
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    top_features = importance.head(20).sort_values(
        "Mean_Absolute_SHAP"
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Mean_Absolute_SHAP"]
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Top 20 Features Influencing Fraud Predictions")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_feature_importance.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # 3. Individual fraud transaction explanation.
    print("Generating individual SHAP explanation...")

    local_position = X_sample.index.get_loc(selected_index)

    local_explanation = shap.Explanation(
        values=shap_values[local_position],
        base_values=expected_value,
        data=X_sample.iloc[local_position].values,
        feature_names=feature_columns
    )

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        local_explanation,
        max_display=15,
        show=False
    )

    plt.title(
        "Individual Fraud Transaction Explanation",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "individual_fraud_explanation.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTOP 15 FEATURES")
    print("-" * 55)

    print(
        importance.head(15).to_string(index=False)
    )

    print("\nSaved files:")
    print("reports/explainability/shap_summary.png")
    print("reports/explainability/shap_feature_importance.png")
    print("reports/explainability/shap_feature_importance.csv")
    print("reports/explainability/individual_fraud_explanation.png")

    print("\nSHAP EXPLAINABILITY COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
