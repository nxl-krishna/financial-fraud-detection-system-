
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_PATH = (
    BASE_DIR
    / "reports"
    / "final_test_evaluation.json"
)


def render_model_monitoring():

    st.subheader("Model Performance & Monitoring")

    st.caption(
        "Held-out test evaluation of the trained "
        "Class-weighted Random Forest fraud detection model."
    )

    # --------------------------------------------------------
    # LOAD EVALUATION REPORT
    # --------------------------------------------------------

    try:

        with open(
            EVALUATION_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            report = json.load(file)

    except FileNotFoundError:

        st.error(
            "Final evaluation report was not found."
        )
        return

    except (json.JSONDecodeError, OSError) as error:

        st.error(
            f"Could not load evaluation report: {error}"
        )
        return

    # --------------------------------------------------------
    # EXTRACT EVALUATION METRICS
    # --------------------------------------------------------

    model_name = report["model"]

    threshold = report["threshold"]

    total_transactions = report["test_transactions"]

    fraud_cases = report["test_fraud_cases"]

    precision = report["precision"]
    recall = report["recall"]
    f1 = report["f1"]

    pr_auc = report["pr_auc"]
    roc_auc = report["roc_auc"]

    specificity = report["specificity"]
    false_positive_rate = report["false_positive_rate"]

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown("### Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    info_col1.metric(
        "Selected Model",
        model_name,
    )

    info_col2.metric(
        "Decision Threshold",
        f"{threshold:.4f}",
    )

    info_col3.metric(
        "Test Transactions",
        f"{total_transactions:,}",
    )

    st.divider()

    # --------------------------------------------------------
    # PRIMARY PERFORMANCE METRICS
    # --------------------------------------------------------

    st.markdown("### Classification Performance")

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    metric_col1.metric(
        "Precision",
        f"{precision:.2%}",
    )

    metric_col2.metric(
        "Recall",
        f"{recall:.2%}",
    )

    metric_col3.metric(
        "F1 Score",
        f"{f1:.2%}",
    )

    metric_col4.metric(
        "PR-AUC",
        f"{pr_auc:.2%}",
    )

    metric_col5, metric_col6, metric_col7 = st.columns(3)

    metric_col5.metric(
        "ROC-AUC",
        f"{roc_auc:.2%}",
    )

    metric_col6.metric(
        "Specificity",
        f"{specificity:.2%}",
    )

    metric_col7.metric(
        "False Positive Rate",
        f"{false_positive_rate:.3%}",
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("### Confusion Matrix")

    st.caption(
        "Actual transaction classification compared with "
        "the model's predicted classification."
    )

    tn = report["true_negatives"]
    fp = report["false_positives"]
    fn = report["false_negatives"]
    tp = report["true_positives"]

    fig = go.Figure(
        data=go.Heatmap(
            z=[
                [tn, fp],
                [fn, tp],
            ],
            x=[
                "Predicted Legitimate",
                "Predicted Fraud",
            ],
            y=[
                "Actual Legitimate",
                "Actual Fraud",
            ],
            colorscale="Blues",
            text=[
                [f"{tn:,}", f"{fp:,}"],
                [f"{fn:,}", f"{tp:,}"],
            ],
            texttemplate="%{text}",
            textfont=dict(size=20),
            hovertemplate=(
                "Actual: %{y}<br>"
                "Prediction: %{x}<br>"
                "Transactions: %{z:,}"
                "<extra></extra>"
            ),
            showscale=True,
        )
    )

    fig.update_layout(
        title="Fraud Detection Confusion Matrix",
        height=450,
        xaxis_title="Predicted Classification",
        yaxis_title="Actual Classification",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # PREDICTION OUTCOMES
    # --------------------------------------------------------

    st.markdown("### Prediction Outcomes")

    outcome_col1, outcome_col2, outcome_col3, outcome_col4 = (
        st.columns(4)
    )

    outcome_col1.metric(
        "True Positives",
        f"{tp:,}",
        help="Fraudulent transactions correctly detected.",
    )

    outcome_col2.metric(
        "False Positives",
        f"{fp:,}",
        help="Legitimate transactions incorrectly flagged.",
    )

    outcome_col3.metric(
        "False Negatives",
        f"{fn:,}",
        help="Fraudulent transactions missed by the model.",
    )

    outcome_col4.metric(
        "True Negatives",
        f"{tn:,}",
        help="Legitimate transactions correctly identified.",
    )

    st.divider()

    # --------------------------------------------------------
    # TEST DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("### Evaluation Dataset")

    fraud_rate = (
        fraud_cases / total_transactions * 100
        if total_transactions > 0
        else 0
    )

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

    dataset_col1.metric(
        "Total Test Cases",
        f"{total_transactions:,}",
    )

    dataset_col2.metric(
        "Actual Fraud Cases",
        f"{fraud_cases:,}",
    )

    dataset_col3.metric(
        "Fraud Percentage",
        f"{fraud_rate:.3f}%",
    )

    st.info(
        "Evaluation limitation: the held-out test dataset contains "
        f"only {fraud_cases} fraudulent transactions. "
        "Performance estimates may vary with different data. "
        "These are offline test results, not live production metrics."
    )
