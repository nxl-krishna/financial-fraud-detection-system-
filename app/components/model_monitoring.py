
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.drift_monitoring import (
    load_model_features,
    load_reference_data,
    calculate_feature_drift,
    calculate_overall_drift,
    calculate_live_drift,
)


BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_PATH = (
    BASE_DIR
    / "reports"
    / "final_test_evaluation.json"
)

TEST_FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_features.csv"
)


# ============================================================
# CACHED OFFLINE DRIFT ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def load_offline_drift_results():
    """
    Compare training reference features with held-out test features.

    This is offline historical analysis, not live production monitoring.
    """

    features = load_model_features()

    reference = load_reference_data()

    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Test feature dataset not found: {TEST_FEATURES_PATH}"
        )

    current = pd.read_csv(
        TEST_FEATURES_PATH,
        usecols=features,
    )

    results = calculate_feature_drift(
        reference,
        current,
    )

    summary = calculate_overall_drift(results)

    return (
        results,
        summary,
        reference.shape[0],
        current.shape[0],
    )


# ============================================================
# MAIN MODEL MONITORING PAGE
# ============================================================

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
            encoding="utf-8",
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

    st.divider()

    # ========================================================
    # DATA DRIFT MONITORING
    # ========================================================

    st.markdown("## Data Drift Monitoring")

    st.caption(
        "Population Stability Index (PSI) analysis across all "
        "37 model input features."
    )

    st.info(
        "Offline historical comparison: the training dataset is "
        "the reference population and the held-out test dataset "
        "is the comparison population. This is not live production "
        "drift monitoring and does not establish model performance "
        "degradation."
    )

       # --------------------------------------------------------
    # LOAD AND DISPLAY OFFLINE DRIFT RESULTS
    # --------------------------------------------------------

    offline_drift_available = False

    try:

        with st.spinner(
            "Analyzing feature distributions..."
        ):

            (
                drift_results,
                drift_summary,
                reference_count,
                current_count,
            ) = load_offline_drift_results()

        offline_drift_available = True

    except FileNotFoundError:

        st.warning(
            "Offline historical drift analysis is unavailable "
            "because the processed training or test datasets "
            "are not included in this deployment."
        )

        st.info(
            "The API, model performance report, and live "
            "transaction monitoring can continue operating. "
            "Offline drift analysis requires the original "
            "processed datasets."
        )

    except Exception as error:

        st.error(
            f"Could not calculate offline data drift: {error}"
        )

    # --------------------------------------------------------
    # DISPLAY OFFLINE DRIFT RESULTS
    # --------------------------------------------------------

    if offline_drift_available:

        # ----------------------------------------------------
        # OVERALL DRIFT STATUS
        # ----------------------------------------------------

        severity = drift_summary["severity"]

        st.markdown("### Overall Drift Status")

        if severity == "SIGNIFICANT":

            st.error(
                "Significant distribution drift detected "
                "in the historical comparison."
            )

        elif severity == "MODERATE":

            st.warning(
                "Moderate distribution drift detected."
            )

        elif severity == "STABLE":

            st.success(
                "Feature distributions are broadly stable."
            )

        else:

            st.warning(
                "Insufficient valid data to determine overall drift."
            )

        # ----------------------------------------------------
        # DATASET COUNTS
        # ----------------------------------------------------

        count_col1, count_col2 = st.columns(2)

        count_col1.metric(
            "Reference Training Records",
            f"{reference_count:,}",
        )

        count_col2.metric(
            "Comparison Test Records",
            f"{current_count:,}",
        )

        # ----------------------------------------------------
        # DRIFT SUMMARY METRICS
        # ----------------------------------------------------

        st.markdown("### Feature Drift Summary")

        drift_col1, drift_col2, drift_col3, drift_col4 = (
            st.columns(4)
        )

        drift_col1.metric(
            "Features Monitored",
            drift_summary["features_monitored"],
        )

        drift_col2.metric(
            "Stable Features",
            drift_summary["stable_features"],
        )

        drift_col3.metric(
            "Moderate Drift",
            drift_summary["moderate_features"],
        )

        drift_col4.metric(
            "Significant Drift",
            drift_summary["significant_features"],
        )

        # ----------------------------------------------------
        # TOP DRIFTED FEATURES
        # ----------------------------------------------------

        st.markdown("### Most Drifted Features")

        valid_results = drift_results.dropna(
            subset=["psi"]
        )

        if not valid_results.empty:

            top_features = (
                valid_results
                .sort_values(
                    "psi",
                    ascending=False,
                )
                .head(15)
                .sort_values(
                    "psi",
                    ascending=True,
                )
            )

            severity_colors = {
                "STABLE": "#2E8B57",
                "MODERATE": "#E6A23C",
                "SIGNIFICANT": "#D9534F",
            }

            bar_colors = [
                severity_colors.get(
                    value,
                    "#808080",
                )
                for value in top_features["severity"]
            ]

            drift_fig = go.Figure(
                go.Bar(
                    x=top_features["psi"],
                    y=top_features["feature"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=top_features["psi"].map(
                        lambda value: f"{value:.3f}"
                    ),
                    textposition="outside",
                    hovertemplate=(
                        "Feature: %{y}<br>"
                        "PSI: %{x:.4f}"
                        "<extra></extra>"
                    ),
                )
            )

            drift_fig.update_layout(
                title=(
                    "Top 15 Features by "
                    "Population Stability Index"
                ),
                xaxis_title="Population Stability Index (PSI)",
                yaxis_title="Model Feature",
                height=650,
                showlegend=False,
            )

            st.plotly_chart(
                drift_fig,
                use_container_width=True,
            )

        else:

            st.warning(
                "No valid PSI measurements are available."
            )

        # ----------------------------------------------------
        # FEATURE-LEVEL DRIFT TABLE
        # ----------------------------------------------------

        st.markdown("### Detailed Feature Drift Report")

        display_columns = [
            "feature",
            "psi",
            "severity",
            "reference_count",
            "current_count",
            "missing_reference",
            "missing_current",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in drift_results.columns
        ]

        display_results = drift_results[
            available_columns
        ].copy()

        if "psi" in display_results.columns:

            display_results["psi"] = (
                display_results["psi"].round(4)
            )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # PSI INTERPRETATION
        # ----------------------------------------------------

        with st.expander(
            "Understand PSI severity thresholds"
        ):

            st.markdown(
                """
                | PSI value | Interpretation |
                |---|---|
                | Below 0.10 | Stable |
                | 0.10 to below 0.25 | Moderate drift |
                | 0.25 or above | Significant drift |

                These are commonly used monitoring heuristics,
                not universal statistical significance thresholds.

                A high PSI indicates that a feature's distribution
                has changed. It does not automatically mean that
                the model is inaccurate or needs retraining.
                """
            )

    # --------------------------------------------------------
    # LIVE MONITORING STATUS
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Live Production Drift Monitoring")

    st.caption(
        "Live monitoring uses genuine API transactions stored "
        "in the application database. Synthetic TEST records "
        "are excluded."
    )

    try:

        with st.spinner("Checking live transaction data..."):

            live_result = calculate_live_drift(
                min_sample_size=100
            )

        live_status = live_result["status"]

        live_count = live_result["live_records"]

        minimum_required = live_result["minimum_required"]

        live_col1, live_col2 = st.columns(2)

        live_col1.metric(
            "Valid Live API Records",
            f"{live_count:,}",
        )

        live_col2.metric(
            "Minimum Required Records",
            f"{minimum_required:,}",
        )

        if live_status == "INSUFFICIENT_DATA":

            st.warning(
                "Live drift monitoring is not yet available."
            )

            st.info(live_result["message"])

        elif live_status == "INVALID_SCHEMA":

            st.error(
                "Live feature schema validation failed."
            )

            st.error(live_result["message"])

        elif live_status == "AVAILABLE":

            live_summary = live_result["summary"]

            live_severity = live_summary["severity"]

            if live_severity == "SIGNIFICANT":

                st.error(
                    "Significant drift detected in live API traffic."
                )

            elif live_severity == "MODERATE":

                st.warning(
                    "Moderate drift detected in live API traffic."
                )

            elif live_severity == "STABLE":

                st.success(
                    "Live feature distributions are broadly stable."
                )

            else:

                st.warning(
                    "Live drift severity could not be determined."
                )

            st.markdown("#### Live Drift Summary")

            live_metric1, live_metric2, live_metric3 = (
                st.columns(3)
            )

            live_metric1.metric(
                "Stable Features",
                live_summary["stable_features"],
            )

            live_metric2.metric(
                "Moderate Drift",
                live_summary["moderate_features"],
            )

            live_metric3.metric(
                "Significant Drift",
                live_summary["significant_features"],
            )

            st.markdown("#### Live Feature Drift Results")

            live_display = (
                live_result["results"]
                .sort_values(
                    "psi",
                    ascending=False,
                    na_position="last",
                )
            )

            st.dataframe(
                live_display[
                    [
                        "feature",
                        "psi",
                        "severity",
                        "reference_count",
                        "current_count",
                    ]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )

    except FileNotFoundError:

        st.warning(
            "Live drift analysis cannot run because the "
            "training reference dataset is unavailable "
            "in this deployment."
        )

    except Exception as error:

        st.error(
            f"Live monitoring encountered an error: {error}"
        )

    st.caption(
        "Live PSI is a feature-distribution monitoring signal, "
        "not a direct measurement of fraud detection accuracy. "
        "Model performance monitoring requires verified "
        "transaction labels."
    )
