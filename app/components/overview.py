
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import (
    get_dashboard_metrics,
    get_transactions,
)


def render_overview():

    # --------------------------------------------------------
    # LOAD DATABASE INFORMATION
    # --------------------------------------------------------

    metrics = get_dashboard_metrics()
    transactions = get_transactions(limit=5000)

    total = metrics["total_transactions"]
    frauds = metrics["detected_frauds"]
    legitimate = metrics["legitimate_transactions"]

    fraud_rate = (
        (frauds / total) * 100
        if total > 0
        else 0
    )

    # --------------------------------------------------------
    # EXECUTIVE KPI SECTION
    # --------------------------------------------------------

    st.subheader("Transaction Monitoring")

    st.caption(
        "Live monitoring of transactions processed and recorded "
        "by the fraud detection API."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Total Transactions",
        value=f"{total:,}",
    )

    col2.metric(
        label="Fraud Alerts",
        value=f"{frauds:,}",
    )

    col3.metric(
        label="Fraud Detection Rate",
        value=f"{fraud_rate:.2f}%",
    )

    col4.metric(
        label="Average API Latency",
        value=f"{metrics['average_response_time_ms']:.2f} ms",
    )

    st.divider()

    # --------------------------------------------------------
    # ADDITIONAL METRICS
    # --------------------------------------------------------

    amount_col, probability_col = st.columns(2)

    amount_col.metric(
        label="Flagged Transaction Amount",
        value=f"{metrics['flagged_amount']:,.2f}",
    )

    probability_col.metric(
        label="Average Fraud Probability",
        value=f"{metrics['average_fraud_probability'] * 100:.2f}%",
    )

    st.divider()

    # --------------------------------------------------------
    # EMPTY DATABASE STATE
    # --------------------------------------------------------

    if transactions.empty:

        st.info(
            "No transactions have been recorded yet. "
            "Start the API and submit transactions to populate "
            "the monitoring dashboard."
        )

        return

    # --------------------------------------------------------
    # FRAUD VS LEGITIMATE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Transaction Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        distribution = pd.DataFrame({
            "Transaction Type": [
                "Fraud",
                "Legitimate",
            ],
            "Count": [
                frauds,
                legitimate,
            ],
        })

        fig_distribution = px.pie(
            distribution,
            names="Transaction Type",
            values="Count",
            hole=0.62,
            title="Fraud vs Legitimate Transactions",
        )

        fig_distribution.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        fig_distribution.update_layout(
            legend_title_text="Classification",
            margin=dict(t=60, b=20, l=20, r=20),
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # RISK LEVEL DISTRIBUTION
    # --------------------------------------------------------

    with chart_col2:

        risk_distribution = (
            transactions["risk_level"]
            .value_counts()
            .rename_axis("Risk Level")
            .reset_index(name="Count")
        )

        fig_risk = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            title="Transaction Risk Distribution",
            text="Count",
        )

        fig_risk.update_layout(
            showlegend=False,
            margin=dict(t=60, b=20, l=20, r=20),
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------------
    # TRANSACTION ACTIVITY OVER TIME
    # --------------------------------------------------------

    st.subheader("Transaction Activity")

    activity = transactions.copy()

    activity["created_at"] = pd.to_datetime(
        activity["created_at"],
        utc=True,
        errors="coerce",
    )

    activity = activity.dropna(
        subset=["created_at"]
    )

    if not activity.empty:

        activity["time_bucket"] = (
            activity["created_at"].dt.floor("5min")
        )

        timeline = (
            activity.groupby(
                ["time_bucket", "result"]
            )
            .size()
            .reset_index(name="Transactions")
        )

        fig_activity = px.line(
            timeline,
            x="time_bucket",
            y="Transactions",
            color="result",
            markers=True,
            title="Transactions Recorded Over Time",
        )

        fig_activity.update_layout(
            xaxis_title="Recorded Time (UTC)",
            yaxis_title="Transaction Count",
            legend_title="Classification",
        )

        st.plotly_chart(
            fig_activity,
            use_container_width=True,
        )

    else:

        st.info(
            "Transaction timestamps are not available for the activity chart."
        )

    st.divider()

    # --------------------------------------------------------
    # RECENT FRAUD ALERTS
    # --------------------------------------------------------

    st.subheader("Recent Fraud Alerts")

    fraud_alerts = transactions[
        transactions["prediction"] == 1
    ].copy()

    if not fraud_alerts.empty:

        fraud_alerts["created_at"] = pd.to_datetime(
            fraud_alerts["created_at"],
            utc=True,
            errors="coerce",
        )

        display_columns = [
            "id",
            "created_at",
            "amount",
            "fraud_probability",
            "risk_level",
            "model_name",
            "source",
        ]

        st.dataframe(
            fraud_alerts[display_columns].head(10),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No fraud alerts have been recorded yet."
        )
