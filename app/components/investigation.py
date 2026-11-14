
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import get_transactions


def render_investigation():

    st.subheader("Fraud Investigation Center")

    st.caption(
        "Search, filter, and investigate transactions "
        "processed by the fraud detection engine."
    )

    transactions = get_transactions(limit=5000)

    if transactions.empty:
        st.info(
            "No transactions are available for investigation yet."
        )
        return

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    transactions = transactions.copy()

    transactions["fraud_probability"] = pd.to_numeric(
        transactions["fraud_probability"],
        errors="coerce",
    )

    transactions["amount"] = pd.to_numeric(
        transactions["amount"],
        errors="coerce",
    )

    transactions["id"] = transactions["id"].astype(str)

    # --------------------------------------------------------
    # FILTER CONTROLS
    # --------------------------------------------------------

    st.markdown("### Investigation Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        classification = st.selectbox(
            "Transaction Classification",
            options=[
                "All",
                "FRAUD",
                "LEGITIMATE",
            ],
        )

    with filter_col2:

        risk_level = st.selectbox(
            "Risk Level",
            options=[
                "All",
                "HIGH",
                "LOW",
            ],
        )

    with filter_col3:

        search_id = st.text_input(
            "Search Transaction ID",
            placeholder="Enter transaction ID",
        ).strip()

    min_probability, max_probability = st.slider(
        "Fraud Probability Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered = transactions.copy()

    if classification != "All":

        filtered = filtered[
            filtered["result"] == classification
        ]

    if risk_level != "All":

        filtered = filtered[
            filtered["risk_level"] == risk_level
        ]

    if search_id:

        filtered = filtered[
            filtered["id"].str.contains(
                search_id,
                case=False,
                regex=False,
            )
        ]

    filtered = filtered[
        filtered["fraud_probability"].between(
            min_probability,
            max_probability,
        )
    ]

    st.divider()

    # --------------------------------------------------------
    # FILTER RESULTS SUMMARY
    # --------------------------------------------------------

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Matching Transactions",
        f"{len(filtered):,}",
    )

    result_col2.metric(
        "Matching Fraud Alerts",
        f"{int((filtered['prediction'] == 1).sum()):,}",
    )

    result_col3.metric(
        "Average Fraud Probability",
        f"{filtered['fraud_probability'].mean() * 100:.2f}%"
        if not filtered.empty
        else "N/A",
    )

    # --------------------------------------------------------
    # TRANSACTION TABLE
    # --------------------------------------------------------

    st.markdown("### Transaction Records")

    display_columns = [
        "id",
        "created_at",
        "amount",
        "fraud_probability",
        "result",
        "risk_level",
        "model_name",
        "source",
    ]

    if filtered.empty:

        st.warning(
            "No transactions match the selected filters."
        )
        return

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "Transaction ID",
            "created_at": "Recorded At",
            "amount": st.column_config.NumberColumn(
                "Amount",
                format="%.2f",
            ),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "result": "Classification",
            "risk_level": "Risk Level",
            "model_name": "Model",
            "source": "Source",
        },
    )

    st.divider()

    # --------------------------------------------------------
    # FRAUD PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("### Fraud Probability Analysis")

    fig = px.histogram(
        filtered,
        x="fraud_probability",
        color="result",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        title="Fraud Probability Distribution",
        labels={
            "fraud_probability": "Fraud Probability",
            "count": "Transaction Count",
            "result": "Classification",
        },
    )

    fig.update_layout(
        xaxis=dict(
            tickformat=".0%",
            title="Fraud Probability",
        ),
        yaxis_title="Transaction Count",
        legend_title="Classification",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # INDIVIDUAL TRANSACTION INVESTIGATION
    # --------------------------------------------------------

    st.markdown("### Investigate Individual Transaction")

    transaction_ids = filtered["id"].tolist()

    selected_id = st.selectbox(
        "Select Transaction ID",
        options=transaction_ids,
        key="investigation_transaction_id",
    )

    selected = filtered[
        filtered["id"] == selected_id
    ].iloc[0]

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    detail_col1.metric(
        "Transaction Amount",
        f"{selected['amount']:,.2f}",
    )

    detail_col2.metric(
        "Fraud Probability",
        f"{selected['fraud_probability'] * 100:.2f}%",
    )

    detail_col3.metric(
        "Classification",
        selected["result"],
    )

    st.markdown("#### Transaction Details")

    details = {
        "Transaction ID": selected["id"],
        "Recorded At (UTC)": str(selected["created_at"]),
        "Transaction Time": selected["transaction_time"],
        "Amount": selected["amount"],
        "Fraud Probability": selected["fraud_probability"],
        "Prediction": selected["prediction"],
        "Risk Level": selected["risk_level"],
        "Model": selected["model_name"],
        "API Response Time (ms)": selected["response_time_ms"],
        "Source": selected["source"],
    }

    st.json(details)
