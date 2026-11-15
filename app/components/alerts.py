
import streamlit as st
import pandas as pd

from src.database import (
    get_alerts,
    get_alert_metrics,
    update_alert_status,
    assign_alert,
    add_alert_note,
    get_alert_history,
)


# ============================================================
# PAGE HEADER
# ============================================================

def render_alert_management():

    st.title("Fraud Alert Management")

    st.caption(
        "Centralized fraud alert operations, investigator "
        "assignment, case resolution and audit tracking."
    )

    st.divider()

    # ========================================================
    # ALERT METRICS
    # ========================================================

    metrics = get_alert_metrics()

    st.subheader("Alert Operations Overview")

    metric_columns = st.columns(5)

    metric_items = [
        ("Total Alerts", metrics["total_alerts"]),
        ("Open", metrics["open_alerts"]),
        ("Under Review", metrics["under_review_alerts"]),
        ("Escalated", metrics["escalated_alerts"]),
        ("Resolved", metrics["resolved_alerts"]),
    ]

    for column, (label, value) in zip(
        metric_columns,
        metric_items
    ):

        column.metric(
            label,
            f"{value:,}"
        )

    st.metric(
        "False Positive Alerts",
        f"{metrics['false_positive_alerts']:,}"
    )

    st.divider()

    # ========================================================
    # FILTERS
    # ========================================================

    st.subheader("Alert Investigation Queue")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        status_filter = st.selectbox(
            "Alert Status",
            [
                "ALL",
                "OPEN",
                "UNDER_REVIEW",
                "ESCALATED",
                "RESOLVED",
                "FALSE_POSITIVE",
            ]
        )

    with filter_col2:

        priority_filter = st.selectbox(
            "Priority",
            [
                "ALL",
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ]
        )

    with filter_col3:

        search_transaction = st.text_input(
            "Search Transaction ID",
            placeholder="Enter transaction ID"
        )

    # ========================================================
    # FETCH ALERTS
    # ========================================================

    try:

        alerts = get_alerts(
            status=(
                None if status_filter == "ALL"
                else status_filter
            ),
            priority=(
                None if priority_filter == "ALL"
                else priority_filter
            ),
        )

    except Exception as error:

        st.error(f"Unable to retrieve alerts: {error}")
        return

    if not alerts.empty and search_transaction.strip():

        try:

            transaction_id = int(search_transaction)

            alerts = alerts[
                alerts["transaction_id"] == transaction_id
            ]

        except ValueError:

            alerts = alerts.iloc[0:0]

    st.caption(
        f"Matching alerts: {len(alerts):,}"
    )

    # ========================================================
    # ALERT TABLE
    # ========================================================

    if alerts.empty:

        st.info(
            "No alerts match the selected filters. "
            "New fraud predictions will automatically "
            "appear here."
        )

        return

    display_columns = [
        "alert_id",
        "transaction_id",
        "status",
        "priority",
        "assigned_to",
        "amount",
        "fraud_probability",
        "risk_level",
        "alert_created_at",
    ]

    display_alerts = alerts[display_columns].copy()

    display_alerts["fraud_probability"] = (
        display_alerts["fraud_probability"] * 100
    ).round(2)

    display_alerts["amount"] = (
        display_alerts["amount"].round(2)
    )

    display_alerts = display_alerts.rename(
    columns={
        "alert_id": "Alert ID",
        "transaction_id": "Transaction ID",
        "status": "Status",
        "priority": "Priority",
        "assigned_to": "Assigned Investigator",
        "amount": "Amount",
        "fraud_probability": "Fraud Probability (%)",
        "risk_level": "Risk Level",
        "model_name": "Model",
        "model_version": "Model Version",
        "alert_created_at": "Created At",
    }
)

    st.dataframe(
        display_alerts,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ========================================================
    # SELECT ALERT
    # ========================================================

    st.subheader("Investigate Individual Alert")

    alert_options = {
        (
            f"Alert #{int(row.alert_id)} "
            f"| Transaction #{int(row.transaction_id)} "
            f"| {row.status}"
        ): int(row.alert_id)
        for row in alerts.itertuples()
    }

    selected_label = st.selectbox(
        "Select Alert",
        list(alert_options.keys())
    )

    selected_alert_id = alert_options[selected_label]

    selected = alerts[
        alerts["id"] == selected_alert_id
    ] if "id" in alerts.columns else alerts[
        alerts["alert_id"] == selected_alert_id
    ]

    if selected.empty:
        st.error("Selected alert could not be found.")
        return

    alert = selected.iloc[0]

    # ========================================================
    # ALERT DETAILS
    # ========================================================

    st.subheader("Alert Details")

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    detail_col1.metric(
        "Transaction Amount",
        f"₹{alert['amount']:,.2f}"
    )

    detail_col2.metric(
        "Fraud Probability",
        f"{alert['fraud_probability'] * 100:.2f}%"
    )

    detail_col3.metric(
        "Risk Level",
        alert["risk_level"]
    )

    st.write("**Current Status:**", alert["status"])

    st.write(
        "**Assigned Investigator:**",
        alert["assigned_to"] or "Unassigned"
    )

    st.write(
        "**Model:**",
        alert["model_name"]
    )

    st.write(
        "**Alert Created:**",
        alert["alert_created_at"]
    )

    st.divider()

    # ========================================================
    # UPDATE ALERT
    # ========================================================

    st.subheader("Update Investigation")

    status_options = [
        "OPEN",
        "UNDER_REVIEW",
        "ESCALATED",
        "RESOLVED",
        "FALSE_POSITIVE",
    ]

    with st.form("update_alert_form"):

        new_status = st.selectbox(
            "Update Status",
            status_options,
            index=status_options.index(alert["status"])
        )

        resolution = st.text_area(
            "Resolution / Investigation Reason",
            placeholder=(
                "Required when resolving an alert "
                "or marking it as a false positive."
            )
        )

        submitted = st.form_submit_button(
            "Update Alert Status",
            use_container_width=True
        )

        if submitted:

            try:

                update_alert_status(
                    alert_id=selected_alert_id,
                    new_status=new_status,
                    actor="DASHBOARD_INVESTIGATOR",
                    resolution=(
                        resolution.strip() or None
                    ),
                )

                st.success(
                    "Alert status updated successfully."
                )

                st.rerun()

            except ValueError as error:

                st.error(str(error))

            except Exception as error:

                st.error(
                    f"Unable to update alert: {error}"
                )

    st.divider()

    # ========================================================
    # INVESTIGATOR ASSIGNMENT
    # ========================================================

    st.subheader("Investigator Assignment")

    with st.form("assign_alert_form"):

        investigator = st.text_input(
            "Investigator Name",
            value=alert["assigned_to"] or "",
            placeholder="Enter investigator name"
        )

        assign_submitted = st.form_submit_button(
            "Assign Investigator",
            use_container_width=True
        )

        if assign_submitted:

            try:

                assign_alert(
                    alert_id=selected_alert_id,
                    assigned_to=investigator,
                    actor="DASHBOARD_ADMIN",
                )

                st.success(
                    "Investigator assigned successfully."
                )

                st.rerun()

            except ValueError as error:

                st.error(str(error))

            except Exception as error:

                st.error(
                    f"Assignment failed: {error}"
                )

    st.divider()

    # ========================================================
    # INVESTIGATOR NOTES
    # ========================================================

    st.subheader("Investigation Notes")

    with st.form("alert_note_form"):

        note = st.text_area(
            "Add Investigation Note",
            placeholder=(
                "Document evidence, observations, "
                "investigation findings or next steps."
            )
        )

        note_submitted = st.form_submit_button(
            "Save Investigation Note",
            use_container_width=True
        )

        if note_submitted:

            try:

                add_alert_note(
                    alert_id=selected_alert_id,
                    note=note,
                    actor="DASHBOARD_INVESTIGATOR",
                )

                st.success(
                    "Investigation note saved."
                )

                st.rerun()

            except ValueError as error:

                st.error(str(error))

            except Exception as error:

                st.error(
                    f"Unable to save note: {error}"
                )

    st.divider()

    # ========================================================
    # AUDIT HISTORY
    # ========================================================

    st.subheader("Alert Audit History")

    history = get_alert_history(selected_alert_id)

    if history.empty:

        st.info("No audit history available.")

    else:

        history_columns = [
            "created_at",
            "event_type",
            "previous_status",
            "new_status",
            "actor",
            "note",
        ]

        st.dataframe(
            history[history_columns],
            use_container_width=True,
            hide_index=True,
        )
