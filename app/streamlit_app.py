
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.api_client import check_api_health

from app.components.overview import render_overview
from app.components.investigation import render_investigation
from app.components.model_monitoring import render_model_monitoring
from app.components.simulator import render_simulator
from app.components.alerts import render_alert_management

from src.database import (
    initialize_database,
    check_database,
    get_dashboard_metrics,
    get_alert_metrics,
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Detection Command Center",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: clamp(1rem, 3vw, 3rem);
            padding-right: clamp(1rem, 3vw, 3rem);
            max-width: 100%;
            overflow: visible;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        .main-header {
            font-size: clamp(1.5rem, 3vw, 2.2rem);
            font-weight: 750;
            letter-spacing: -0.8px;
            line-height: 1.3;
            margin-bottom: 0.4rem;
            overflow-wrap: anywhere;
        }

        .main-subtitle {
            color: gray;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
            overflow-wrap: anywhere;
        }

        .section-label {
            font-weight: 650;
            letter-spacing: 0.5px;
        }

        .dashboard-footer {
            width: 100%;
            box-sizing: border-box;
            white-space: normal;
            overflow-wrap: anywhere;
            line-height: 1.6;
            padding: 1rem 0;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SQLITE DATABASE
# ============================================================

try:

    initialize_database()

    database_ready = True

except Exception as error:

    database_ready = False

    st.error(
        f"Database initialization failed: {error}"
    )


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns([5, 1])

with header_col1:

    st.markdown(
        '<div class="main-header">'
        '🔐 Financial Fraud Detection Command Center'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-subtitle">'
        'AI-powered transaction monitoring, fraud investigation '
        'and model performance analytics'
        '</div>',
        unsafe_allow_html=True,
    )


with header_col2:

    if st.button(
        "↻ Refresh",
        use_container_width=True,
    ):
        st.rerun()


st.divider()


# ============================================================
# SIDEBAR: APPLICATION STATUS
# ============================================================

st.sidebar.title("Fraud Intelligence")

st.sidebar.caption(
    "Financial Transaction Monitoring Platform"
)

st.sidebar.divider()

st.sidebar.markdown("**SYSTEM STATUS**")


# API status
api_status = check_api_health()

if api_status["online"]:

    st.sidebar.success("Prediction API Online")

    st.sidebar.caption(
        f"Model: {api_status['model']}"
    )

else:

    st.sidebar.error("Prediction API Offline")

    st.sidebar.caption(
        api_status["message"]
    )


# Database status and metrics
if database_ready:

    try:

        db_health = check_database()

        metrics = get_dashboard_metrics()

        alert_metrics = get_alert_metrics()

        st.sidebar.success("SQLite Database Healthy")

        st.sidebar.metric(
            "Recorded Transactions",
            f"{metrics['total_transactions']:,}",
        )

        st.sidebar.metric(
            "Fraud Predictions",
            f"{metrics['detected_frauds']:,}",
        )

        st.sidebar.metric(
            "Open Investigation Alerts",
            f"{alert_metrics['open_alerts']:,}",
        )

    except Exception as error:

        st.sidebar.error(
            f"Database unavailable: {error}"
        )

else:

    st.sidebar.error("Database Not Ready")


st.sidebar.divider()


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown("**WORKSPACE NAVIGATION**")

selected_page = st.sidebar.radio(
    "Select Dashboard",
    options=[
        "Executive Overview",
        "Fraud Investigation",
        "Fraud Alert Management",
        "Live Transaction Simulator",
        "Model Performance",
    ],
    label_visibility="collapsed",
)


# ============================================================
# NAVIGATION CONTENT
# ============================================================

if selected_page == "Executive Overview":

    render_overview()


elif selected_page == "Fraud Investigation":

    render_investigation()


elif selected_page == "Fraud Alert Management":

    render_alert_management()


elif selected_page == "Live Transaction Simulator":

    render_simulator()


elif selected_page == "Model Performance":

    render_model_monitoring()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Financial Fraud Detection System | "
    "Random Forest • FastAPI • SQLite • Streamlit • Plotly"
)

st.caption(
    "Portfolio simulation using an anonymized historical "
    "credit card transaction dataset. "
    "Not a production banking system."
)
