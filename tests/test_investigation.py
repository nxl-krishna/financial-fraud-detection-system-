
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.components import investigation


def test_investigation_renders_without_errors(monkeypatch):

    sample_transactions = pd.DataFrame([
        {
            "id": 1,
            "created_at": "2026-11-15 10:00:00",
            "amount": 250.0,
            "fraud_probability": 0.0001,
            "result": "LEGITIMATE",
            "prediction": 0,
            "risk_level": "LOW",
            "model_name": "Random Forest",
            "model_version": "1.0.0",
            "source": "TEST",
            "transaction_time": 100000.0,
            "response_time_ms": 35.0,
            "feature_values_json": None,
        },
        {
            "id": 2,
            "created_at": "2026-11-15 10:01:00",
            "amount": 500.0,
            "fraud_probability": 0.85,
            "result": "FRAUD",
            "prediction": 1,
            "risk_level": "HIGH",
            "model_name": "Random Forest",
            "model_version": "1.0.0",
            "source": "TEST",
            "transaction_time": 100060.0,
            "response_time_ms": 40.0,
            "feature_values_json": None,
        },
    ])

    monkeypatch.setattr(
        investigation,
        "get_transactions",
        lambda limit: sample_transactions.copy(),
    )

    app = AppTest.from_string(
        """
from app.components.investigation import render_investigation

render_investigation()
"""
    )

    app.run(timeout=30)

    assert len(app.exception) == 0
    assert len(app.selectbox) >= 3
