
import pytest

from src import database


def setup_test_database(tmp_path, monkeypatch):
    """Create an isolated database for each test."""
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    database.initialize_database()
    return db_path


def create_fraud_transaction(source="API"):
    """Create a reusable fraud transaction."""
    return database.save_transaction(
        transaction_time=100000,
        amount=250.50,
        fraud_probability=0.92,
        prediction=1,
        result="FRAUD",
        risk_level="HIGH",
        model_name="Random Forest",
        response_time_ms=12.5,
        source=source,
    )


def test_initialize_database_creates_database(tmp_path, monkeypatch):
    db_path = setup_test_database(tmp_path, monkeypatch)

    assert db_path.exists()


def test_empty_dashboard_metrics_return_zero(tmp_path, monkeypatch):
    setup_test_database(tmp_path, monkeypatch)

    metrics = database.get_dashboard_metrics()

    assert metrics["total_transactions"] == 0
    assert metrics["detected_frauds"] == 0
    assert metrics["legitimate_transactions"] == 0


def test_save_transaction_and_retrieve_it(tmp_path, monkeypatch):
    setup_test_database(tmp_path, monkeypatch)

    transaction_id = database.save_transaction(
        transaction_time=100000,
        amount=250.50,
        fraud_probability=0.92,
        prediction=1,
        result="FRAUD",
        risk_level="HIGH",
        model_name="Random Forest",
        response_time_ms=12.5,
        source="TEST",
    )

    transactions = database.get_transactions()

    assert transaction_id > 0
    assert len(transactions) == 1
    assert transactions.iloc[0]["amount"] == 250.50
    assert transactions.iloc[0]["result"] == "FRAUD"


def test_dashboard_metrics_calculate_correctly(tmp_path, monkeypatch):
    setup_test_database(tmp_path, monkeypatch)

    database.save_transaction(
        transaction_time=100000,
        amount=250,
        fraud_probability=0.95,
        prediction=1,
        result="FRAUD",
        risk_level="HIGH",
        model_name="Random Forest",
        response_time_ms=10,
        source="TEST",
    )

    database.save_transaction(
        transaction_time=100100,
        amount=100,
        fraud_probability=0.02,
        prediction=0,
        result="LEGITIMATE",
        risk_level="LOW",
        model_name="Random Forest",
        response_time_ms=8,
        source="TEST",
    )

    metrics = database.get_dashboard_metrics()

    assert metrics["total_transactions"] == 2
    assert metrics["detected_frauds"] == 1
    assert metrics["legitimate_transactions"] == 1


def test_database_health_check(tmp_path, monkeypatch):
    setup_test_database(tmp_path, monkeypatch)

    health = database.check_database()

    assert health["status"] == "healthy"
    assert health["transaction_count"] == 0


# ============================================================
# FRAUD ALERT MANAGEMENT TESTS
# ============================================================

def test_fraud_prediction_automatically_creates_alert(
    tmp_path, monkeypatch
):
    setup_test_database(tmp_path, monkeypatch)

    transaction_id = create_fraud_transaction(source="API")

    alerts = database.get_alerts()

    assert transaction_id > 0
    assert len(alerts) == 1
    assert alerts.iloc[0]["transaction_id"] == transaction_id
    assert alerts.iloc[0]["status"] == "OPEN"
    assert alerts.iloc[0]["priority"] == "HIGH"


def test_legitimate_transaction_does_not_create_alert(
    tmp_path, monkeypatch
):
    setup_test_database(tmp_path, monkeypatch)

    database.save_transaction(
        transaction_time=100000,
        amount=100,
        fraud_probability=0.02,
        prediction=0,
        result="LEGITIMATE",
        risk_level="LOW",
        model_name="Random Forest",
        response_time_ms=8,
        source="API",
    )

    alerts = database.get_alerts()

    assert len(alerts) == 0


def test_alert_status_assignment_notes_and_audit_history(
    tmp_path, monkeypatch
):
    setup_test_database(tmp_path, monkeypatch)

    transaction_id = create_fraud_transaction()
    alerts = database.get_alerts()

    assert len(alerts) == 1

    alert_id = int(alerts.iloc[0]["alert_id"])

    # Move the alert into investigation.
    database.update_alert_status(
        alert_id,
        "UNDER_REVIEW",
        actor="INVESTIGATOR",
    )

    # Assign the alert to an investigator.
    database.assign_alert(
        alert_id,
        "Fraud Analyst",
        actor="INVESTIGATOR",
    )

    # Add an investigation note.
    database.add_alert_note(
        alert_id,
        "Transaction requires manual investigation.",
        actor="INVESTIGATOR",
    )

    # Resolve the alert with a resolution explanation.
    database.update_alert_status(
        alert_id,
        "RESOLVED",
        actor="INVESTIGATOR",
        resolution="Investigation completed.",
    )

    updated_alerts = database.get_alerts()
    selected = updated_alerts[
        updated_alerts["alert_id"] == alert_id
    ].iloc[0]

    assert selected["transaction_id"] == transaction_id
    assert selected["status"] == "RESOLVED"
    assert selected["assigned_to"] == "Fraud Analyst"
    assert selected["resolution"] == "Investigation completed."

    history = database.get_alert_history(alert_id)

    assert len(history) >= 5


def test_resolving_alert_requires_resolution(
    tmp_path, monkeypatch
):
    setup_test_database(tmp_path, monkeypatch)

    create_fraud_transaction()
    alerts = database.get_alerts()
    alert_id = int(alerts.iloc[0]["alert_id"])

    with pytest.raises(ValueError):
        database.update_alert_status(
            alert_id,
            "RESOLVED",
            actor="INVESTIGATOR",
        )


def test_alert_metrics_update_after_status_change(
    tmp_path, monkeypatch
):
    setup_test_database(tmp_path, monkeypatch)

    create_fraud_transaction()

    metrics = database.get_alert_metrics()

    assert metrics["total_alerts"] == 1
    assert metrics["open_alerts"] == 1
    assert metrics["resolved_alerts"] == 0

    alert_id = int(database.get_alerts().iloc[0]["alert_id"])

    database.update_alert_status(
        alert_id,
        "RESOLVED",
        actor="INVESTIGATOR",
        resolution="Confirmed and resolved.",
    )

    metrics = database.get_alert_metrics()

    assert metrics["total_alerts"] == 1
    assert metrics["open_alerts"] == 0
    assert metrics["resolved_alerts"] == 1
