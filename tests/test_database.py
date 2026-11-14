
import pandas as pd

from src import database


def test_initialize_database_creates_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    database.initialize_database()

    assert db_path.exists()


def test_empty_dashboard_metrics_return_zero(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    database.initialize_database()
    metrics = database.get_dashboard_metrics()

    assert metrics["total_transactions"] == 0
    assert metrics["detected_frauds"] == 0
    assert metrics["legitimate_transactions"] == 0


def test_save_transaction_and_retrieve_it(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    database.initialize_database()

    transaction_id = database.save_transaction(
        transaction_time=100000,
        amount=250.50,
        fraud_probability=0.92,
        prediction=1,
        result="FRAUD",
        risk_level="HIGH",
        model_name="Random Forest",
        response_time_ms=12.5,
        source="TEST"
    )

    transactions = database.get_transactions()

    assert transaction_id > 0
    assert len(transactions) == 1
    assert transactions.iloc[0]["amount"] == 250.50
    assert transactions.iloc[0]["result"] == "FRAUD"


def test_dashboard_metrics_calculate_correctly(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    database.initialize_database()

    database.save_transaction(
        transaction_time=100000,
        amount=250,
        fraud_probability=0.95,
        prediction=1,
        result="FRAUD",
        risk_level="HIGH",
        model_name="Random Forest",
        response_time_ms=10,
        source="TEST"
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
        source="TEST"
    )

    metrics = database.get_dashboard_metrics()

    assert metrics["total_transactions"] == 2
    assert metrics["detected_frauds"] == 1
    assert metrics["legitimate_transactions"] == 1


def test_database_health_check(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    database.initialize_database()

    health = database.check_database()

    assert health["status"] == "healthy"
    assert health["transaction_count"] == 0
