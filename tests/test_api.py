
import pytest
from fastapi.testclient import TestClient

import api.main as api_module
from src import database as database_module

from api.main import (
    app,
    MODEL_FEATURES,
    calculate_activity_features,
)


client = TestClient(app)



# ============================================================
# RESET TRANSACTION HISTORY AND ISOLATE DATABASE BEFORE EACH TEST
# ============================================================

@pytest.fixture(autouse=True)
def reset_transaction_history(tmp_path, monkeypatch):

    # Redirect API database operations to a temporary database.
    test_database_path = tmp_path / "test_fraud_monitoring.db"

    monkeypatch.setattr(
        database_module,
        "DATABASE_PATH",
        test_database_path
    )

    # Create tables in the temporary database.
    database_module.initialize_database()

    # Reset in-memory transaction history.
    with api_module.history_lock:
        api_module.transaction_history.clear()
        api_module.last_processed_time = None
        api_module.cached_activity_features = None

    yield

    # Reset in-memory transaction history after each test.
    with api_module.history_lock:
        api_module.transaction_history.clear()
        api_module.last_processed_time = None
        api_module.cached_activity_features = None



# ============================================================
# SAMPLE TRANSACTION
# ============================================================

def create_sample_transaction(time=100000, amount=100):

    transaction = {
        "Time": time,
        "Amount": amount,
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    return transaction


# ============================================================
# API TESTS
# ============================================================

def test_health_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_feature_count_and_order():

    assert len(MODEL_FEATURES) == 37

    assert list(
        api_module.model.feature_names_in_
    ) == MODEL_FEATURES


def test_legitimate_transaction_prediction():

    response = client.post(
        "/predict",
        json=create_sample_transaction()
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["model"] == "Random Forest"


def test_fraud_transaction_prediction():

    # Load a known fraud example from the held-out test data.
    import pandas as pd

    test_data = pd.read_csv(
        api_module.BASE_DIR / "data" / "processed" / "test_features.csv"
    )

    fraud = test_data[
        test_data["Class"] == 1
    ].iloc[0]

    transaction = {
        "Time": float(fraud["Time"]),
        "Amount": float(fraud["Amount"]),
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = float(fraud[f"V{i}"])

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_prediction_is_saved_to_database():

    response = client.post(
        "/predict",
        json=create_sample_transaction()
    )

    assert response.status_code == 200

    result = response.json()

    # Verify the API returns a transaction ID.
    assert result["transaction_id"] > 0

    # Verify response latency is recorded.
    assert result["response_time_ms"] >= 0

    # Verify the transaction was saved to SQLite.
    transactions = database_module.get_transactions(limit=10)

    assert len(transactions) == 1
    assert transactions.iloc[0]["source"] == "API"
    assert transactions.iloc[0]["model_name"] == "Random Forest"
    assert transactions.iloc[0]["amount"] == 100



def test_negative_amount_rejected():

    transaction = create_sample_transaction()

    transaction["Amount"] = -10

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422


def test_negative_time_rejected():

    transaction = create_sample_transaction()

    transaction["Time"] = -1

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422


def test_missing_feature_rejected():

    transaction = create_sample_transaction()

    del transaction["V14"]

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422


# ============================================================
# HISTORICAL ACTIVITY TESTS
# ============================================================

def test_same_timestamp_gets_identical_historical_features():

    first = calculate_activity_features(100)

    second = calculate_activity_features(100)

    assert first == second

    assert first == {
        "txn_count_prev_60s": 0,
        "txn_count_prev_300s": 0,
        "txn_count_prev_3600s": 0,
    }


def test_same_timestamp_transactions_counted_for_next_timestamp():

    calculate_activity_features(100)
    calculate_activity_features(100)

    next_transaction = calculate_activity_features(130)

    assert next_transaction == {
        "txn_count_prev_60s": 2,
        "txn_count_prev_300s": 2,
        "txn_count_prev_3600s": 2,
    }


def test_out_of_order_timestamp_rejected():

    calculate_activity_features(200)

    with pytest.raises(api_module.HTTPException) as error:

        calculate_activity_features(100)

    assert error.value.status_code == 422


def test_exact_60_second_boundary_included():

    calculate_activity_features(100)

    result = calculate_activity_features(160)

    assert result["txn_count_prev_60s"] == 1


def test_transaction_older_than_3600_seconds_removed():

    calculate_activity_features(100)

    result = calculate_activity_features(3701)

    assert result["txn_count_prev_3600s"] == 0
