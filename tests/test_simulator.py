
import pandas as pd

from app.components.simulator import (
    load_simulation_data,
    prepare_api_transaction,
    calculate_simulation_metrics,
)


def test_simulation_data_loads_correctly():

    data = load_simulation_data()

    assert not data.empty
    assert "Time" in data.columns
    assert "Amount" in data.columns
    assert "Class" in data.columns

    assert data["Time"].is_monotonic_increasing


def test_api_transaction_excludes_actual_class():

    row = pd.Series({
        "Time": 100000.0,
        "Amount": 250.0,
        "Class": 1,
        **{
            f"V{i}": 0.0
            for i in range(1, 29)
        },
    })

    transaction = prepare_api_transaction(row)

    assert "Class" not in transaction
    assert "Time" in transaction
    assert "Amount" in transaction
    assert len(transaction) == 30


def test_simulation_metrics_calculated_correctly():

    results = [
        {
            "actual_class": 1,
            "prediction": 1,
        },
        {
            "actual_class": 0,
            "prediction": 1,
        },
        {
            "actual_class": 1,
            "prediction": 0,
        },
        {
            "actual_class": 0,
            "prediction": 0,
        },
    ]

    metrics = calculate_simulation_metrics(results)

    assert metrics["transactions"] == 4
    assert metrics["fraud_alerts"] == 2
    assert metrics["actual_frauds"] == 2

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
