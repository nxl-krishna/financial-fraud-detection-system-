
import pandas as pd

from app.components import simulator


def make_sample_dataset():

    data = pd.DataFrame({
        "Time": [100000.0, 100060.0, 100120.0],
        "Amount": [100.0, 250.0, 500.0],
        "Class": [0, 1, 0],
    })

    for i in range(1, 29):
        data[f"V{i}"] = [0.0, 0.1, -0.1]

    return data


def test_simulation_data_loads_correctly(tmp_path, monkeypatch):

    sample_data = make_sample_dataset()

    sample_path = tmp_path / "test_features.csv"
    sample_data.to_csv(sample_path, index=False)

    monkeypatch.setattr(
        simulator,
        "TEST_DATA_PATH",
        sample_path,
    )

    simulator.load_simulation_data.clear()

    data = simulator.load_simulation_data()

    assert not data.empty
    assert "Time" in data.columns
    assert "Amount" in data.columns
    assert "Class" in data.columns

    assert data["Time"].is_monotonic_increasing
    assert len(data) == 3


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

    transaction = simulator.prepare_api_transaction(row)

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

    metrics = simulator.calculate_simulation_metrics(results)

    assert metrics["transactions"] == 4
    assert metrics["fraud_alerts"] == 2
    assert metrics["actual_frauds"] == 2

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
