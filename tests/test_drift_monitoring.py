import numpy as np
import pandas as pd
import pytest

import src.drift_monitoring as drift


def test_calculate_psi_detects_stable_distribution():
    rng = np.random.default_rng(42)

    reference = rng.normal(0, 1, 1000)
    current = rng.normal(0, 1, 1000)

    result = drift.calculate_psi(reference, current)

    assert result["valid"] is True
    assert result["psi"] >= 0
    assert result["current_count"] == 1000


def test_calculate_psi_detects_distribution_shift():
    rng = np.random.default_rng(42)

    reference = rng.normal(0, 1, 1000)
    current = rng.normal(3, 1, 1000)

    result = drift.calculate_psi(reference, current)

    assert result["valid"] is True
    assert result["psi"] > 0.25


def test_calculate_psi_rejects_small_samples():
    reference = np.arange(50)
    current = np.arange(50)

    with pytest.raises(ValueError):
        drift.calculate_psi(reference, current)


def test_classify_drift_thresholds():
    assert drift.classify_drift(0.05) == "STABLE"
    assert drift.classify_drift(0.15) == "MODERATE"
    assert drift.classify_drift(0.30) == "SIGNIFICANT"
    assert drift.classify_drift(None) == "UNAVAILABLE"


def test_calculate_feature_drift(monkeypatch):
    rng = np.random.default_rng(42)

    features = ["feature_a", "feature_b"]
    monkeypatch.setattr(
        drift,
        "load_model_features",
        lambda: features,
    )

    reference = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 500),
        "feature_b": rng.normal(0, 1, 500),
    })

    current = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 500),
        "feature_b": rng.normal(3, 1, 500),
    })

    results = drift.calculate_feature_drift(reference, current)

    assert len(results) == 2
    assert set(results["feature"]) == set(features)
    assert results["psi"].notna().all()
    assert results["psi"].ge(0).all()

    shifted = results.loc[
        results["feature"] == "feature_b"
    ].iloc[0]

    assert shifted["severity"] == "SIGNIFICANT"


def test_calculate_feature_drift_handles_insufficient_data(monkeypatch):
    features = ["feature_a"]

    monkeypatch.setattr(
        drift,
        "load_model_features",
        lambda: features,
    )

    reference = pd.DataFrame({
        "feature_a": np.arange(200)
    })

    current = pd.DataFrame({
        "feature_a": np.arange(50)
    })

    results = drift.calculate_feature_drift(reference, current)

    assert len(results) == 1
    assert results.iloc[0]["severity"] == "INSUFFICIENT_DATA"


def test_calculate_overall_drift():
    results = pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "severity": [
            "STABLE",
            "MODERATE",
            "SIGNIFICANT",
            "STABLE",
        ],
    })

    summary = drift.calculate_overall_drift(results)

    assert summary["severity"] == "SIGNIFICANT"
    assert summary["features_monitored"] == 4
    assert summary["stable_features"] == 2
    assert summary["moderate_features"] == 1
    assert summary["significant_features"] == 1


def test_calculate_overall_drift_with_insufficient_data():
    results = pd.DataFrame({
        "feature": ["a", "b"],
        "severity": [
            "INSUFFICIENT_DATA",
            "UNAVAILABLE",
        ],
    })

    summary = drift.calculate_overall_drift(results)

    assert summary["severity"] == "INSUFFICIENT_DATA"
    assert summary["features_monitored"] == 2
