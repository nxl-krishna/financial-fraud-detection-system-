
import httpx
import pytest

from app import api_client


def test_health_check_success(monkeypatch):

    def mock_get(url, timeout):
        return httpx.Response(
            200,
            json={
                "message": "API is running.",
                "model": "Random Forest",
                "status": "healthy",
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(
        api_client.httpx,
        "get",
        mock_get
    )

    result = api_client.check_api_health()

    assert result["online"] is True
    assert result["model"] == "Random Forest"
    assert result["status"] == "healthy"


def test_health_check_connection_failure(monkeypatch):

    def mock_get(url, timeout):
        raise httpx.ConnectError(
            "Connection refused",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(
        api_client.httpx,
        "get",
        mock_get
    )

    result = api_client.check_api_health()

    assert result["online"] is False
    assert result["status"] == "offline"


def test_prediction_success(monkeypatch):

    def mock_post(url, json, timeout):
        return httpx.Response(
            200,
            json={
                "transaction_id": 1,
                "prediction": 1,
                "result": "FRAUD",
                "fraud_probability": 0.95,
                "fraud_threshold": 0.244,
                "risk_level": "HIGH",
                "model": "Random Forest",
                "response_time_ms": 12.5,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        api_client.httpx,
        "post",
        mock_post
    )

    result = api_client.predict_transaction(
        {"Time": 100000, "Amount": 250}
    )

    assert result["prediction"] == 1
    assert result["result"] == "FRAUD"
    assert result["risk_level"] == "HIGH"
    assert result["transaction_id"] == 1


def test_prediction_api_error(monkeypatch):

    def mock_post(url, json, timeout):
        return httpx.Response(
            422,
            json={
                "detail": "Invalid transaction"
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        api_client.httpx,
        "post",
        mock_post
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid transaction"
    ):
        api_client.predict_transaction(
            {"Time": -1, "Amount": 250}
        )
