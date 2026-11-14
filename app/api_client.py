
import os
from typing import Any

import httpx


API_BASE_URL = os.getenv(
    "FRAUD_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

HEALTH_URL = f"{API_BASE_URL}/"
PREDICTION_URL = f"{API_BASE_URL}/predict"


def check_api_health() -> dict[str, Any]:
    """
    Check whether the fraud detection API is reachable.
    """

    try:
        response = httpx.get(
            HEALTH_URL,
            timeout=5.0
        )

        response.raise_for_status()

        data = response.json()

        return {
            "online": True,
            "message": data.get(
                "message",
                "API is running."
            ),
            "model": data.get(
                "model",
                "Unknown"
            ),
            "status": data.get(
                "status",
                "unknown"
            ),
        }

    except httpx.TimeoutException:
        return {
            "online": False,
            "message": "API request timed out.",
            "model": None,
            "status": "offline",
        }

    except httpx.HTTPError as error:
        return {
            "online": False,
            "message": f"API connection failed: {error}",
            "model": None,
            "status": "offline",
        }

    except ValueError:
        return {
            "online": False,
            "message": "API returned an invalid response.",
            "model": None,
            "status": "error",
        }


def predict_transaction(
    transaction: dict[str, Any]
) -> dict[str, Any]:
    """
    Send a transaction to FastAPI and return its prediction.
    """

    try:
        response = httpx.post(
            PREDICTION_URL,
            json=transaction,
            timeout=20.0
        )

        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as error:

        try:
            detail = error.response.json().get(
                "detail",
                error.response.text
            )
        except ValueError:
            detail = error.response.text

        raise RuntimeError(
            f"Prediction API error "
            f"(HTTP {error.response.status_code}): {detail}"
        ) from error

    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Prediction request timed out."
        ) from error

    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Could not connect to prediction API: {error}"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "Prediction API returned invalid JSON."
        ) from error
