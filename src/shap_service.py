
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"


# ============================================================
# LOAD MODEL AND SHAP EXPLAINER
# ============================================================

@lru_cache(maxsize=1)
def get_model():

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "Model does not contain feature names."
        )

    return model


@lru_cache(maxsize=1)
def get_explainer():

    model = get_model()

    return shap.TreeExplainer(model)


# ============================================================
# VALIDATE STORED FEATURE VECTOR
# ============================================================

def prepare_feature_frame(feature_values_json):

    if feature_values_json is None:
        raise ValueError(
            "SHAP explanation is unavailable for this transaction. "
            "Its original model features were not stored."
        )

    if isinstance(feature_values_json, str):

        try:
            feature_values = json.loads(feature_values_json)

        except json.JSONDecodeError as error:
            raise ValueError(
                "Stored feature data is not valid JSON."
            ) from error

    elif isinstance(feature_values_json, dict):

        feature_values = feature_values_json

    else:
        raise ValueError(
            "Invalid stored feature data."
        )

    if not isinstance(feature_values, dict):
        raise ValueError(
            "Stored features must be a dictionary."
        )

    model = get_model()

    feature_columns = list(model.feature_names_in_)

    if set(feature_values.keys()) != set(feature_columns):
        raise ValueError(
            "Stored feature names do not match the model schema."
        )

    # Always use the exact model feature order.
    feature_frame = pd.DataFrame(
        [
            {
                column: feature_values[column]
                for column in feature_columns
            }
        ],
        columns=feature_columns
    )

    # Reject missing, non-numeric and infinite values.
    feature_frame = feature_frame.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
        raise ValueError(
            "Stored features contain missing or non-finite values."
        )

    return feature_frame


# ============================================================
# NORMALIZE SHAP OUTPUT FOR THE FRAUD CLASS
# ============================================================

def get_fraud_shap_values(explainer, feature_frame):

    raw_values = explainer.shap_values(
        feature_frame,
        check_additivity=False
    )

    # Compatibility with different SHAP output formats.
    if isinstance(raw_values, list):

        values = np.asarray(raw_values[1])[0]

    else:

        values = np.asarray(raw_values)

        if values.ndim == 3:
            values = values[0, :, 1]

        elif values.ndim == 2:
            values = values[0]

        elif values.ndim == 1:
            pass

        else:
            raise ValueError(
                "Unexpected SHAP output dimensions."
            )

    expected_value = np.asarray(
        explainer.expected_value
    )

    if expected_value.ndim > 0:

        expected_value = expected_value.reshape(-1)

        if len(expected_value) > 1:
            expected_value = expected_value[1]
        else:
            expected_value = expected_value[0]

    expected_value = float(expected_value)

    if len(values) != feature_frame.shape[1]:
        raise ValueError(
            "SHAP contribution count does not match model features."
        )

    return values.astype(float), expected_value


# ============================================================
# EXPLAIN INDIVIDUAL TRANSACTION
# ============================================================

def explain_transaction(feature_values_json):

    # 1. Validate and prepare the exact stored model input.
    feature_frame = prepare_feature_frame(
        feature_values_json
    )

    model = get_model()
    explainer = get_explainer()

    # 2. Calculate the model's fraud probability.
    fraud_probability = float(
        model.predict_proba(feature_frame)[0, 1]
    )

    # 3. Calculate SHAP contributions.
    shap_values, base_value = get_fraud_shap_values(
        explainer,
        feature_frame
    )

    # 4. Create a waterfall-compatible explanation.
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_frame.iloc[0].to_numpy(dtype=float),
        feature_names=list(feature_frame.columns)
    )

    # 5. Rank features by their contribution.
    contributions = pd.DataFrame({

        "Feature": feature_frame.columns,

        "Feature Value": (
            feature_frame.iloc[0].to_numpy(dtype=float)
        ),

        "SHAP Value": shap_values,

    })

    contributions["Impact"] = np.where(
        contributions["SHAP Value"] > 0,
        "Increases Fraud Risk",
        np.where(
            contributions["SHAP Value"] < 0,
            "Decreases Fraud Risk",
            "No Impact"
        )
    )

    contributions["Absolute Impact"] = (
        contributions["SHAP Value"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute Impact",
        ascending=False
    ).reset_index(drop=True)

    # 6. Return all explanation results.
    return {

        "fraud_probability": fraud_probability,

        "base_value": base_value,

        "explanation": explanation,

        "contributions": contributions,

        "feature_count": len(shap_values),

        "shap_contribution_sum": float(
            np.sum(shap_values)
        ),

        "reconstructed_probability": float(
            base_value + np.sum(shap_values)
        ),

    }
