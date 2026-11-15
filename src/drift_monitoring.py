from src.database import get_live_feature_data
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "train_features.csv"
)

FEATURE_SCHEMA_PATH = (
    BASE_DIR / "reports" / "feature_schema.json"
)

EPSILON = 1e-6
MIN_SAMPLE_SIZE = 100


def load_model_features():
    """Load the exact feature names used by the trained model."""

    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)

    # The preprocessing schema contains two additional intermediate
    # columns. The actual model uses 37 features.
    import joblib

    model = joblib.load(BASE_DIR / "models" / "random_forest.joblib")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The trained model does not expose feature_names_in_."
        )

    model_features = list(model.feature_names_in_)

    if len(model_features) != 37:
        raise ValueError(
            f"Expected 37 model features, found {len(model_features)}."
        )

    missing = set(model_features) - set(schema["feature_columns"])

    if missing:
        raise ValueError(
            f"Model features missing from feature schema: {sorted(missing)}"
        )

    return model_features


def load_reference_data():
    """Load training data using only the actual model input columns."""

    features = load_model_features()

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training reference dataset not found: {TRAIN_DATA_PATH}"
        )

    reference = pd.read_csv(
        TRAIN_DATA_PATH,
        usecols=features,
    )

    reference = reference.replace([np.inf, -np.inf], np.nan)

    return reference


def _clean_numeric_series(series):
    """Convert input data into finite numeric values."""

    values = pd.Series(series)
    values = pd.to_numeric(values, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    return values.dropna().to_numpy(dtype=float)

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index using reference quantile bins.

    Returns a PSI score and diagnostic information.
    """

    reference_values = _clean_numeric_series(reference)
    current_values = _clean_numeric_series(current)

    if len(reference_values) < MIN_SAMPLE_SIZE:
        raise ValueError(
            "Reference sample must contain at least 100 valid observations."
        )

    if len(current_values) < MIN_SAMPLE_SIZE:
        raise ValueError(
            "Current sample must contain at least 100 valid observations."
        )

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(reference_values, quantiles)
    )

    # A constant reference feature has no meaningful quantile intervals.
    if len(edges) < 2:
        return {
            "psi": 0.0 if np.all(current_values == reference_values[0]) else None,
            "reference_count": len(reference_values),
            "current_count": len(current_values),
            "valid": False,
            "reason": "Reference feature is constant; PSI is not meaningful.",
        }

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    reference_proportions = (
        reference_counts / len(reference_values)
    )

    current_proportions = (
        current_counts / len(current_values)
    )

    reference_proportions = np.clip(
        reference_proportions,
        EPSILON,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        EPSILON,
        None,
    )

    psi_components = (
        current_proportions - reference_proportions
    ) * np.log(
        current_proportions / reference_proportions
    )

    psi_score = float(np.sum(psi_components))

    return {
        "psi": psi_score,
        "reference_count": len(reference_values),
        "current_count": len(current_values),
        "valid": True,
        "reason": None,
    }


def classify_drift(psi):
    """Classify drift using commonly used PSI thresholds."""

    if psi is None or not np.isfinite(psi):
        return "UNAVAILABLE"

    if psi < 0.10:
        return "STABLE"

    if psi < 0.25:
        return "MODERATE"

    return "SIGNIFICANT"


def calculate_feature_drift(reference_df, current_df):
    """Calculate drift for all model features."""

    features = load_model_features()

    missing_reference = set(features) - set(reference_df.columns)
    missing_current = set(features) - set(current_df.columns)

    if missing_reference:
        raise ValueError(
            f"Missing reference features: {sorted(missing_reference)}"
        )

    if missing_current:
        raise ValueError(
            f"Missing current features: {sorted(missing_current)}"
        )

    results = []

    for feature in features:

        reference_values = reference_df[feature]
        current_values = current_df[feature]

        reference_valid = _clean_numeric_series(reference_values)
        current_valid = _clean_numeric_series(current_values)

        if (
            len(reference_valid) < MIN_SAMPLE_SIZE
            or len(current_valid) < MIN_SAMPLE_SIZE
        ):
            results.append({
                "feature": feature,
                "psi": None,
                "severity": "INSUFFICIENT_DATA",
                "reference_count": len(reference_valid),
                "current_count": len(current_valid),
                "missing_reference": int(reference_values.isna().sum()),
                "missing_current": int(current_values.isna().sum()),
            })
            continue

        result = calculate_psi(
            reference_values,
            current_values,
        )

        psi = result["psi"]

        results.append({
            "feature": feature,
            "psi": psi,
            "severity": classify_drift(psi),
            "reference_count": result["reference_count"],
            "current_count": result["current_count"],
            "missing_reference": int(reference_values.isna().sum()),
            "missing_current": int(current_values.isna().sum()),
        })

    return pd.DataFrame(results).sort_values(
        "psi",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def calculate_prediction_drift(reference_probabilities, current_probabilities):
    """Calculate PSI for fraud prediction probabilities."""

    result = calculate_psi(
        reference_probabilities,
        current_probabilities,
    )

    return {
        **result,
        "severity": classify_drift(result["psi"]),
    }


def calculate_overall_drift(feature_results):
    """Summarize drift severity across all monitored features."""

    if feature_results.empty:
        return {
            "severity": "INSUFFICIENT_DATA",
            "features_monitored": 0,
            "stable_features": 0,
            "moderate_features": 0,
            "significant_features": 0,
        }

    severity_counts = feature_results["severity"].value_counts()

    stable = int(severity_counts.get("STABLE", 0))
    moderate = int(severity_counts.get("MODERATE", 0))
    significant = int(severity_counts.get("SIGNIFICANT", 0))

    if significant > 0:
        overall_severity = "SIGNIFICANT"
    elif moderate > 0:
        overall_severity = "MODERATE"
    elif stable > 0:
        overall_severity = "STABLE"
    else:
        overall_severity = "INSUFFICIENT_DATA"

    return {
        "severity": overall_severity,
        "features_monitored": len(feature_results),
        "stable_features": stable,
        "moderate_features": moderate,
        "significant_features": significant,
    }

def calculate_live_drift(min_sample_size=MIN_SAMPLE_SIZE):
    """
    Assess live API feature drift against the training reference.

    Returns insufficient-data status instead of calculating unreliable
    PSI scores when too few genuine API feature records are available.
    """

    if min_sample_size < 1:
        raise ValueError("min_sample_size must be at least 1.")

    live_data = get_live_feature_data(limit=5000)

    features = load_model_features()

    if live_data.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "No genuine API feature records are available.",
            "live_records": 0,
            "minimum_required": min_sample_size,
            "results": pd.DataFrame(),
            "summary": calculate_overall_drift(pd.DataFrame()),
        }

    missing_features = [
        feature
        for feature in features
        if feature not in live_data.columns
    ]

    if missing_features:
        return {
            "status": "INVALID_SCHEMA",
            "message": (
                "Live records are missing required model features: "
                + ", ".join(missing_features)
            ),
            "live_records": len(live_data),
            "minimum_required": min_sample_size,
            "results": pd.DataFrame(),
            "summary": calculate_overall_drift(pd.DataFrame()),
        }

    live_features = live_data[features].copy()

    valid_rows = (
        live_features
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )

    live_features = live_features.loc[valid_rows]

    if len(live_features) < min_sample_size:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": (
                f"Only {len(live_features)} valid live records are "
                f"available. At least {min_sample_size} are required."
            ),
            "live_records": len(live_features),
            "minimum_required": min_sample_size,
            "results": pd.DataFrame(),
            "summary": calculate_overall_drift(pd.DataFrame()),
        }
    reference = load_reference_data()

    results = calculate_feature_drift(
        reference,
        live_features,
    )

    summary = calculate_overall_drift(results)

    return {
        "status": "AVAILABLE",
        "message": "Live feature drift calculated successfully.",
        "live_records": len(live_features),
        "minimum_required": min_sample_size,
        "results": results,
        "summary": summary,
    }
