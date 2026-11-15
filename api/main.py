from pathlib import Path

from collections import deque

from threading import Lock

import json

import time

import logging

from logging.handlers import RotatingFileHandler

import joblib

import numpy as np

import pandas as pd

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

from src.model_registry import load_active_model_registry

from src.database import initialize_database, save_transaction

# ============================================================

# PROJECT PATHS AND CONFIGURATION

# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================

# STRUCTURED APPLICATION LOGGING

# ============================================================

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "fraud_detection.log"

LOG_FORMAT = (

    "%(asctime)s | %(levelname)s | "

    "%(name)s | %(message)s"

)

LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

file_handler = RotatingFileHandler(

    LOG_FILE,

    maxBytes=5 * 1024 * 1024,

    backupCount=5,

    encoding="utf-8",

)

file_handler.setFormatter(

    logging.Formatter(

        LOG_FORMAT,

        datefmt=LOG_DATE_FORMAT,

    )

)

console_handler = logging.StreamHandler()

console_handler.setFormatter(

    logging.Formatter(

        LOG_FORMAT,

        datefmt=LOG_DATE_FORMAT,

    )

)

logger = logging.getLogger("fraud_detection.api")

logger.setLevel(logging.INFO)

logger.propagate = False

if not logger.handlers:

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

SCALER_PATH = BASE_DIR / "models" / "amount_scaler.joblib"

SCHEMA_PATH = BASE_DIR / "reports" / "feature_schema.json"

# The dataset's Time column represents elapsed seconds.

# This API uses the same elapsed-time convention.

# ============================================================

# LOAD AND VERIFY ACTIVE MODEL REGISTRY

# ============================================================

(

    model_metadata,

    MODEL_PATH,

    actual_model_sha256,

    MODEL_REGISTRY_PATH,

) = load_active_model_registry(BASE_DIR)

# ============================================================

# LOAD VERIFIED MODEL AND SUPPORTING ARTIFACTS

# ============================================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:

    schema = json.load(file)

MODEL_FEATURES = [

    "Time",

    *[f"V{i}" for i in range(1, 29)],

    "hour_sin",

    "hour_cos",

    "elapsed_day",

    "txn_count_prev_60s",

    "txn_count_prev_300s",

    "txn_count_prev_3600s",

    "amount_robust",

    "log_amount_robust",

]

# Validate model feature names and order.

if list(model.feature_names_in_) != MODEL_FEATURES:

    raise RuntimeError(

        "Model feature schema does not match API schema."

    )

if len(schema["feature_columns"]) != 39:

    raise RuntimeError("Unexpected feature schema.")

if schema["target"] != "Class":

    raise RuntimeError("Unexpected target column.")

if model.n_features_in_ != len(MODEL_FEATURES):

    raise RuntimeError("Incorrect model input feature count.")

# ============================================================

# VALIDATE MODEL METADATA

# ============================================================

# Validate feature count.

if model_metadata.get("feature_count") != len(MODEL_FEATURES):

    raise RuntimeError(

        "Registered model feature count does not match "

        "the loaded model."

    )

# Validate decision threshold.

registered_threshold = model_metadata.get(

    "decision_threshold"

)

if (

    isinstance(registered_threshold, bool)

    or not isinstance(registered_threshold, (int, float))

    or not 0 <= registered_threshold <= 1

):

    raise RuntimeError(

        "Invalid decision threshold in model registry."

    )

# ============================================================

# ACTIVE MODEL IDENTITY

# ============================================================

MODEL_ID = model_metadata["model_id"]

MODEL_VERSION = model_metadata["model_version"]

MODEL_NAME = model_metadata["model_name"]

MODEL_SHA256 = actual_model_sha256

# Registry is the single source of truth.

FRAUD_THRESHOLD = float(registered_threshold)

logger.info(

    "Active model loaded | model_id=%s | version=%s | "

    "features=%s | threshold=%.6f",

    MODEL_ID,

    MODEL_VERSION,

    len(MODEL_FEATURES),

    FRAUD_THRESHOLD,

)

logger.info(

    "Model artifact integrity verified | sha256=%s",

    MODEL_SHA256,

)

# ============================================================

# FASTAPI APPLICATION

# ============================================================

app = FastAPI(

    title="Real-Time Financial Fraud Detection API",

    description=(

        "Fraud detection API using a trained Random Forest model, "

        "historical activity features and a validation-selected threshold."

    ),

    version="1.0.0",

)

# Initialize the SQLite database before accepting API requests.

initialize_database()

logger.info(
    "API initialized | model_id=%s | model_version=%s",
    MODEL_ID,
    MODEL_VERSION,
)

# ============================================================

# TRANSACTION HISTORY

# ============================================================

transaction_history = deque()

history_lock = Lock()

last_processed_time = None

cached_activity_features = None

# ============================================================

# HISTORICAL ACTIVITY FEATURES

# ============================================================

def calculate_activity_features(current_time):

    global last_processed_time

    global cached_activity_features

    with history_lock:

        # Reject out-of-order transactions.

        if (

            last_processed_time is not None

            and current_time < last_processed_time

        ):

            raise HTTPException(

                status_code=422,

                detail=(

                    "Transaction Time must be non-decreasing. "

                    "Out-of-order transactions are not supported."

                )

            )

        # Transactions sharing the same timestamp must receive

        # identical historical activity features.

        if current_time == last_processed_time:

            transaction_history.append(current_time)

            return cached_activity_features.copy()

        # Retain the exact 3600-second lower boundary.

        while (

            transaction_history

            and transaction_history[0] < current_time - 3600

        ):

            transaction_history.popleft()

        previous_times = list(transaction_history)

        counts = {}

        for seconds, feature_name in [

            (60, "txn_count_prev_60s"),

            (300, "txn_count_prev_300s"),

            (3600, "txn_count_prev_3600s"),

        ]:

            counts[feature_name] = sum(

                current_time - t <= seconds

                for t in previous_times

            )

        # Cache features before adding the current timestamp.

        cached_activity_features = counts.copy()

        transaction_history.append(current_time)

        last_processed_time = current_time

        return counts

# ============================================================

# REQUEST VALIDATION

# ============================================================

class TransactionRequest(BaseModel):

    Time: float = Field(

        ge=0,

        description="Elapsed seconds in the dataset time convention."

    )

    V1: float

    V2: float

    V3: float

    V4: float

    V5: float

    V6: float

    V7: float

    V8: float

    V9: float

    V10: float

    V11: float

    V12: float

    V13: float

    V14: float

    V15: float

    V16: float

    V17: float

    V18: float

    V19: float

    V20: float

    V21: float

    V22: float

    V23: float

    V24: float

    V25: float

    V26: float

    V27: float

    V28: float

    Amount: float = Field(

        ge=0,

        description="Transaction amount."

    )

# ============================================================

# FEATURE ENGINEERING

# ============================================================

def prepare_features(transaction):

    amount = transaction.Amount

    log_amount = np.log1p(amount)

    # Same cyclical time formulas as training.

    seconds_in_day = transaction.Time % 86400

    hour_sin = np.sin(

        2 * np.pi * seconds_in_day / 86400

    )

    hour_cos = np.cos(

        2 * np.pi * seconds_in_day / 86400

    )

    elapsed_day = int(

        np.floor(transaction.Time / 86400)

    )

    # Calculate historical transaction activity.

    activity = calculate_activity_features(

        transaction.Time

    )

    # Use the exact scaler fitted on training data.

    scaled_values = scaler.transform(

        pd.DataFrame(

            [[amount, log_amount]],

            columns=["Amount", "log_amount"]

        )

    )

    features = {

        "Time": transaction.Time,

        **{

            f"V{i}": getattr(transaction, f"V{i}")

            for i in range(1, 29)

        },

        "hour_sin": hour_sin,

        "hour_cos": hour_cos,

        "elapsed_day": elapsed_day,

        **activity,

        "amount_robust": scaled_values[0, 0],

        "log_amount_robust": scaled_values[0, 1],

    }

    feature_frame = pd.DataFrame(

        [features],

        columns=MODEL_FEATURES

    )

    return feature_frame

# ============================================================

# HEALTH CHECK

# ============================================================

@app.get("/")

def home():

    return {

        "message": "Financial Fraud Detection API is running.",

        "model": "Random Forest",

        "model_features": len(MODEL_FEATURES),

        "fraud_threshold": FRAUD_THRESHOLD,

        "status": "healthy",

    }

# ============================================================

# MODEL INFORMATION ENDPOINT

# ============================================================

@app.get("/model-info")

def get_model_info():

    return {

        "model_id": MODEL_ID,

        "model_version": MODEL_VERSION,

        "model_name": MODEL_NAME,

        "algorithm": model_metadata["algorithm"],

        "feature_count": len(MODEL_FEATURES),

        "fraud_threshold": FRAUD_THRESHOLD,

        "artifact_sha256": MODEL_SHA256,

        "status": model_metadata["status"],

        "evaluation": model_metadata["evaluation"],

    }

# ============================================================

# FRAUD PREDICTION ENDPOINT

# ============================================================

@app.post("/predict")

def predict_fraud(transaction: TransactionRequest):

    try:

        # Start measuring prediction execution time.

        start_time = time.perf_counter()

        features = prepare_features(transaction)

        fraud_probability = float(

            model.predict_proba(features)[0][1]

        )

        prediction = int(

            fraud_probability >= FRAUD_THRESHOLD

        )

        risk_level = (

            "HIGH"

            if prediction == 1

            else "LOW"

        )

        result = (

            "FRAUD"

            if prediction == 1

            else "LEGITIMATE"

        )

        response_time_ms = (

            time.perf_counter() - start_time

        ) * 1000

        # Save the successful prediction with model version metadata.

        transaction_id = save_transaction(

            transaction_time=transaction.Time,

            amount=transaction.Amount,

            fraud_probability=fraud_probability,

            prediction=prediction,

            result=result,

            risk_level=risk_level,

            model_name="Random Forest",

            response_time_ms=response_time_ms,

            source="API",

            feature_values=features.iloc[0].to_dict(),

            model_version=MODEL_VERSION,

        )

        logger.info(
            "Prediction completed | transaction_id=%s | "
            "result=%s | risk_level=%s | model_version=%s | "
            "response_time_ms=%.3f",
            transaction_id,
            result,
            risk_level,
            MODEL_VERSION,
            response_time_ms,
        )

        return {

            "transaction_id": transaction_id,

            "prediction": prediction,

            "result": result,

            "fraud_probability": round(

                fraud_probability, 6

            ),

            "fraud_threshold": FRAUD_THRESHOLD,

            "risk_level": risk_level,

            "model": "Random Forest",

            "response_time_ms": round(

                response_time_ms, 3

            ),

        }


    except HTTPException as error:

        logger.warning(
            "Prediction request rejected | status_code=%s",
            error.status_code,
        )
        raise

    except Exception:

        logger.exception(
            "Unexpected prediction failure | model_version=%s",
            MODEL_VERSION,
        )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the prediction.",
        )
