
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from contextlib import contextmanager

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    BASE_DIR / "data" / "fraud_monitoring.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

@contextmanager
def get_connection():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                transaction_time REAL NOT NULL,

                amount REAL NOT NULL,

                fraud_probability REAL NOT NULL,

                prediction INTEGER NOT NULL,

                result TEXT NOT NULL,

                risk_level TEXT NOT NULL,

                model_name TEXT NOT NULL,

                created_at TEXT NOT NULL,

                response_time_ms REAL,

                source TEXT DEFAULT 'API'
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_transactions_prediction
            ON transactions(prediction)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_transactions_created_at
            ON transactions(created_at)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_transactions_risk
            ON transactions(risk_level)
        """)

        connection.commit()


# ============================================================
# SAVE TRANSACTION
# ============================================================

def save_transaction(
    transaction_time,
    amount,
    fraud_probability,
    prediction,
    result,
    risk_level,
    model_name="Random Forest",
    response_time_ms=None,
    source="API",
):

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO transactions (

                transaction_time,
                amount,
                fraud_probability,
                prediction,
                result,
                risk_level,
                model_name,
                created_at,
                response_time_ms,
                source

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                transaction_time,
                amount,
                fraud_probability,
                prediction,
                result,
                risk_level,
                model_name,
                created_at,
                response_time_ms,
                source,
            )
        )

        connection.commit()

        return cursor.lastrowid


# ============================================================
# FETCH TRANSACTION HISTORY
# ============================================================

def get_transactions(limit=1000):

    with get_connection() as connection:

        dataframe = pd.read_sql_query(
            """
            SELECT *
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,)
        )

    return dataframe


# ============================================================
# FETCH DASHBOARD METRICS
# ============================================================

def get_dashboard_metrics():

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT

                COUNT(*) AS total_transactions,

                COALESCE(SUM(prediction = 1), 0) AS detected_frauds,

                COALESCE(SUM(prediction = 0), 0) AS legitimate_transactions,

                COALESCE(
                    SUM(CASE WHEN prediction = 1 THEN amount ELSE 0 END),
                    0
                ) AS flagged_amount,

                COALESCE(
                    AVG(fraud_probability),
                    0
                ) AS average_fraud_probability,

                COALESCE(
                    AVG(response_time_ms),
                    0
                ) AS average_response_time_ms

            FROM transactions
            """
        ).fetchone()

    return dict(row)


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_database():

    with get_connection() as connection:

        result = connection.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()

    return {
        "database": str(DATABASE_PATH),
        "status": "healthy",
        "transaction_count": result[0],
    }
