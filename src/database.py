import sqlite3

import json

from pathlib import Path

from datetime import datetime, timezone

from contextlib import contextmanager

import os
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = Path(
    os.getenv(
        "FRAUD_DATABASE_PATH",
        str(BASE_DIR / "data" / "fraud_monitoring.db"),
    )
)

ALERT_STATUSES = {

    "OPEN",

    "UNDER_REVIEW",

    "ESCALATED",

    "RESOLVED",

    "FALSE_POSITIVE",

}

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

    # Enforce foreign-key relationships.

    connection.execute("PRAGMA foreign_keys = ON")

    try:

        yield connection

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()

def _utc_now():

    return datetime.now(timezone.utc).isoformat()

# ============================================================

# DATABASE INITIALIZATION

# ============================================================

def initialize_database():

    with get_connection() as connection:

        # Original transaction table.

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

                model_version TEXT,

                created_at TEXT NOT NULL,

                response_time_ms REAL,

                source TEXT DEFAULT 'API',

                feature_values_json TEXT

            )

        """)

        # Backward-compatible migration.

        columns = {

            row["name"]

            for row in connection.execute(

                "PRAGMA table_info(transactions)"

            ).fetchall()

        }

        if "feature_values_json" not in columns:

            connection.execute("""

                ALTER TABLE transactions

                ADD COLUMN feature_values_json TEXT

            """)

        if "model_version" not in columns:

            connection.execute("""
                ALTER TABLE transactions
                ADD COLUMN model_version TEXT
            """)

        # Original transaction indexes.

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

        # ====================================================

        # FRAUD ALERTS

        # ====================================================

        connection.execute("""

            CREATE TABLE IF NOT EXISTS fraud_alerts (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                transaction_id INTEGER NOT NULL UNIQUE,

                status TEXT NOT NULL DEFAULT 'OPEN'

                    CHECK (

                        status IN (

                            'OPEN',

                            'UNDER_REVIEW',

                            'ESCALATED',

                            'RESOLVED',

                            'FALSE_POSITIVE'

                        )

                    ),

                priority TEXT NOT NULL,

                assigned_to TEXT,

                resolution TEXT,

                created_at TEXT NOT NULL,

                updated_at TEXT NOT NULL,

                FOREIGN KEY (transaction_id)

                    REFERENCES transactions(id)

                    ON DELETE CASCADE

            )

        """)

        # ====================================================

        # ALERT AUDIT HISTORY

        # ====================================================

        connection.execute("""

            CREATE TABLE IF NOT EXISTS alert_events (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                alert_id INTEGER NOT NULL,

                event_type TEXT NOT NULL,

                previous_status TEXT,

                new_status TEXT,

                actor TEXT NOT NULL DEFAULT 'SYSTEM',

                note TEXT,

                created_at TEXT NOT NULL,

                FOREIGN KEY (alert_id)

                    REFERENCES fraud_alerts(id)

                    ON DELETE CASCADE

            )

        """)

        # Alert indexes.

        connection.execute("""

            CREATE INDEX IF NOT EXISTS

            idx_alerts_status

            ON fraud_alerts(status)

        """)

        connection.execute("""

            CREATE INDEX IF NOT EXISTS

            idx_alerts_priority

            ON fraud_alerts(priority)

        """)

        connection.execute("""

            CREATE INDEX IF NOT EXISTS

            idx_alert_events_alert_id

            ON alert_events(alert_id)

        """)

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

    feature_values=None,

    model_version=None,

):

    created_at = _utc_now()

    feature_values_json = (

        json.dumps(

            feature_values,

            default=lambda value: (

                value.item()

                if hasattr(value, "item")

                else str(value)

            )

        )

        if feature_values is not None

        else None

    )

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

                model_version,

                created_at,

                response_time_ms,

                source,

                feature_values_json

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """,

            (

                transaction_time,

                amount,

                fraud_probability,

                prediction,

                result,

                risk_level,

                model_name,

                model_version,

                created_at,

                response_time_ms,

                source,

                feature_values_json,

            )

        )

        transaction_id = cursor.lastrowid

        # Automatically create an alert for fraud predictions.

        if int(prediction) == 1:

            alert_cursor = connection.execute(

                """

                INSERT OR IGNORE INTO fraud_alerts (

                    transaction_id,

                    status,

                    priority,

                    created_at,

                    updated_at

                )

                VALUES (?, 'OPEN', ?, ?, ?)

                """,

                (

                    transaction_id,

                    str(risk_level).upper(),

                    created_at,

                    created_at,

                )

            )

            # Record initial alert creation in the audit history.

            if alert_cursor.rowcount > 0:

                alert_id = alert_cursor.lastrowid

                connection.execute(

                    """

                    INSERT INTO alert_events (

                        alert_id,

                        event_type,

                        new_status,

                        actor,

                        note,

                        created_at

                    )

                    VALUES (?, ?, ?, ?, ?, ?)

                    """,

                    (

                        alert_id,

                        "ALERT_CREATED",

                        "OPEN",

                        "SYSTEM",

                        "Automatically created from a fraud prediction.",

                        created_at,

                    )

                )

        return transaction_id

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

                COALESCE(

                    SUM(prediction = 1), 0

                ) AS detected_frauds,

                COALESCE(

                    SUM(prediction = 0), 0

                ) AS legitimate_transactions,

                COALESCE(

                    SUM(

                        CASE

                            WHEN prediction = 1 THEN amount

                            ELSE 0

                        END

                    ),

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

# FETCH FRAUD ALERTS

# ============================================================

def get_alerts(

    status=None,

    priority=None,

    limit=1000

):

    query = """

        SELECT

            a.id AS alert_id,

            a.transaction_id,

            a.status,

            a.priority,

            a.assigned_to,

            a.resolution,

            a.created_at AS alert_created_at,

            a.updated_at AS alert_updated_at,

            t.amount,

            t.fraud_probability,

            t.risk_level,

            t.model_name,

            t.model_version,

            t.created_at AS transaction_created_at,

            t.source

        FROM fraud_alerts a

        JOIN transactions t

            ON a.transaction_id = t.id

        WHERE 1 = 1

    """

    params = []

    if status:

        if status not in ALERT_STATUSES:

            raise ValueError("Invalid alert status.")

        query += " AND a.status = ?"

        params.append(status)

    if priority:

        query += " AND a.priority = ?"

        params.append(priority.upper())

    query += " ORDER BY a.created_at DESC LIMIT ?"

    params.append(limit)

    with get_connection() as connection:

        dataframe = pd.read_sql_query(

            query,

            connection,

            params=params

        )

    return dataframe

# ============================================================

# UPDATE ALERT STATUS

# ============================================================

def update_alert_status(

    alert_id,

    new_status,

    actor="INVESTIGATOR",

    resolution=None

):

    new_status = str(new_status).upper()

    if new_status not in ALERT_STATUSES:

        raise ValueError("Invalid alert status.")

    if (

        new_status in {"RESOLVED", "FALSE_POSITIVE"}

        and not resolution

    ):

        raise ValueError(

            "A resolution reason is required for this status."

        )

    now = _utc_now()

    with get_connection() as connection:

        alert = connection.execute(

            """

            SELECT status

            FROM fraud_alerts

            WHERE id = ?

            """,

            (alert_id,)

        ).fetchone()

        if alert is None:

            raise ValueError("Alert not found.")

        previous_status = alert["status"]

        connection.execute(

            """

            UPDATE fraud_alerts

            SET

                status = ?,

                resolution = ?,

                updated_at = ?

            WHERE id = ?

            """,

            (

                new_status,

                resolution,

                now,

                alert_id,

            )

        )

        connection.execute(

            """

            INSERT INTO alert_events (

                alert_id,

                event_type,

                previous_status,

                new_status,

                actor,

                note,

                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

            """,

            (

                alert_id,

                "STATUS_CHANGED",

                previous_status,

                new_status,

                actor,

                resolution,

                now,

            )

        )

    return True

# ============================================================

# ASSIGN ALERT TO INVESTIGATOR

# ============================================================

def assign_alert(

    alert_id,

    assigned_to,

    actor="INVESTIGATOR"

):

    if not assigned_to or not assigned_to.strip():

        raise ValueError("Investigator name cannot be empty.")

    now = _utc_now()

    with get_connection() as connection:

        alert = connection.execute(

            """

            SELECT id, assigned_to

            FROM fraud_alerts

            WHERE id = ?

            """,

            (alert_id,)

        ).fetchone()

        if alert is None:

            raise ValueError("Alert not found.")

        previous_assignee = alert["assigned_to"]

        connection.execute(

            """

            UPDATE fraud_alerts

            SET assigned_to = ?, updated_at = ?

            WHERE id = ?

            """,

            (

                assigned_to.strip(),

                now,

                alert_id,

            )

        )

        connection.execute(

            """

            INSERT INTO alert_events (

                alert_id,

                event_type,

                actor,

                note,

                created_at

            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                alert_id,

                "ALERT_ASSIGNED",

                actor,

                f"Assigned to {assigned_to.strip()}. Previous assignee: "

                f"{previous_assignee or 'Unassigned'}.",

                now,

            )

        )

    return True

# ============================================================

# ADD INVESTIGATOR NOTE

# ============================================================

def add_alert_note(

    alert_id,

    note,

    actor="INVESTIGATOR"

):

    if not note or not note.strip():

        raise ValueError("Note cannot be empty.")

    now = _utc_now()

    with get_connection() as connection:

        alert = connection.execute(

            """

            SELECT id

            FROM fraud_alerts

            WHERE id = ?

            """,

            (alert_id,)

        ).fetchone()

        if alert is None:

            raise ValueError("Alert not found.")

        connection.execute(

            """

            INSERT INTO alert_events (

                alert_id,

                event_type,

                actor,

                note,

                created_at

            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                alert_id,

                "INVESTIGATOR_NOTE",

                actor,

                note.strip(),

                now,

            )

        )

        connection.execute(

            """

            UPDATE fraud_alerts

            SET updated_at = ?

            WHERE id = ?

            """,

            (now, alert_id)

        )

    return True

# ============================================================

# FETCH ALERT AUDIT HISTORY

# ============================================================

def get_alert_history(alert_id):

    with get_connection() as connection:

        dataframe = pd.read_sql_query(

            """

            SELECT *

            FROM alert_events

            WHERE alert_id = ?

            ORDER BY id ASC

            """,

            connection,

            params=(alert_id,)

        )

    return dataframe

# ============================================================

# ALERT METRICS

# ============================================================

def get_alert_metrics():

    with get_connection() as connection:

        row = connection.execute(

            """

            SELECT

                COUNT(*) AS total_alerts,

                COALESCE(

                    SUM(status = 'OPEN'), 0

                ) AS open_alerts,

                COALESCE(

                    SUM(status = 'UNDER_REVIEW'), 0

                ) AS under_review_alerts,

                COALESCE(

                    SUM(status = 'ESCALATED'), 0

                ) AS escalated_alerts,

                COALESCE(

                    SUM(status = 'RESOLVED'), 0

                ) AS resolved_alerts,

                COALESCE(

                    SUM(status = 'FALSE_POSITIVE'), 0

                ) AS false_positive_alerts

            FROM fraud_alerts

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

        alert_result = connection.execute(

            "SELECT COUNT(*) FROM fraud_alerts"

        ).fetchone()

    return {

        "database": str(DATABASE_PATH),

        "status": "healthy",

        "transaction_count": result[0],

        "alert_count": alert_result[0],

    }

def get_live_feature_data(limit=5000):

    """

    Retrieve genuine API transaction features for live drift monitoring.

    Synthetic TEST records and records without stored features are excluded.

    """

    import json

    import pandas as pd

    if limit < 1:

        raise ValueError("limit must be greater than zero.")

    with get_connection() as connection:

        rows = connection.execute(

            """

            SELECT

                id,

                feature_values_json,

                created_at

            FROM transactions

            WHERE source = 'API'

              AND feature_values_json IS NOT NULL

              AND feature_values_json != ''

            ORDER BY id DESC

            LIMIT ?

            """,

            (int(limit),),

        ).fetchall()

    records = []

    for transaction_id, feature_json, created_at in rows:

        try:

            features = json.loads(feature_json)

            if not isinstance(features, dict):

                continue

            records.append({

                "transaction_id": transaction_id,

                "created_at": created_at,

                **features,

            })

        except (json.JSONDecodeError, TypeError):

            continue

    return pd.DataFrame(records)
