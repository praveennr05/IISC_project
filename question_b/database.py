"""
Question B - Database Layer
Pure Hand-Written SQL using Python's built-in sqlite3.
No ORM (Object-Relational Mapping) used, strictly adhering to assignment rules.
"""

import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import DB_PATH


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates a connection to SQLite database with WAL mode and row factory."""
    target_path = str(db_path or DB_PATH)
    conn = sqlite3.connect(target_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    # Enable Write-Ahead Logging (WAL) for concurrent read/write throughput
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initializes the SQLite schema using raw DDL.
    Creates table 'predictions' to persist every request and inference outcome.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Hand-written DDL table creation
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        age INTEGER NOT NULL,
        sex INTEGER NOT NULL,
        cp INTEGER NOT NULL,
        trestbps INTEGER NOT NULL,
        chol INTEGER NOT NULL,
        fbs INTEGER NOT NULL,
        restecg INTEGER NOT NULL,
        thalach INTEGER NOT NULL,
        exang INTEGER NOT NULL,
        oldpeak REAL NOT NULL,
        slope INTEGER NOT NULL,
        ca INTEGER NOT NULL,
        thal INTEGER NOT NULL,
        risk_score REAL NOT NULL,
        prediction INTEGER NOT NULL,
        risk_category TEXT NOT NULL,
        user_agent TEXT
    );
    """
    cursor.execute(create_table_sql)

    # Index for fast statistical aggregations and timestamps
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions (risk_score);")

    conn.commit()
    conn.close()


def save_prediction(
    input_data: Dict[str, Any],
    risk_score: float,
    prediction: int,
    risk_category: str,
    user_agent: str = "Web Client",
    db_path: Optional[Path] = None
) -> int:
    """
    Inserts a single prediction record using hand-written parameterized SQL.
    Prevents SQL injection vulnerabilities.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    insert_sql = """
    INSERT INTO predictions (
        timestamp, age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal,
        risk_score, prediction, risk_category, user_agent
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?
    );
    """

    cursor.execute(insert_sql, (
        now_iso,
        int(input_data["age"]),
        int(input_data["sex"]),
        int(input_data["cp"]),
        int(input_data["trestbps"]),
        int(input_data["chol"]),
        int(input_data["fbs"]),
        int(input_data["restecg"]),
        int(input_data["thalach"]),
        int(input_data["exang"]),
        float(input_data["oldpeak"]),
        int(input_data["slope"]),
        int(input_data["ca"]),
        int(input_data["thal"]),
        float(risk_score),
        int(prediction),
        str(risk_category),
        str(user_agent)
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Executes hand-written SQL (no ORM) to return aggregate health metrics:
    1. Total requests
    2. Average predicted risk
    3. Share of high-risk results (risk_score >= 0.50)
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Hand-written SQL aggregation with defensive COALESCE and zero-division guards
    stats_sql = """
    SELECT 
        COUNT(*) AS total_requests,
        COALESCE(AVG(risk_score), 0.0) AS avg_predicted_risk,
        COALESCE(SUM(CASE WHEN risk_score >= 0.50 THEN 1 ELSE 0 END), 0) AS high_risk_count,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                CAST(SUM(CASE WHEN risk_score >= 0.50 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
            ELSE 0.0 
        END AS share_of_high_risk_results
    FROM predictions;
    """

    cursor.execute(stats_sql)
    row = cursor.fetchone()

    total_requests = int(row["total_requests"])
    avg_risk = float(row["avg_predicted_risk"])
    high_risk_count = int(row["high_risk_count"])
    share_high_risk = float(row["share_of_high_risk_results"])

    conn.close()

    return {
        "total_requests": total_requests,
        "average_predicted_risk": round(avg_risk, 4),
        "average_predicted_risk_percent": f"{round(avg_risk * 100, 2)}%",
        "high_risk_count": high_risk_count,
        "low_risk_count": total_requests - high_risk_count,
        "share_of_high_risk_results": round(share_high_risk, 4),
        "share_of_high_risk_percent": f"{round(share_high_risk * 100, 2)}%"
    }


def get_recent_predictions(limit: int = 5, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns the most recent predictions using raw SQL."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    recent_sql = """
    SELECT id, timestamp, age, sex, risk_score, prediction, risk_category
    FROM predictions
    ORDER BY id DESC
    LIMIT ?;
    """

    cursor.execute(recent_sql, (limit,))
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully at: {DB_PATH}")
    stats = get_stats()
    print("Initial Database Stats (Hand-written SQL):", stats)
