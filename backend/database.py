"""
database.py
-----------
SQLite database setup and helper functions for AutoGuard AI.

Uses Python's built-in sqlite3 module — no extra dependencies.
The database file is created automatically at first startup.

Table: diagnostics
  - id          INTEGER PRIMARY KEY AUTOINCREMENT
  - filename    TEXT    — uploaded audio filename
  - prediction  TEXT    — engine_knock | healthy | misfire
  - confidence  REAL    — top class probability (0.0–1.0)
  - knock_score REAL    — engine_knock probability
  - healthy_score REAL  — healthy probability
  - misfire_score REAL  — misfire probability
  - duration_sec  REAL  — audio duration in seconds
  - inference_ms  REAL  — model inference time in milliseconds
  - created_at  TEXT    — ISO-8601 timestamp
"""

import sqlite3
import os
from datetime import datetime, timezone

# Database file lives alongside the backend code
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoguard.db")


def get_connection() -> sqlite3.Connection:
    """Open and return a database connection with row factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")   # safer concurrent writes
    return conn


def init_db() -> None:
    """Create the diagnostics table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT    NOT NULL,
                prediction      TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                knock_score     REAL    NOT NULL,
                healthy_score   REAL    NOT NULL,
                misfire_score   REAL    NOT NULL,
                duration_sec    REAL    NOT NULL,
                inference_ms    REAL    NOT NULL,
                created_at      TEXT    NOT NULL
            )
        """)
        conn.commit()
    print(f"[DB] Database ready at: {DB_PATH}")


def insert_diagnostic(
    filename: str,
    prediction: str,
    confidence: float,
    class_scores: dict,
    duration_sec: float,
    inference_ms: float,
) -> int:
    """
    Insert a new diagnostic record and return its auto-assigned ID.

    Parameters
    ----------
    filename    : Original uploaded filename
    prediction  : Predicted class label
    confidence  : Confidence of the top prediction (0.0–1.0)
    class_scores: Dict mapping class name → probability
    duration_sec: Audio duration in seconds
    inference_ms: Inference time in milliseconds

    Returns
    -------
    int : The new row's ID
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO diagnostics
                (filename, prediction, confidence, knock_score, healthy_score,
                 misfire_score, duration_sec, inference_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                prediction,
                round(confidence, 4),
                round(class_scores.get("engine_knock", 0.0), 4),
                round(class_scores.get("healthy", 0.0), 4),
                round(class_scores.get("misfire", 0.0), 4),
                round(duration_sec, 2),
                round(inference_ms, 1),
                created_at,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_all_diagnostics(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return diagnostic records ordered newest-first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM diagnostics
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def get_diagnostic_by_id(scan_id: int) -> dict | None:
    """Return a single diagnostic record by ID, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM diagnostics WHERE id = ?", (scan_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_diagnostic(scan_id: int) -> bool:
    """Delete a diagnostic record by ID. Returns True if a row was deleted."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM diagnostics WHERE id = ?", (scan_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_stats() -> dict:
    """Return aggregate statistics across all diagnostic records."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                            AS total_scans,
                COUNT(*)                                            AS completed_scans,
                SUM(CASE WHEN prediction != 'healthy' THEN 1 ELSE 0 END) AS detected_faults,
                SUM(CASE WHEN prediction = 'healthy'  THEN 1 ELSE 0 END) AS healthy_results,
                SUM(CASE WHEN prediction = 'engine_knock' THEN 1 ELSE 0 END) AS knock_results,
                SUM(CASE WHEN prediction = 'misfire' THEN 1 ELSE 0 END)  AS misfire_results
            FROM diagnostics
        """).fetchone()
    return dict(row) if row else {
        "total_scans": 0,
        "completed_scans": 0,
        "detected_faults": 0,
        "healthy_results": 0,
        "knock_results": 0,
        "misfire_results": 0,
    }
