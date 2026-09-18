import sqlite3
from pathlib import Path
from typing import List, Optional

from src.config import DATABASE_PATH
from src.transcript_parser import Expert, TranscriptSegment

SCHEMA = """
CREATE TABLE IF NOT EXISTS experts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    market TEXT NOT NULL,
    source_file TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    expert_id INTEGER NOT NULL REFERENCES experts(id),
    source_file TEXT NOT NULL UNIQUE,
    segment_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    expert_name TEXT NOT NULL,
    role TEXT NOT NULL,
    market TEXT NOT NULL,
    source_file TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_type TEXT NOT NULL,   -- 'interview_guide' or 'cross_call'
    expert_name TEXT,              -- NULL for cross-call results
    question_key TEXT NOT NULL,    -- question text or theme identifier
    result_json TEXT NOT NULL,
    UNIQUE(analysis_type, expert_name, question_key)
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not already exist. Safe to call every run."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def is_database_populated() -> bool:
    """Return True if transcript segments have already been ingested."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS count FROM transcript_segments").fetchone()
        return bool(row and row["count"] > 0)
    except sqlite3.OperationalError:
        # Tables do not exist yet.
        return False
    finally:
        conn.close()


def insert_expert_and_transcript(expert: Expert, segments: List[TranscriptSegment]) -> int:
    """Insert an expert, its transcript record and all segments. Returns expert_id."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT OR REPLACE INTO experts (name, role, market, source_file) "
            "VALUES (?, ?, ?, ?)",
            (expert.name, expert.role, expert.market, expert.source_file),
        )
        expert_id = cursor.lastrowid

        cursor = conn.execute(
            "INSERT OR REPLACE INTO transcripts (expert_id, source_file, segment_count) "
            "VALUES (?, ?, ?)",
            (expert_id, expert.source_file, len(segments)),
        )
        transcript_id = cursor.lastrowid

        # Avoid duplicate segments if this transcript was ingested before.
        conn.execute(
            "DELETE FROM transcript_segments WHERE transcript_id = ?", (transcript_id,)
        )

        conn.executemany(
            """
            INSERT INTO transcript_segments
                (transcript_id, expert_name, role, market, source_file,
                 segment_index, timestamp, speaker, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    transcript_id,
                    s.expert_name,
                    s.role,
                    s.market,
                    s.source_file,
                    s.segment_index,
                    s.timestamp,
                    s.speaker,
                    s.text,
                )
                for s in segments
            ],
        )
        conn.commit()
        return expert_id
    finally:
        conn.close()


def get_all_experts() -> List[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute("SELECT * FROM experts ORDER BY id").fetchall()
    finally:
        conn.close()


def get_segments_for_expert(expert_name: str) -> List[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM transcript_segments WHERE expert_name = ? "
            "ORDER BY segment_index",
            (expert_name,),
        ).fetchall()
    finally:
        conn.close()


def get_all_segments() -> List[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM transcript_segments ORDER BY source_file, segment_index"
        ).fetchall()
    finally:
        conn.close()


def get_segment_texts_for_source(source_file: str) -> List[str]:
    """All raw segment texts for a given transcript file (for quote validation)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT text FROM transcript_segments WHERE source_file = ?",
            (source_file,),
        ).fetchall()
        return [r["text"] for r in rows]
    finally:
        conn.close()


def cache_analysis_result(
    analysis_type: str, question_key: str, result_json: str, expert_name: Optional[str] = None
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO analysis_results
                (analysis_type, expert_name, question_key, result_json)
            VALUES (?, ?, ?, ?)
            """,
            (analysis_type, expert_name, question_key, result_json),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_analysis_result(
    analysis_type: str, question_key: str, expert_name: Optional[str] = None
) -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT result_json FROM analysis_results
            WHERE analysis_type = ? AND question_key = ?
              AND ((expert_name IS NULL AND ? IS NULL) OR expert_name = ?)
            """,
            (analysis_type, question_key, expert_name, expert_name),
        ).fetchone()
        return row["result_json"] if row else None
    finally:
        conn.close()
