"""
Database layer — supports both SQLite (local/test) and PostgreSQL (production).

When DATABASE_URL env var is set: uses PostgreSQL via psycopg2.
Otherwise: uses SQLite with DB_PATH (monkeypatchable for tests).
"""

import os
import uuid
from datetime import datetime

# SQLite fallback path — used when DATABASE_URL env var is not set (local dev + tests)
DB_PATH = "memory.db"

# Read at import time for PH only — _connect() always re-reads at call time
# so Streamlit Cloud secrets injected after import still take effect.
DATABASE_URL = os.environ.get("DATABASE_URL")

# SQL parameter placeholder — re-evaluated at call time via _ph()
def _ph():
    return "%s" if os.environ.get("DATABASE_URL") else "?"


def _add_column_if_missing(conn, table, column, col_type):
    """Add a column to an existing table if it doesn't already exist."""
    c = conn.cursor()
    if os.environ.get("DATABASE_URL"):
        c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
            (table, column),
        )
        if not c.fetchone():
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()
    else:
        import sqlite3
        c.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in c.fetchall()]
        if column not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()


def _connect():
    url = os.environ.get("DATABASE_URL")
    if url:
        import psycopg2
        return psycopg2.connect(url)
    import sqlite3
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _connect()
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            source_url TEXT,
            filename TEXT,
            processed_at TEXT,
            summary TEXT
        )
    """)
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            concept_title TEXT,
            understanding TEXT,
            concept_type TEXT,
            importance INTEGER,
            section TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    # Migrate existing databases — add columns if they don't exist yet
    _add_column_if_missing(conn, "documents", "summary", "TEXT")
    _add_column_if_missing(conn, "concepts", "concept_type", "TEXT")
    _add_column_if_missing(conn, "concepts", "importance", "INTEGER")
    _add_column_if_missing(conn, "concepts", "section", "TEXT")
    conn.commit()
    conn.close()


def document_exists(source_url):
    conn = _connect()
    c = conn.cursor()
    c.execute(f"SELECT id FROM documents WHERE source_url = {_ph()}", (source_url,))
    row = c.fetchone()
    conn.close()
    return row is not None


def insert_document(title, source_url, filename, summary=None):
    doc_id = str(uuid.uuid4())
    conn = _connect()
    c = conn.cursor()
    c.execute(
        f"INSERT INTO documents VALUES ({_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()})",
        (doc_id, title, source_url, filename, datetime.now().isoformat(), summary),
    )
    conn.commit()
    conn.close()
    return doc_id


def insert_concept(document_id, concept_title, understanding, concept_type=None, importance=None, section=None):
    concept_id = str(uuid.uuid4())
    conn = _connect()
    c = conn.cursor()
    c.execute(
        f"INSERT INTO concepts VALUES ({_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()})",
        (concept_id, document_id, concept_title, understanding, concept_type, importance, section),
    )
    conn.commit()
    conn.close()


def search_concepts(keywords):
    """Search concept understandings for any of the given keywords (case-insensitive)."""
    conn = _connect()
    c = conn.cursor()
    seen = set()
    results = []
    for kw in keywords:
        c.execute(
            f"""
            SELECT c.concept_title, c.understanding, d.title, d.source_url
            FROM concepts c
            JOIN documents d ON c.document_id = d.id
            WHERE LOWER(c.understanding) LIKE LOWER({_ph()})
               OR LOWER(c.concept_title) LIKE LOWER({_ph()})
            """,
            (f"%{kw}%", f"%{kw}%"),
        )
        for row in c.fetchall():
            key = row[0] + row[2]
            if key not in seen:
                seen.add(key)
                results.append(row)
    conn.close()
    return results


def list_documents():
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT id, title, filename, processed_at FROM documents ORDER BY processed_at DESC"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_concept_types():
    """Return the distinct concept types currently in the database."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT DISTINCT concept_type FROM concepts WHERE concept_type IS NOT NULL ORDER BY concept_type")
    types = [row[0] for row in c.fetchall()]
    conn.close()
    return types


def update_concept_importance(concept_id, importance):
    """Update the importance score of a single concept (used by rescore.py)."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        f"UPDATE concepts SET importance = {_ph()} WHERE id = {_ph()}",
        (importance, concept_id),
    )
    conn.commit()
    conn.close()


def get_all_concepts_for_rescore():
    """Return all concepts with enough context for global importance re-ranking."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT c.id, c.concept_title, c.understanding, c.concept_type, c.importance, d.title
        FROM concepts c
        JOIN documents d ON c.document_id = d.id
        ORDER BY d.title, c.concept_title
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def concept_count():
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM concepts")
    n = c.fetchone()[0]
    conn.close()
    return n
