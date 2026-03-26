"""
Database layer — supports both SQLite (local/test) and PostgreSQL (production).

When DATABASE_URL env var is set: uses PostgreSQL via psycopg2.
Otherwise: uses SQLite with DB_PATH (monkeypatchable for tests).
"""

import os
import uuid
from datetime import datetime

# PostgreSQL connection string — set this in .env or Streamlit secrets for production
DATABASE_URL = os.environ.get("DATABASE_URL")

# SQLite fallback — used when DATABASE_URL is not set (local dev + tests)
DB_PATH = "memory.db"

# SQL parameter placeholder differs between backends
PH = "%s" if DATABASE_URL else "?"


def _connect():
    if DATABASE_URL:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
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
            processed_at TEXT
        )
    """)
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            concept_title TEXT,
            understanding TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    conn.commit()
    conn.close()


def document_exists(source_url):
    conn = _connect()
    c = conn.cursor()
    c.execute(f"SELECT id FROM documents WHERE source_url = {PH}", (source_url,))
    row = c.fetchone()
    conn.close()
    return row is not None


def insert_document(title, source_url, filename):
    doc_id = str(uuid.uuid4())
    conn = _connect()
    c = conn.cursor()
    c.execute(
        f"INSERT INTO documents VALUES ({PH}, {PH}, {PH}, {PH}, {PH})",
        (doc_id, title, source_url, filename, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return doc_id


def insert_concept(document_id, concept_title, understanding):
    concept_id = str(uuid.uuid4())
    conn = _connect()
    c = conn.cursor()
    c.execute(
        f"INSERT INTO concepts VALUES ({PH}, {PH}, {PH}, {PH})",
        (concept_id, document_id, concept_title, understanding),
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
            WHERE LOWER(c.understanding) LIKE LOWER({PH})
               OR LOWER(c.concept_title) LIKE LOWER({PH})
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


def concept_count():
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM concepts")
    n = c.fetchone()[0]
    conn.close()
    return n
