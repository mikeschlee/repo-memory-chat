import sqlite3
import uuid
from datetime import datetime

DB_PATH = "memory.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            source_url TEXT,
            filename TEXT,
            processed_at TEXT
        )
    """)
    c.execute("""
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM documents WHERE source_url = ?", (source_url,))
    row = c.fetchone()
    conn.close()
    return row is not None


def insert_document(title, source_url, filename):
    doc_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
        (doc_id, title, source_url, filename, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return doc_id


def insert_concept(document_id, concept_title, understanding):
    concept_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO concepts VALUES (?, ?, ?, ?)",
        (concept_id, document_id, concept_title, understanding),
    )
    conn.commit()
    conn.close()


def search_concepts(keywords):
    """Search concept understandings for any of the given keywords (case-insensitive)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    seen = set()
    results = []
    for kw in keywords:
        c.execute(
            """
            SELECT c.concept_title, c.understanding, d.title, d.source_url
            FROM concepts c
            JOIN documents d ON c.document_id = d.id
            WHERE LOWER(c.understanding) LIKE LOWER(?)
               OR LOWER(c.concept_title) LIKE LOWER(?)
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, title, filename, processed_at FROM documents ORDER BY processed_at DESC"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def concept_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM concepts")
    n = c.fetchone()[0]
    conn.close()
    return n
