"""
Document CRUD — dedup on source_url (idempotent ingestion).
"""

import uuid
from datetime import datetime, timezone

from .schema import _connect


def document_exists(source_url: str) -> bool:
    conn = _connect()
    row = conn.execute("SELECT id FROM documents WHERE source_url = ?", [source_url]).fetchone()
    conn.close()
    return row is not None


def insert_document(title: str, source_url: str, filename: str = None, raw_text: str = None) -> str:
    doc_id = str(uuid.uuid4())
    conn = _connect()
    conn.execute(
        "INSERT INTO documents (id, title, source_url, filename, processed_at, raw_text) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [doc_id, title, source_url, filename, datetime.now(timezone.utc).isoformat(), raw_text],
    )
    conn.commit()
    conn.close()
    return doc_id


def list_documents() -> list:
    """Return (id, title, filename, processed_at) — preserves MCP contract shape."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, filename, processed_at FROM documents ORDER BY processed_at DESC"
    ).fetchall()
    conn.close()
    return rows


def get_document_by_id(doc_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, title, source_url, raw_text FROM documents WHERE id = ?",
        [doc_id],
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "title": row[1], "source_url": row[2], "raw_text": row[3]}


def document_count() -> int:
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    return int(n)
