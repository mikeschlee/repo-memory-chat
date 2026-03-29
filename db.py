"""
Database layer — supports both SQLite (local/test) and PostgreSQL (production).

When DATABASE_URL env var is set: uses PostgreSQL via psycopg2 with pgvector.
Otherwise: uses SQLite with DB_PATH (monkeypatchable for tests); embeddings
stored as JSON text and similarity computed in Python via numpy.
"""

import json
import os
import uuid
from datetime import datetime

# SQLite fallback path — used when DATABASE_URL env var is not set (local dev + tests)
DB_PATH = "memory.db"

# Read at import time for PH only — _connect() always re-reads at call time
# so Streamlit Cloud secrets injected after import still take effect.
DATABASE_URL = os.environ.get("DATABASE_URL")

# Voyage AI embedding dimension
EMBEDDING_DIM = 1024

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

    if os.environ.get("DATABASE_URL"):
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")

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

    # SQLite uses TEXT for embeddings; Postgres uses vector(1024) via pgvector
    embedding_col = f"vector({EMBEDDING_DIM})" if os.environ.get("DATABASE_URL") else "TEXT"
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            concept_title TEXT,
            understanding TEXT,
            concept_type TEXT,
            importance INTEGER,
            section TEXT,
            embedding {embedding_col},
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    # Migrate existing databases — add columns if they don't exist yet
    _add_column_if_missing(conn, "documents", "summary", "TEXT")
    _add_column_if_missing(conn, "concepts", "concept_type", "TEXT")
    _add_column_if_missing(conn, "concepts", "importance", "INTEGER")
    _add_column_if_missing(conn, "concepts", "section", "TEXT")
    embedding_type = f"vector({EMBEDDING_DIM})" if os.environ.get("DATABASE_URL") else "TEXT"
    _add_column_if_missing(conn, "concepts", "embedding", embedding_type)

    # Create ivfflat index on Postgres for fast ANN search (only if table has rows)
    if os.environ.get("DATABASE_URL"):
        c.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'concepts' AND indexname = 'concepts_embedding_idx'
        """)
        if c.fetchone()[0] == 0:
            c.execute("""
                CREATE INDEX IF NOT EXISTS concepts_embedding_idx
                ON concepts USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50)
            """)

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


def insert_concept(document_id, concept_title, understanding, concept_type=None, importance=None, section=None, embedding=None):
    concept_id = str(uuid.uuid4())
    conn = _connect()
    c = conn.cursor()

    # Serialize embedding: pgvector accepts a list directly; SQLite stores as JSON text
    if embedding is not None and not os.environ.get("DATABASE_URL"):
        embedding = json.dumps(embedding)

    c.execute(
        f"INSERT INTO concepts VALUES ({_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()})",
        (concept_id, document_id, concept_title, understanding, concept_type, importance, section, embedding),
    )
    conn.commit()
    conn.close()
    return concept_id


def update_concept_embedding(concept_id, embedding):
    """Update the embedding of an existing concept (used by backfill_embeddings.py)."""
    conn = _connect()
    c = conn.cursor()
    if not os.environ.get("DATABASE_URL"):
        embedding = json.dumps(embedding)
    c.execute(
        f"UPDATE concepts SET embedding = {_ph()} WHERE id = {_ph()}",
        (embedding, concept_id),
    )
    conn.commit()
    conn.close()


def get_concepts_without_embeddings():
    """Return concepts that have no embedding yet — for backfill."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT c.id, c.concept_title, c.understanding
        FROM concepts c
        WHERE c.embedding IS NULL
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def search_concepts_by_keyword(keywords):
    """Keyword substring search. Returns rich rows including metadata."""
    conn = _connect()
    c = conn.cursor()
    seen = set()
    results = []
    for kw in keywords:
        c.execute(
            f"""
            SELECT c.id, c.concept_title, c.understanding, c.concept_type,
                   c.importance, c.section, d.title, d.source_url
            FROM concepts c
            JOIN documents d ON c.document_id = d.id
            WHERE LOWER(c.understanding) LIKE LOWER({_ph()})
               OR LOWER(c.concept_title) LIKE LOWER({_ph()})
            """,
            (f"%{kw}%", f"%{kw}%"),
        )
        for row in c.fetchall():
            if row[0] not in seen:
                seen.add(row[0])
                results.append(row)
    conn.close()
    return results


def search_concepts_by_vector(query_embedding, top_k=30):
    """Vector similarity search. Returns rich rows with similarity score."""
    if os.environ.get("DATABASE_URL"):
        return _vector_search_postgres(query_embedding, top_k)
    return _vector_search_sqlite(query_embedding, top_k)


def _vector_search_postgres(query_embedding, top_k):
    import psycopg2
    conn = _connect()
    c = conn.cursor()
    # pgvector: <=> is cosine distance (lower = more similar), so 1 - distance = similarity
    c.execute(
        """
        SELECT c.id, c.concept_title, c.understanding, c.concept_type,
               c.importance, c.section, d.title, d.source_url,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM concepts c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def _vector_search_sqlite(query_embedding, top_k):
    import numpy as np
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT c.id, c.concept_title, c.understanding, c.concept_type,
               c.importance, c.section, d.title, d.source_url, c.embedding
        FROM concepts c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
    """)
    rows = c.fetchall()
    conn.close()

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    scored = []
    for row in rows:
        emb = np.array(json.loads(row[8]), dtype=np.float32)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        similarity = float(np.dot(q_norm, emb_norm))
        scored.append((*row[:8], similarity))

    scored.sort(key=lambda x: x[8], reverse=True)
    return scored[:top_k]


# Legacy search kept for backward compatibility (used by mcp_server.py)
def search_concepts(keywords):
    """Keyword search — returns (concept_title, understanding, doc_title, source_url)."""
    rows = search_concepts_by_keyword(keywords)
    return [(r[1], r[2], r[6], r[7]) for r in rows]


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
