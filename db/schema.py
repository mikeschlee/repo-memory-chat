"""
Database schema — connection, table creation, and shared constants.
DuckDB backend (single local file; vss + fts extensions).

15 tables per spec §5:
    documents, concepts, rag_chunks,
    synonyms, predicate_canonical,
    preliminary_concept_clusters, concept_clusters, cluster_triples,
    graph_vertices, graph_edges_v2, graph_vertex_concepts,
    pipeline_state, query_step_cache, search_results, usage_log

HNSW (cosine) on concepts.embedding and rag_chunks.embedding.
FTS on concepts(concept_title, understanding) — created on demand after
bulk inserts via `rebuild_fts_index()` (DuckDB FTS is static per build).
"""

import os
import threading

import duckdb

# Voyage AI embedding dimension
EMBEDDING_DIM = 1024

# Columns on the concepts table that are safe to use in keyword search
SEARCHABLE_CONCEPT_FIELDS = {"concept_title", "understanding", "section", "concept_type", "source_text"}

# Path to the local DuckDB file (override via DB_PATH env var)
DB_PATH = os.environ.get("DB_PATH", "./ltm_expert.duckdb")

_extensions_installed = False
_shared_conn: "duckdb.DuckDBPyConnection | None" = None
_conn_lock = threading.Lock()


class _ConnectionProxy:
    """Wraps the shared DuckDB connection. close() is a no-op so callers
    don't accidentally tear down the shared connection."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def close(self):
        pass  # shared connection — never closed by individual callers

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _connect() -> _ConnectionProxy:
    """Return a per-call DuckDB cursor that shares the underlying database
    with the process-wide connection."""
    global _shared_conn, _extensions_installed
    with _conn_lock:
        if _shared_conn is None:
            _shared_conn = duckdb.connect(DB_PATH)
            if not _extensions_installed:
                _shared_conn.execute("INSTALL vss")
                _shared_conn.execute("INSTALL fts")
                _extensions_installed = True
            _shared_conn.execute("LOAD vss")
            _shared_conn.execute("LOAD fts")
            # Required for HNSW on persistent (file-backed) DuckDB.
            _shared_conn.execute("SET hnsw_enable_experimental_persistence = true")
        cursor = _shared_conn.cursor()
    return _ConnectionProxy(cursor)


def close_shared_connection() -> None:
    global _shared_conn
    with _conn_lock:
        if _shared_conn is not None:
            try:
                _shared_conn.close()
            except Exception:
                pass
            _shared_conn = None


def rebuild_fts_index(conn=None) -> None:
    """Rebuild the FTS index on concepts. Call after bulk inserts."""
    close = conn is None
    if conn is None:
        conn = _connect()
    conn.execute(
        "PRAGMA create_fts_index('concepts', 'id', 'concept_title', 'understanding', "
        "stemmer='english', overwrite=1)"
    )
    if close:
        conn.close()


def init_db():
    conn = _connect()

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            source_url TEXT,
            filename TEXT,
            processed_at TEXT,
            raw_text TEXT
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            concept_title TEXT,
            understanding TEXT,
            concept_type TEXT,
            importance INTEGER,
            section TEXT,
            source_text TEXT,
            embedding FLOAT[{EMBEDDING_DIM}],
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            chunk_text TEXT,
            chunk_index INTEGER,
            embedding FLOAT[{EMBEDDING_DIM}],
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS synonyms (
            canonical TEXT NOT NULL,
            aliases JSON NOT NULL DEFAULT '[]',
            entity_type TEXT,
            source TEXT,
            domain TEXT,
            confidence TEXT DEFAULT 'low',
            created_at TEXT,
            PRIMARY KEY (canonical, source)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predicate_canonical (
            raw_predicate TEXT PRIMARY KEY,
            canonical_predicate TEXT NOT NULL,
            direction TEXT DEFAULT 'forward'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS preliminary_concept_clusters (
            concept_id TEXT NOT NULL,
            tag VARCHAR NOT NULL DEFAULT 'legacy',
            cluster_id INTEGER NOT NULL,
            created_at TEXT,
            PRIMARY KEY (concept_id, tag),
            FOREIGN KEY (concept_id) REFERENCES concepts(id)
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS concept_clusters (
            id TEXT PRIMARY KEY,
            preliminary_cluster_id INTEGER NOT NULL,
            sub_topic_index INTEGER NOT NULL DEFAULT 0,
            refined_concept TEXT,
            source_concept_ids JSON NOT NULL DEFAULT '[]',
            embedding FLOAT[{EMBEDDING_DIM}],
            created_at TEXT,
            tag VARCHAR DEFAULT 'legacy'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cluster_triples (
            id TEXT PRIMARY KEY,
            concept_cluster_id TEXT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            source_concept_ids JSON NOT NULL DEFAULT '[]',
            attributes JSON,
            created_at TEXT,
            FOREIGN KEY (concept_cluster_id) REFERENCES concept_clusters(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_vertices (
            id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            entity_type TEXT,
            UNIQUE (canonical_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges_v2 (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT NOT NULL,
            source_concept_ids JSON NOT NULL DEFAULT '[]',
            UNIQUE (subject_id, predicate, object_id),
            FOREIGN KEY (subject_id) REFERENCES graph_vertices(id),
            FOREIGN KEY (object_id) REFERENCES graph_vertices(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_vertex_concepts (
            vertex_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            PRIMARY KEY (vertex_id, concept_id),
            FOREIGN KEY (vertex_id) REFERENCES graph_vertices(id),
            FOREIGN KEY (concept_id) REFERENCES concepts(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_state (
            step_name TEXT PRIMARY KEY,
            completed_at TEXT NOT NULL
        )
    """)

    # Per-step query understanding cache.
    # step ∈ {hyde_vector, entities_keyword, entities_graph}
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_step_cache (
            question_hash TEXT NOT NULL,
            step TEXT NOT NULL,
            question TEXT NOT NULL,
            payload JSON NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (question_hash, step)
        )
    """)

    # Single-run-id "live" tag populated by the retrieval pipeline (spec §5).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_results (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            provider TEXT,
            model TEXT,
            operation TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            estimated_cost_usd REAL,
            run_id TEXT
        )
    """)

    # HNSW index for vector search on concepts and rag_chunks.
    # The persistence flag is a per-cursor setting in recent DuckDB builds, so
    # set it on the active cursor immediately before the DDL.
    try:
        conn.execute("SET hnsw_enable_experimental_persistence = true")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS concepts_embedding_hnsw_idx "
            "ON concepts USING HNSW(embedding) WITH (metric = 'cosine')"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx "
            "ON rag_chunks USING HNSW(embedding) WITH (metric = 'cosine')"
        )
    except Exception as e:
        print(f"[schema] HNSW index creation deferred: {e}")

    conn.close()
