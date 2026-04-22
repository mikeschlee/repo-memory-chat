"""
db package — DuckDB-backed storage for the LTM expert.

Greenfield replacement for the top-level SQLite/Postgres `db.py` (which
Agent B/C will remove as they replace dependent call sites).
"""

from .schema import (
    DB_PATH,
    EMBEDDING_DIM,
    SEARCHABLE_CONCEPT_FIELDS,
    _connect,
    close_shared_connection,
    init_db,
    rebuild_fts_index,
)
from .documents import (
    document_count,
    document_exists,
    get_document_by_id,
    insert_document,
    list_documents,
)

__all__ = [
    "DB_PATH",
    "EMBEDDING_DIM",
    "SEARCHABLE_CONCEPT_FIELDS",
    "_connect",
    "close_shared_connection",
    "init_db",
    "rebuild_fts_index",
    "document_count",
    "document_exists",
    "get_document_by_id",
    "insert_document",
    "list_documents",
]
