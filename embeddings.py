"""
Embedding provider — wraps Voyage AI.

Requires VOYAGE_API_KEY in environment/.env/Streamlit secrets.
Model: voyage-3 (1024 dimensions, optimised for retrieval on technical text).
"""

import os
import voyageai

VOYAGE_MODEL = "voyage-3"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set — add it to .env or Streamlit secrets")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    Embed a list of texts.

    input_type:
      "document" — for concept text being stored (ingest / backfill)
      "query"    — for query text being searched (retrieval)
    """
    client = _get_client()
    result = client.embed(texts, model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string for retrieval."""
    return embed_texts([text], input_type="query")[0]


def embed_concept(concept_title: str, understanding: str) -> list[float]:
    """Embed a concept for storage. Combines title + understanding for richer representation."""
    text = f"{concept_title}\n\n{understanding}"
    return embed_texts([text], input_type="document")[0]
