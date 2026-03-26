"""
REST API for the repo-memory-chat semantic memory database.

Endpoints:
  GET /health          — status + concept count
  GET /search?q=...    — search concepts by natural language query
  GET /papers          — list all ingested papers
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from db import concept_count, init_db, list_documents, search_concepts

app = FastAPI(
    title="Repo Memory API",
    description="Semantic memory search over LLM research papers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "concepts": concept_count()}


@app.get("/search")
def search(q: str = Query(..., description="Natural language query")):
    """
    Search concept understandings for the given query.
    Splits the query into keywords and matches each against stored concepts.
    """
    keywords = [kw for kw in q.lower().split() if len(kw) > 2]
    if not keywords:
        return {"query": q, "keywords": [], "results": []}

    rows = search_concepts(keywords)
    results = [
        {
            "concept_title": r[0],
            "understanding": r[1],
            "doc_title": r[2],
            "source_url": r[3],
        }
        for r in rows
    ]
    return {"query": q, "keywords": keywords, "results": results}


@app.get("/papers")
def papers():
    """List all ingested research papers."""
    rows = list_documents()
    return [
        {"id": r[0], "title": r[1], "filename": r[2], "processed_at": r[3]}
        for r in rows
    ]
