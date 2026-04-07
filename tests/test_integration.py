"""
Integration test — real SQLite + real pdfplumber, only LLM APIs are mocked.

Exercises the full pipeline:
  fixture PDF → ingest_pdf() → concepts in DB → answer_with_context()
"""

import json
from unittest.mock import patch, MagicMock

import db
import memory
import app
import embeddings
from search import ConceptResult


CONCEPTS = [
    {
        "concept_title": "Episodic Memory in Language Models",
        "understanding": (
            "Episodic memory allows language models to store and retrieve discrete "
            "past experiences. This mechanism is central to long-term agent behaviour, "
            "enabling recall of specific events rather than generalised knowledge. "
            "Hierarchical organisation of these episodes supports efficient scanning."
        ),
    },
    {
        "concept_title": "Semantic Memory Compression",
        "understanding": (
            "Semantic memory compression reduces raw text into dense conceptual "
            "summaries that preserve meaning while shrinking storage. This is key "
            "to the retrieval mechanism that scans summaries rather than raw documents."
        ),
    },
]

INGEST_RESPONSE = json.dumps({"summary": "A paper about memory.", "concepts": CONCEPTS})

# Fake 1024-dim embedding returned instead of real Voyage AI calls
FAKE_EMBEDDING = [0.0] * 1024


def test_full_pipeline_ingest_to_answer(tmp_db, fixture_pdf, make_groq_response):
    # ── 1. Ingest ──────────────────────────────────────────────────────────
    with patch.object(
        memory.client.chat.completions, "create",
        return_value=make_groq_response(INGEST_RESPONSE),
    ), patch("memory.embed_texts", return_value=[FAKE_EMBEDDING, FAKE_EMBEDDING]):
        doc_id = memory.ingest_pdf(
            fixture_pdf,
            "Test Memory Paper",
            "https://example.com/test-paper",
        )

    assert db.document_exists("https://example.com/test-paper")
    assert db.concept_count() == 2
    assert any(d[0] == doc_id for d in db.list_documents())

    # ── 2. Verify keyword search works against stored concepts ─────────────
    results = db.search_concepts(["episodic", "hierarchical"])
    assert len(results) >= 1
    titles = [r[0] for r in results]
    assert "Episodic Memory in Language Models" in titles

    concept_title, understanding, doc_title, source_url = results[0]
    assert doc_title == "Test Memory Paper"
    assert source_url == "https://example.com/test-paper"
    assert "episodic" in understanding.lower()

    # ── 3. Answer using ConceptResult objects (real pipeline shape) ────────
    concept = ConceptResult(
        concept_id="fake-id",
        concept_title=concept_title,
        understanding=understanding,
        concept_type=None,
        importance=8,
        section="methods",
        paper_title=doc_title,
        source_url=source_url,
        vector_similarity=0.9,
        keyword_hit=True,
    )

    with patch.object(
        app.client.chat.completions, "create",
        return_value=make_groq_response("Episodic memory stores discrete experiences."),
    ):
        answer = app.answer_with_context(
            "How do language models use episodic memory?",
            [concept],
        )

    assert "Episodic memory" in answer


def test_pipeline_returns_no_results_for_unrelated_query(tmp_db, fixture_pdf, make_groq_response):
    with patch.object(
        memory.client.chat.completions, "create",
        return_value=make_groq_response(INGEST_RESPONSE),
    ), patch("memory.embed_texts", return_value=[FAKE_EMBEDDING, FAKE_EMBEDDING]):
        memory.ingest_pdf(fixture_pdf, "Test Paper", "https://example.com/test")

    results = db.search_concepts(["quantum_tunnelling_photosynthesis_xyz"])
    assert results == []

    answer = app.answer_with_context("Unrelated question", [])
    assert "No relevant concepts" in answer


def test_pipeline_deduplicates_across_keywords(tmp_db, fixture_pdf, make_groq_response):
    with patch.object(
        memory.client.chat.completions, "create",
        return_value=make_groq_response(INGEST_RESPONSE),
    ), patch("memory.embed_texts", return_value=[FAKE_EMBEDDING, FAKE_EMBEDDING]):
        memory.ingest_pdf(fixture_pdf, "Test Paper", "https://example.com/test")

    # Both keywords match the same concept — should appear once
    results = db.search_concepts(["episodic", "long-term"])
    episodic_hits = [r for r in results if r[0] == "Episodic Memory in Language Models"]
    assert len(episodic_hits) == 1
