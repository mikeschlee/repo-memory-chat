"""
Integration test — real SQLite + real pdfplumber, only the Anthropic API is mocked.

This test exercises the full pipeline:
  fixture PDF → ingest_pdf() → concepts in DB → search_concepts() → answer_with_context()
"""

import json
from unittest.mock import patch

import db
import memory
import app


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


def test_full_pipeline_ingest_to_answer(tmp_db, fixture_pdf, make_claude_response):
    # ── 1. Ingest ──────────────────────────────────────────────────────────
    with patch.object(
        memory.client.messages, "create",
        return_value=make_claude_response(json.dumps(CONCEPTS)),
    ):
        doc_id = memory.ingest_pdf(
            fixture_pdf,
            "Test Memory Paper",
            "https://example.com/test-paper",
        )

    # Verify document and concepts are persisted
    assert db.document_exists("https://example.com/test-paper")
    assert db.concept_count() == 2
    docs = db.list_documents()
    assert any(d[0] == doc_id for d in docs)

    # ── 2. Search ──────────────────────────────────────────────────────────
    results = db.search_concepts(["episodic", "hierarchical"])
    assert len(results) >= 1
    titles = [r[0] for r in results]
    assert "Episodic Memory in Language Models" in titles

    # Each result tuple has the expected shape
    concept_title, understanding, doc_title, source_url = results[0]
    assert doc_title == "Test Memory Paper"
    assert source_url == "https://example.com/test-paper"
    assert "episodic" in understanding.lower()

    # ── 3. Answer ──────────────────────────────────────────────────────────
    with patch.object(
        app.client.messages, "create",
        return_value=make_claude_response("Episodic memory stores discrete experiences."),
    ):
        answer = app.answer_with_context(
            "How do language models use episodic memory?",
            results,
        )

    assert "Episodic memory" in answer


def test_pipeline_returns_no_results_for_unrelated_query(tmp_db, fixture_pdf, make_claude_response):
    with patch.object(
        memory.client.messages, "create",
        return_value=make_claude_response(json.dumps(CONCEPTS)),
    ):
        memory.ingest_pdf(fixture_pdf, "Test Paper", "https://example.com/test")

    results = db.search_concepts(["quantum_tunnelling_photosynthesis_xyz"])
    assert results == []

    answer = app.answer_with_context("Unrelated question", [])
    assert "No relevant concepts" in answer


def test_pipeline_deduplicates_across_keywords(tmp_db, fixture_pdf, make_claude_response):
    with patch.object(
        memory.client.messages, "create",
        return_value=make_claude_response(json.dumps(CONCEPTS)),
    ):
        memory.ingest_pdf(fixture_pdf, "Test Paper", "https://example.com/test")

    # Both keywords match the same concept — should appear once
    results = db.search_concepts(["episodic", "long-term"])
    episodic_hits = [r for r in results if r[0] == "Episodic Memory in Language Models"]
    assert len(episodic_hits) == 1
