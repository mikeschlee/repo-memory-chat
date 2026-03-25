"""
Unit tests for db.py — all run against a throwaway temp database.
No mocking needed: SQLite is fast and side-effect-free via the tmp_db fixture.
"""

import re
import sqlite3
import time

import db


def test_init_creates_tables(tmp_db):
    conn = sqlite3.connect(tmp_db)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "documents" in tables
    assert "concepts" in tables


def test_insert_document_returns_uuid(tmp_db):
    doc_id = db.insert_document("Test Paper", "https://example.com", "test.pdf")
    assert re.match(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", doc_id)


def test_insert_document_persists(tmp_db):
    db.insert_document("Test Paper", "https://example.com", "test.pdf")
    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0][1] == "Test Paper"


def test_document_exists_true(tmp_db):
    db.insert_document("Paper", "https://example.com/paper", "p.pdf")
    assert db.document_exists("https://example.com/paper") is True


def test_document_exists_false(tmp_db):
    assert db.document_exists("https://example.com/nonexistent") is False


def test_insert_concept_persists(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Test Concept", "This concept covers episodic memory.")
    results = db.search_concepts(["episodic"])
    assert len(results) == 1
    assert results[0][0] == "Test Concept"


def test_search_concepts_matches_understanding(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Memory Architecture", "Uses hierarchical episodic memory.")
    results = db.search_concepts(["hierarchical"])
    assert any(r[0] == "Memory Architecture" for r in results)


def test_search_concepts_matches_title(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Episodic Memory Store", "Stores past experiences.")
    results = db.search_concepts(["Episodic Memory Store"])
    assert len(results) == 1


def test_search_concepts_case_insensitive(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Concept", "Uses TRANSFORMER architecture.")
    # lowercase search should match uppercase stored text
    results = db.search_concepts(["transformer"])
    assert len(results) == 1


def test_search_concepts_no_match(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Concept", "Uses transformer architecture.")
    results = db.search_concepts(["quantum_bananas"])
    assert results == []


def test_search_concepts_deduplicates(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Memory Concept", "episodic memory retrieval systems.")
    # Both keywords match the same concept — must appear only once
    results = db.search_concepts(["episodic", "memory"])
    assert len(results) == 1


def test_search_concepts_multi_keyword_merges_results(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "Concept A", "Uses attention mechanisms.")
    db.insert_concept(doc_id, "Concept B", "Applies gradient descent optimization.")
    results = db.search_concepts(["attention", "gradient"])
    titles = {r[0] for r in results}
    assert "Concept A" in titles
    assert "Concept B" in titles


def test_search_concepts_result_shape(tmp_db):
    doc_id = db.insert_document("My Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "A Concept", "Some understanding text.")
    results = db.search_concepts(["understanding"])
    concept_title, understanding, doc_title, source_url = results[0]
    assert concept_title == "A Concept"
    assert understanding == "Some understanding text."
    assert doc_title == "My Paper"
    assert source_url == "https://example.com"


def test_concept_count_empty(tmp_db):
    assert db.concept_count() == 0


def test_concept_count_after_inserts(tmp_db):
    doc_id = db.insert_document("Paper", "https://example.com", "p.pdf")
    db.insert_concept(doc_id, "C1", "Understanding one.")
    db.insert_concept(doc_id, "C2", "Understanding two.")
    db.insert_concept(doc_id, "C3", "Understanding three.")
    assert db.concept_count() == 3


def test_list_documents_ordered_newest_first(tmp_db):
    db.insert_document("Paper A", "https://example.com/a", "a.pdf")
    time.sleep(0.01)
    db.insert_document("Paper B", "https://example.com/b", "b.pdf")
    docs = db.list_documents()
    assert docs[0][1] == "Paper B"
    assert docs[1][1] == "Paper A"


def test_list_documents_returns_all_fields(tmp_db):
    db.insert_document("Paper", "https://example.com", "paper.pdf")
    docs = db.list_documents()
    doc_id, title, filename, processed_at = docs[0]
    assert title == "Paper"
    assert filename == "paper.pdf"
    assert processed_at is not None
