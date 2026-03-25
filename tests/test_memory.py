"""
Unit tests for memory.py.

pdfplumber and the Anthropic client are mocked — no real PDFs or API calls.
DB operations use the tmp_db fixture so they don't touch memory.db.
"""

import json
from unittest.mock import MagicMock, patch

import memory
import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pdf(pages_text: list[str | None]):
    """Build a mock pdfplumber context manager with the given per-page text."""
    pages = []
    for text in pages_text:
        p = MagicMock()
        p.extract_text.return_value = text
        pages.append(p)
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = pages
    return mock_pdf


# ---------------------------------------------------------------------------
# extract_text_from_pdf
# ---------------------------------------------------------------------------

def test_extract_text_concatenates_pages():
    mock_pdf = _make_mock_pdf(["Page one content.", "Page two content."])
    with patch("pdfplumber.open", return_value=mock_pdf):
        text = memory.extract_text_from_pdf("fake.pdf")
    assert "Page one content." in text
    assert "Page two content." in text


def test_extract_text_skips_none_pages():
    mock_pdf = _make_mock_pdf([None, "Good page.", None])
    with patch("pdfplumber.open", return_value=mock_pdf):
        text = memory.extract_text_from_pdf("fake.pdf")
    assert "Good page." in text
    assert "None" not in text


def test_extract_text_empty_pdf_returns_empty_string():
    mock_pdf = _make_mock_pdf([])
    with patch("pdfplumber.open", return_value=mock_pdf):
        text = memory.extract_text_from_pdf("fake.pdf")
    assert text == ""


def test_extract_text_joins_with_newlines():
    mock_pdf = _make_mock_pdf(["First.", "Second."])
    with patch("pdfplumber.open", return_value=mock_pdf):
        text = memory.extract_text_from_pdf("fake.pdf")
    assert text == "First.\nSecond.\n"


# ---------------------------------------------------------------------------
# extract_concepts
# ---------------------------------------------------------------------------

def test_extract_concepts_parses_json(make_claude_response):
    concepts = [
        {"concept_title": "Hierarchical Memory", "understanding": "A multi-tier memory system."},
        {"concept_title": "Memory Paging", "understanding": "Swapping context like an OS."},
    ]
    with patch.object(memory.client.messages, "create", return_value=make_claude_response(json.dumps(concepts))):
        result = memory.extract_concepts("Some paper text", "Test Paper")
    assert len(result) == 2
    assert result[0]["concept_title"] == "Hierarchical Memory"
    assert result[1]["understanding"] == "Swapping context like an OS."


def test_extract_concepts_handles_preamble(make_claude_response):
    """Claude sometimes adds prose before the JSON array — find('[') must handle it."""
    concepts = [{"concept_title": "Memory", "understanding": "Core idea."}]
    raw = "Sure! Here are the concepts:\n" + json.dumps(concepts)
    with patch.object(memory.client.messages, "create", return_value=make_claude_response(raw)):
        result = memory.extract_concepts("text", "Paper")
    assert result[0]["concept_title"] == "Memory"


def test_extract_concepts_truncates_long_text(make_claude_response):
    """Text over MAX_TEXT_CHARS must be sliced before sending to Claude."""
    concepts = [{"concept_title": "C", "understanding": "U."}]
    mock_response = make_claude_response(json.dumps(concepts))
    # Use a unique marker at the overflow boundary so substring matching works
    long_text = "x" * memory.MAX_TEXT_CHARS + "OVERFLOW_MARKER_XYZ"

    captured = {}

    def capture_call(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return mock_response

    with patch.object(memory.client.messages, "create", side_effect=capture_call):
        memory.extract_concepts(long_text, "Paper")

    assert "OVERFLOW_MARKER_XYZ" not in captured["content"]


def test_extract_concepts_sends_title_in_prompt(make_claude_response):
    concepts = [{"concept_title": "C", "understanding": "U."}]
    captured = {}

    def capture_call(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return make_claude_response(json.dumps(concepts))

    with patch.object(memory.client.messages, "create", side_effect=capture_call):
        memory.extract_concepts("paper text", "My Unique Title XYZ")

    assert "My Unique Title XYZ" in captured["content"]


# ---------------------------------------------------------------------------
# ingest_pdf (integration of extract_text + extract_concepts + db writes)
# ---------------------------------------------------------------------------

def test_ingest_pdf_stores_concepts(tmp_db, make_claude_response):
    concepts = [
        {"concept_title": "Episodic Memory", "understanding": "Stores past experiences as episodes."},
        {"concept_title": "Memory Retrieval", "understanding": "Scans stored memories to find relevant ones."},
    ]
    mock_pdf = _make_mock_pdf(["Paper text about memory."])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch.object(memory.client.messages, "create", return_value=make_claude_response(json.dumps(concepts))):
        memory.ingest_pdf("fake.pdf", "Test Paper", "https://example.com")

    results = db.search_concepts(["episodic"])
    assert len(results) == 1
    assert results[0][0] == "Episodic Memory"


def test_ingest_pdf_returns_valid_doc_id(tmp_db, make_claude_response):
    concepts = [{"concept_title": "C", "understanding": "U."}]
    mock_pdf = _make_mock_pdf(["text"])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch.object(memory.client.messages, "create", return_value=make_claude_response(json.dumps(concepts))):
        doc_id = memory.ingest_pdf("fake.pdf", "Test Paper", "https://example.com")

    assert any(d[0] == doc_id for d in db.list_documents())


def test_ingest_pdf_stores_all_concepts(tmp_db, make_claude_response):
    concepts = [
        {"concept_title": f"Concept {i}", "understanding": f"Understanding {i}."}
        for i in range(5)
    ]
    mock_pdf = _make_mock_pdf(["text"])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch.object(memory.client.messages, "create", return_value=make_claude_response(json.dumps(concepts))):
        memory.ingest_pdf("fake.pdf", "Test Paper", "https://example.com")

    assert db.concept_count() == 5
