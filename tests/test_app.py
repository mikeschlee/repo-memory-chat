"""
Unit tests for the pure LLM helper functions in app.py.

Streamlit is mocked at the session level (conftest.py) so app.py can be
imported without a running Streamlit server. Only extract_keywords() and
answer_with_context() are tested here — Streamlit UI logic is not unit-testable.
"""

import json
from unittest.mock import patch

import app


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

def test_extract_keywords_returns_list(make_claude_response):
    keywords = ["episodic memory", "retrieval", "long context"]
    with patch.object(app.client.messages, "create", return_value=make_claude_response(json.dumps(keywords))):
        result = app.extract_keywords("How do agents manage long-term memory?")
    assert isinstance(result, list)
    assert "episodic memory" in result


def test_extract_keywords_handles_preamble(make_claude_response):
    """Claude sometimes prefixes the JSON with prose."""
    keywords = ["memory", "agents"]
    raw = "Here are the relevant keywords:\n" + json.dumps(keywords)
    with patch.object(app.client.messages, "create", return_value=make_claude_response(raw)):
        result = app.extract_keywords("some question")
    assert result == ["memory", "agents"]


def test_extract_keywords_sends_question_in_prompt(make_claude_response):
    captured = {}

    def capture(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return make_claude_response('["kw"]')

    with patch.object(app.client.messages, "create", side_effect=capture):
        app.extract_keywords("What is hierarchical memory consolidation?")

    assert "What is hierarchical memory consolidation?" in captured["content"]


def test_extract_keywords_returns_multiple_terms(make_claude_response):
    keywords = ["a", "b", "c", "d", "e", "f"]
    with patch.object(app.client.messages, "create", return_value=make_claude_response(json.dumps(keywords))):
        result = app.extract_keywords("complex multi-concept question")
    assert len(result) == 6


# ---------------------------------------------------------------------------
# answer_with_context
# ---------------------------------------------------------------------------

def test_answer_with_context_empty_concepts_returns_fallback():
    result = app.answer_with_context("What is memory?", [])
    assert "No relevant concepts" in result
    assert "ingest.py" in result


def test_answer_with_context_does_not_call_claude_when_empty():
    with patch.object(app.client.messages, "create") as mock_create:
        app.answer_with_context("question", [])
    mock_create.assert_not_called()


def test_answer_with_context_formats_concept_in_prompt(make_claude_response):
    concepts = [
        ("Episodic Memory", "Stores experiences as discrete episodes.", "MemGPT", "https://arxiv.org/abs/2310.08560"),
    ]
    captured = {}

    def capture(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return make_claude_response("Great answer.")

    with patch.object(app.client.messages, "create", side_effect=capture):
        app.answer_with_context("How does memory work?", concepts)

    assert "Episodic Memory" in captured["prompt"]
    assert "Stores experiences as discrete episodes." in captured["prompt"]
    assert "MemGPT" in captured["prompt"]
    assert "https://arxiv.org/abs/2310.08560" in captured["prompt"]


def test_answer_with_context_numbers_multiple_concepts(make_claude_response):
    concepts = [
        ("Concept A", "Understanding A.", "Paper 1", "https://example.com/1"),
        ("Concept B", "Understanding B.", "Paper 2", "https://example.com/2"),
    ]
    captured = {}

    def capture(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return make_claude_response("answer")

    with patch.object(app.client.messages, "create", side_effect=capture):
        app.answer_with_context("question", concepts)

    assert "[1]" in captured["prompt"]
    assert "[2]" in captured["prompt"]


def test_answer_with_context_returns_claude_text(make_claude_response):
    concepts = [("C", "U.", "Doc", "https://example.com")]
    with patch.object(app.client.messages, "create", return_value=make_claude_response("The answer is 42.")):
        result = app.answer_with_context("question", concepts)
    assert result == "The answer is 42."


def test_answer_with_context_includes_question_in_prompt(make_claude_response):
    concepts = [("C", "U.", "Doc", "https://example.com")]
    captured = {}

    def capture(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return make_claude_response("answer")

    with patch.object(app.client.messages, "create", side_effect=capture):
        app.answer_with_context("What is the Ebbinghaus forgetting curve?", concepts)

    assert "What is the Ebbinghaus forgetting curve?" in captured["prompt"]
