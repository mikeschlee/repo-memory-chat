"""
Unit tests for the pure LLM helper functions in app.py.

Streamlit is mocked at the session level (conftest.py) so app.py can be
imported without a running Streamlit server. Only answer_with_context() is
tested here — Streamlit UI logic and run_search() are tested elsewhere.
"""

from unittest.mock import patch, MagicMock

import app
from search import ConceptResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(**kwargs) -> ConceptResult:
    defaults = dict(
        concept_id="id-1",
        concept_title="Episodic Memory",
        understanding="Stores experiences as discrete episodes.",
        concept_type="memory",
        importance=8,
        section="methods",
        paper_title="MemGPT",
        source_url="https://arxiv.org/abs/2310.08560",
        vector_similarity=0.9,
        keyword_hit=True,
    )
    defaults.update(kwargs)
    return ConceptResult(**defaults)


# ---------------------------------------------------------------------------
# answer_with_context — empty concepts
# ---------------------------------------------------------------------------

def test_answer_with_context_empty_concepts_returns_fallback():
    result = app.answer_with_context("What is memory?", [])
    assert "No relevant concepts" in result
    assert "ingest.py" in result


def test_answer_with_context_does_not_call_llm_when_empty():
    with patch.object(app.client.chat.completions, "create") as mock_create:
        app.answer_with_context("question", [])
    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# answer_with_context — with concepts
# ---------------------------------------------------------------------------

def test_answer_with_context_returns_llm_text(make_groq_response):
    concepts = [_make_concept()]
    with patch.object(app.client.chat.completions, "create",
                      return_value=make_groq_response("The answer is 42.")):
        result = app.answer_with_context("question", concepts)
    assert result == "The answer is 42."


def test_answer_with_context_formats_concept_in_prompt(make_groq_response):
    concepts = [_make_concept(
        concept_title="Episodic Memory",
        understanding="Stores experiences as discrete episodes.",
        paper_title="MemGPT",
        source_url="https://arxiv.org/abs/2310.08560",
    )]
    captured = {}

    def capture(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return make_groq_response("Great answer.")

    with patch.object(app.client.chat.completions, "create", side_effect=capture):
        app.answer_with_context("How does memory work?", concepts)

    assert "Episodic Memory" in captured["prompt"]
    assert "Stores experiences as discrete episodes." in captured["prompt"]
    assert "MemGPT" in captured["prompt"]
    assert "https://arxiv.org/abs/2310.08560" in captured["prompt"]


def test_answer_with_context_numbers_multiple_concepts(make_groq_response):
    concepts = [
        _make_concept(concept_id="id-1", concept_title="Concept A"),
        _make_concept(concept_id="id-2", concept_title="Concept B"),
    ]
    captured = {}

    def capture(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return make_groq_response("answer")

    with patch.object(app.client.chat.completions, "create", side_effect=capture):
        app.answer_with_context("question", concepts)

    assert "[1]" in captured["prompt"]
    assert "[2]" in captured["prompt"]


def test_answer_with_context_includes_question_in_prompt(make_groq_response):
    concepts = [_make_concept()]
    captured = {}

    def capture(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return make_groq_response("answer")

    with patch.object(app.client.chat.completions, "create", side_effect=capture):
        app.answer_with_context("What is the Ebbinghaus forgetting curve?", concepts)

    assert "What is the Ebbinghaus forgetting curve?" in captured["prompt"]
