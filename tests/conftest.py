"""
Shared fixtures for all test modules.

Two things happen at the top of this file before any test imports:
  1. ANTHROPIC_API_KEY is set to a dummy value so anthropic.Anthropic() doesn't
     raise at module-level in memory.py and app.py.
  2. Streamlit is replaced with a MagicMock so app.py can be imported without
     a running Streamlit server.
"""

import os
import sys
from unittest.mock import MagicMock

# Must happen before any project module is imported
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-key-not-real")

# Mock streamlit. Crucially, chat_input() must return None (falsy) so the
# `if prompt := st.chat_input(...)` block in app.py does NOT execute at import
# time and make real API calls.
_mock_st = MagicMock()
_mock_st.chat_input.return_value = None
sys.modules["streamlit"] = _mock_st

import pytest
import db


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """
    Redirect every db.* call to a throwaway SQLite file in tmp_path.
    Also changes cwd to tmp_path so relative paths in ingest.py work cleanly.
    """
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", db_file)
    monkeypatch.chdir(tmp_path)
    db.init_db()
    return db_file


# ---------------------------------------------------------------------------
# Claude response factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_claude_response():
    """
    Returns a factory function that builds a minimal mock Anthropic response.

    Usage:
        mock_resp = make_claude_response('["keyword1", "keyword2"]')
        mock_client.messages.create.return_value = mock_resp
    """
    def _make(text: str):
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        return resp
    return _make


# ---------------------------------------------------------------------------
# Fixture PDF (used by integration test)
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_pdf(tmp_path):
    """
    Generate a small valid PDF with known text content using fpdf2.
    pdfplumber will extract real text from this file.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(
        0, 10,
        text="Memory augmented language models use episodic memory for long-term storage.",
    )
    pdf.ln()
    pdf.cell(
        0, 10,
        text="Hierarchical memory structures allow agents to recall relevant past experiences.",
    )
    pdf.ln()
    pdf.cell(
        0, 10,
        text="The retrieval mechanism scans compressed semantic summaries rather than raw text.",
    )
    path = str(tmp_path / "fixture.pdf")
    pdf.output(path)
    return path
