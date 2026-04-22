"""
Unit tests for pipeline.ingest.fetch_and_convert — the content-type dispatcher.

Network calls are mocked. The dispatcher picks its handler from the URL suffix
first, then from the HTTP Content-Type header if the suffix is ambiguous.
Unknown types must fail loudly (no silent empties).
"""

from unittest.mock import MagicMock, patch

import pytest

from pipeline import ingest


def _mock_http_get(body: bytes, content_type: str):
    resp = MagicMock()
    resp.content = body
    resp.text = body.decode("utf-8", errors="replace")
    resp.headers = {"Content-Type": content_type}
    resp.iter_content.return_value = [body]
    resp.raise_for_status = MagicMock()
    return resp


# ── Suffix-based dispatch ─────────────────────────────────────────────────────

def test_pdf_url_routes_to_pdf_handler(tmp_path):
    url = "https://example.org/paper.pdf"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(b"%PDF-1.4 fake", "application/pdf")), \
         patch("pipeline.ingest.pdf_to_markdown", return_value="# Converted PDF") as m:
        out = ingest.fetch_and_convert(url)
    assert m.called, "PDF URL must be converted via pdf_to_markdown"
    assert out == "# Converted PDF"


def test_html_suffix_routes_to_trafilatura():
    url = "https://example.org/article.html"
    with patch("pipeline.ingest.trafilatura.fetch_url", return_value="<html><body>hi</body></html>"), \
         patch("pipeline.ingest.trafilatura.extract", return_value="hi") as m:
        out = ingest.fetch_and_convert(url)
    assert m.called
    assert out == "hi"


def test_extensionless_html_routes_to_trafilatura_extract():
    url = "https://example.org/article"
    body = b"<html><body>hello world</body></html>"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(body, "text/html")), \
         patch("pipeline.ingest.trafilatura.extract", return_value="hello world") as m:
        out = ingest.fetch_and_convert(url)
    assert m.called
    assert out == "hello world"


def test_markdown_url_passes_through():
    url = "https://example.org/NOTES.md"
    body = b"# heading\n\nbody text\n"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(body, "text/markdown")):
        out = ingest.fetch_and_convert(url)
    assert out == body.decode("utf-8")


def test_plain_text_url_passes_through():
    url = "https://example.org/readme.txt"
    body = b"plain text content"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(body, "text/plain")):
        out = ingest.fetch_and_convert(url)
    assert out == "plain text content"


def test_yaml_url_passes_through():
    url = "https://example.org/config.yaml"
    body = b"key: value\nlist:\n  - a\n  - b\n"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(body, "text/yaml")):
        out = ingest.fetch_and_convert(url)
    assert "key: value" in out


# ── Header-based dispatch (no informative suffix) ─────────────────────────────

def test_extensionless_url_dispatches_on_content_type_pdf():
    """URL has no suffix; Content-Type header says PDF → PDF handler."""
    url = "https://example.org/download?id=42"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(b"%PDF-1.4 ...", "application/pdf")), \
         patch("pipeline.ingest.pdf_to_markdown", return_value="# via header") as m:
        out = ingest.fetch_and_convert(url)
    assert m.called
    assert out == "# via header"


# ── Unknown type — loud failure ───────────────────────────────────────────────

def test_unknown_content_type_fails_loudly():
    """No recognized suffix and unknown Content-Type → raise with context."""
    url = "https://example.org/thing.xyz"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(b"\x00\x01\x02", "application/x-weird")):
        with pytest.raises(ValueError) as ei:
            ingest.fetch_and_convert(url)
    msg = str(ei.value)
    assert "application/x-weird" in msg
    assert url in msg


def test_pdf_markdown_degradation_falls_back_to_raw_text(tmp_path):
    """pymupdf4llm sometimes returns drastically truncated markdown on PDFs it
    can't layout (e.g. two-column scans). When the markdown is <20% of the
    raw text-layer size, pdf_to_markdown must fall back to raw text rather
    than silently lose most of the content."""
    pdf = tmp_path / "lossy.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_doc = MagicMock()
    fake_doc.__len__ = lambda self: 2
    page = MagicMock()
    page.get_text.return_value = "a" * 50_000  # 100k raw text
    fake_doc.__iter__ = lambda self: iter([page, page])

    with patch("pipeline.ingest.pymupdf4llm.to_markdown", return_value="tiny"), \
         patch("pipeline.ingest.fitz.open", return_value=fake_doc):
        out = ingest.pdf_to_markdown(pdf)

    assert len(out) > 50_000, f"expected raw-text fallback, got {len(out)} chars"


def test_empty_conversion_fails_loudly():
    """A recognized type that produces empty output must still raise."""
    url = "https://example.org/empty.pdf"
    with patch("pipeline.ingest.requests.get",
               return_value=_mock_http_get(b"%PDF-1.4", "application/pdf")), \
         patch("pipeline.ingest.pdf_to_markdown", return_value="   "):
        with pytest.raises(ValueError) as ei:
            ingest.fetch_and_convert(url)
    assert "empty" in str(ei.value).lower()
