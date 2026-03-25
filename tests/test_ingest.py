"""
Unit tests for ingest.py.

requests, ingest_pdf, and time.sleep are mocked — no network calls or real PDFs.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

import db
import ingest


# ---------------------------------------------------------------------------
# download_pdf
# ---------------------------------------------------------------------------

def test_download_pdf_writes_file(tmp_path):
    output_path = str(tmp_path / "test.pdf")
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"PDF ", b"content"]
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        ingest.download_pdf("2310.08560", output_path)

    assert os.path.exists(output_path)
    with open(output_path, "rb") as f:
        assert f.read() == b"PDF content"


def test_download_pdf_uses_correct_arxiv_url(tmp_path):
    output_path = str(tmp_path / "test.pdf")
    captured_url = {}
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_response.raise_for_status = MagicMock()

    def capture(url, **kwargs):
        captured_url["url"] = url
        return mock_response

    with patch("requests.get", side_effect=capture):
        ingest.download_pdf("2310.08560", output_path)

    assert captured_url["url"] == "https://arxiv.org/pdf/2310.08560.pdf"


def test_download_pdf_raises_on_http_error(tmp_path):
    output_path = str(tmp_path / "test.pdf")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            ingest.download_pdf("bad_id", output_path)


# ---------------------------------------------------------------------------
# main — routing logic
# ---------------------------------------------------------------------------

def test_main_skips_already_ingested_document(tmp_db, monkeypatch):
    monkeypatch.setattr(
        ingest, "PAPERS",
        [{"title": "Existing Paper", "arxiv_id": "1234.56789"}],
    )
    db.insert_document("Existing Paper", "https://arxiv.org/abs/1234.56789", "papers/1234.56789.pdf")

    with patch("ingest.ingest_pdf") as mock_ingest, \
         patch("ingest.download_pdf") as mock_download, \
         patch("sys.argv", ["ingest.py"]):
        ingest.main()

    mock_ingest.assert_not_called()
    mock_download.assert_not_called()


def test_main_skips_on_download_failure(tmp_db, monkeypatch):
    monkeypatch.setattr(
        ingest, "PAPERS",
        [{"title": "New Paper", "arxiv_id": "9999.00001"}],
    )
    with patch("ingest.download_pdf", side_effect=Exception("network error")), \
         patch("ingest.ingest_pdf") as mock_ingest, \
         patch("time.sleep"), \
         patch("sys.argv", ["ingest.py"]):
        ingest.main()  # must not raise

    mock_ingest.assert_not_called()


def test_main_skips_on_ingest_failure(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingest, "PAPERS",
        [{"title": "New Paper", "arxiv_id": "9999.00002"}],
    )
    # Create the PDF file so the download step is skipped
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "9999.00002.pdf").write_bytes(b"fake pdf bytes")

    with patch("ingest.ingest_pdf", side_effect=Exception("parse error")), \
         patch("sys.argv", ["ingest.py"]):
        ingest.main()  # must not raise


def test_main_skips_download_if_pdf_already_on_disk(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingest, "PAPERS",
        [{"title": "Cached Paper", "arxiv_id": "1111.22222"}],
    )
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "1111.22222.pdf").write_bytes(b"cached content")

    with patch("ingest.download_pdf") as mock_dl, \
         patch("ingest.ingest_pdf"), \
         patch("sys.argv", ["ingest.py"]):
        ingest.main()

    mock_dl.assert_not_called()


def test_main_skip_argument_processes_correct_papers(tmp_db, monkeypatch):
    papers = [
        {"title": f"Paper {i}", "arxiv_id": f"000{i}.0000{i}"}
        for i in range(5)
    ]
    monkeypatch.setattr(ingest, "PAPERS", papers)

    ingested_titles = []

    def fake_ingest(pdf_path, title, source_url):
        ingested_titles.append(title)

    with patch("ingest.download_pdf"), \
         patch("ingest.ingest_pdf", side_effect=fake_ingest), \
         patch("time.sleep"), \
         patch("sys.argv", ["ingest.py", "--skip", "3"]):
        ingest.main()

    assert len(ingested_titles) == 2
    assert "Paper 3" in ingested_titles
    assert "Paper 4" in ingested_titles
    assert "Paper 0" not in ingested_titles


def test_main_processes_all_papers_by_default(tmp_db, monkeypatch):
    papers = [
        {"title": f"Paper {i}", "arxiv_id": f"111{i}.1111{i}"}
        for i in range(3)
    ]
    monkeypatch.setattr(ingest, "PAPERS", papers)

    ingested_titles = []

    def fake_ingest(pdf_path, title, source_url):
        ingested_titles.append(title)

    with patch("ingest.download_pdf"), \
         patch("ingest.ingest_pdf", side_effect=fake_ingest), \
         patch("time.sleep"), \
         patch("sys.argv", ["ingest.py"]):
        ingest.main()

    assert len(ingested_titles) == 3
