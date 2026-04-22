"""
Step 1 — Corpus ingestion.

Reads papers.yaml and, per entry:
  - arxiv: download https://arxiv.org/pdf/<id>.pdf → papers/<id>.pdf → pymupdf4llm.to_markdown()
  - url:   trafilatura.fetch_url() + trafilatura.extract(output_format='markdown')

Writes rows to `documents` with raw_text populated. Dedup key: documents.source_url
(re-running is idempotent).

Manual add flags kept: --url URL --title TITLE.

Usage:
    python -m pipeline.ingest                         # full corpus
    python -m pipeline.ingest --limit 2               # smoke
    python -m pipeline.ingest --url URL --title TITLE # one-off web page
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF, for raw-text fallback when pymupdf4llm's markdown fails
import pymupdf4llm
import requests
import trafilatura
import yaml
from dotenv import load_dotenv

load_dotenv()

from db import document_exists, init_db, insert_document
from db.schema import _connect

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_YAML = REPO_ROOT / "papers.yaml"
PDF_DIR = REPO_ROOT / "papers"

ARXIV_DELAY_SEC = 3   # polite delay between arxiv requests
HTTP_TIMEOUT = 60


# ── Source fetchers ────────────────────────────────────────────────────────────

def arxiv_source_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"    downloading {url}")
    headers = {"User-Agent": "Mozilla/5.0 (research/ltm-expert)"}
    r = requests.get(url, stream=True, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"    saved {size_kb} KB → {output_path}")


_PDF_MARKDOWN_MIN_RATIO = 0.2


def pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a PDF to markdown. pymupdf4llm occasionally collapses output on
    layouts it can't parse (e.g. two-column scans) and returns a handful of
    chars for a PDF with a full text layer. Detect that, and fall back to raw
    text from fitz rather than silently losing the content."""
    md = pymupdf4llm.to_markdown(str(pdf_path))

    doc = fitz.open(str(pdf_path))
    raw = "\n".join(p.get_text() for p in doc)
    doc.close()

    if raw and len(md) < _PDF_MARKDOWN_MIN_RATIO * len(raw):
        print(f"    [pdf] markdown conversion degraded "
              f"({len(md):,} vs raw {len(raw):,} chars) — falling back to raw text")
        return raw
    return md


def url_to_markdown(url: str, timeout: int = 30) -> str:
    # trafilatura's internal fetch can hang on some hosts; give it a hard ceiling.
    try:
        downloaded = trafilatura.fetch_url(url, config=None)
    except Exception as e:
        raise ValueError(f"trafilatura.fetch_url raised {type(e).__name__}: {e}")
    if not downloaded:
        # Fall back to requests with a hard timeout so a dead host doesn't stall ingestion.
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research/ltm-expert)"})
            resp.raise_for_status()
            downloaded = resp.text
        except Exception as e:
            raise ValueError(f"fetch failed for {url}: {type(e).__name__}: {e}")
    markdown = trafilatura.extract(downloaded, output_format="markdown", include_links=False)
    if not markdown:
        raise ValueError(f"trafilatura.extract produced empty output for {url}")
    return markdown


# ── Raw-text conversion dispatcher ────────────────────────────────────────────
#
# Handler selection: URL suffix first (cheap and usually decisive), then HTTP
# Content-Type as fallback for opaque URLs (e.g. ?id=42). Unknown types raise —
# silent empties were the root cause of the v3 ingest failures.

_PDF_SUFFIXES = (".pdf",)
_HTML_SUFFIXES = (".html", ".htm")
_MARKDOWN_SUFFIXES = (".md", ".markdown")
_TEXT_SUFFIXES = (".txt",)
_STRUCTURED_SUFFIXES = (".yml", ".yaml", ".json")

_PDF_MIMES = ("application/pdf",)
_HTML_MIMES = ("text/html", "application/xhtml+xml")
_MARKDOWN_MIMES = ("text/markdown", "text/x-markdown")
_TEXT_MIMES = ("text/plain",)
_STRUCTURED_MIMES = ("text/yaml", "application/yaml", "application/x-yaml",
                     "application/json", "text/json")


def _url_suffix(url: str) -> str:
    path = url.split("?", 1)[0].split("#", 1)[0]
    return Path(path).suffix.lower()


def _http_get(url: str, timeout: int = HTTP_TIMEOUT):
    headers = {"User-Agent": "Mozilla/5.0 (research/ltm-expert)"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp


def _guess_mime(resp) -> str:
    ct = resp.headers.get("Content-Type", "") if getattr(resp, "headers", None) else ""
    return ct.split(";", 1)[0].strip().lower()


def _convert_pdf_bytes(body: bytes, url: str) -> str:
    tmp_dir = PDF_DIR
    tmp_dir.mkdir(exist_ok=True)
    # Stable on-disk name so re-runs don't accumulate tempfiles.
    safe = url.rsplit("/", 1)[-1].split("?", 1)[0] or "download.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    pdf_path = tmp_dir / safe
    pdf_path.write_bytes(body)
    return pdf_to_markdown(pdf_path)


def fetch_and_convert(url: str, timeout: int = HTTP_TIMEOUT) -> str:
    """Fetch `url` and return its content as text/markdown, dispatching on
    suffix first and Content-Type second. Raises ValueError on unknown types
    or empty conversions — never returns empty silently."""
    suffix = _url_suffix(url)

    # HTML is the common case where we prefer the smart extractor over raw bytes.
    if suffix in _HTML_SUFFIXES or suffix == "":
        # Suffix-less URL: peek at Content-Type before committing to HTML.
        if suffix == "":
            resp = _http_get(url, timeout=timeout)
            mime = _guess_mime(resp)
            if mime in _PDF_MIMES:
                out = _convert_pdf_bytes(resp.content, url)
            elif mime in _HTML_MIMES:
                out = trafilatura.extract(resp.text, output_format="markdown", include_links=False) or ""
            elif mime in _MARKDOWN_MIMES or mime in _TEXT_MIMES or mime in _STRUCTURED_MIMES:
                out = resp.text
            else:
                raise ValueError(
                    f"fetch_and_convert: unsupported Content-Type {mime!r} for {url} "
                    f"(no recognized suffix either)"
                )
        else:
            out = url_to_markdown(url, timeout=timeout)

    elif suffix in _PDF_SUFFIXES:
        resp = _http_get(url, timeout=timeout)
        out = _convert_pdf_bytes(resp.content, url)

    elif suffix in _MARKDOWN_SUFFIXES + _TEXT_SUFFIXES + _STRUCTURED_SUFFIXES:
        resp = _http_get(url, timeout=timeout)
        out = resp.text

    else:
        # Peek at Content-Type before giving up — sometimes suffixes lie.
        resp = _http_get(url, timeout=timeout)
        mime = _guess_mime(resp)
        if mime in _PDF_MIMES:
            out = _convert_pdf_bytes(resp.content, url)
        elif mime in _HTML_MIMES:
            out = trafilatura.extract(resp.text, output_format="markdown", include_links=False) or ""
        elif mime in _MARKDOWN_MIMES + _TEXT_MIMES + _STRUCTURED_MIMES:
            out = resp.text
        else:
            raise ValueError(
                f"fetch_and_convert: unsupported type for {url} "
                f"(suffix={suffix!r}, Content-Type={mime!r})"
            )

    if not out or not out.strip():
        raise ValueError(f"fetch_and_convert: empty output for {url}")
    return out


# ── Entry processing ──────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/build_papers_yaml.py first")
    data = yaml.safe_load(path.read_text()) or []
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a YAML list at the top level")
    return data


def _validate_entry(entry: dict, idx: int) -> tuple[str, str, str]:
    """Return (title, source_type, source_id_or_url) — raise ValueError with context if bad."""
    missing = [k for k in ("title", "source_type", "source_id_or_url") if not entry.get(k)]
    if missing:
        raise ValueError(f"papers.yaml entry #{idx} missing keys {missing}: {entry!r}")
    source_type = entry["source_type"]
    if source_type not in ("arxiv", "url"):
        raise ValueError(f"papers.yaml entry #{idx} has unknown source_type {source_type!r}")
    return entry["title"], source_type, str(entry["source_id_or_url"]).strip()


def _ingest_arxiv(title: str, arxiv_id: str) -> tuple[str, int] | None:
    """Returns (doc_id, char_count) on success, None if the source_url already exists."""
    source_url = arxiv_source_url(arxiv_id)
    if document_exists(source_url):
        print("    already ingested — skipping")
        return None

    PDF_DIR.mkdir(exist_ok=True)
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
    downloaded_now = False
    if not pdf_path.exists():
        download_arxiv_pdf(arxiv_id, pdf_path)
        downloaded_now = True

    markdown = pdf_to_markdown(pdf_path)
    if not markdown or not markdown.strip():
        raise ValueError(f"pymupdf4llm produced empty markdown for {pdf_path}")

    doc_id = insert_document(
        title=title,
        source_url=source_url,
        filename=str(pdf_path),
        raw_text=markdown,
    )
    if downloaded_now:
        time.sleep(ARXIV_DELAY_SEC)  # polite delay only between live fetches
    return doc_id, len(markdown)


def _ingest_url(title: str, url: str) -> tuple[str, int] | None:
    if document_exists(url):
        print("    already ingested — skipping")
        return None
    text = fetch_and_convert(url)
    doc_id = insert_document(
        title=title,
        source_url=url,
        filename=None,
        raw_text=text,
    )
    return doc_id, len(text)


def _record_step(step_name: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_state (step_name, completed_at) VALUES (?, ?)",
        [step_name, datetime.now(timezone.utc).isoformat()],
    )
    conn.commit()
    conn.close()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest LTM expert corpus from papers.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Ingest at most N entries (smoke test)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N entries (resume after failure)")
    parser.add_argument("--url", type=str, default=None, help="Manual add: ingest a single web article by URL")
    parser.add_argument("--title", type=str, default=None, help="Title for --url (required with --url)")
    args = parser.parse_args()

    init_db()

    # Manual add mode — bypass papers.yaml.
    if args.url:
        if not args.title:
            parser.error("--title is required when using --url")
        print(f"\n[ingest] Manual URL add: {args.title}")
        print(f"    {args.url}")
        try:
            result = _ingest_url(args.title, args.url)
        except Exception as e:
            print(f"    ERROR: {e}")
            return 1
        if result:
            doc_id, n = result
            print(f"    inserted {doc_id} ({n:,} chars)")
        return 0

    entries = _load_yaml(PAPERS_YAML)
    total = len(entries)
    entries = entries[args.skip:]
    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"\n[ingest] papers.yaml: {total} entries (running {len(entries)}, skipping first {args.skip})")

    inserted = 0
    skipped = 0
    failures: list[tuple[str, str, str]] = []  # (title, source_id_or_url, error)

    for i, entry in enumerate(entries, start=args.skip + 1):
        try:
            title, source_type, sid = _validate_entry(entry, i)
        except ValueError as e:
            print(f"[{i}/{total}] INVALID ENTRY: {e}")
            failures.append((entry.get("title", "<no title>"), repr(entry), str(e)))
            continue

        print(f"[{i}/{total}] [{source_type}] {title}")
        try:
            if source_type == "arxiv":
                result = _ingest_arxiv(title, sid)
            else:  # url
                result = _ingest_url(title, sid)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            failures.append((title, sid, f"{type(e).__name__}: {e}"))
            continue

        if result is None:
            skipped += 1
        else:
            doc_id, n = result
            inserted += 1
            print(f"    inserted {doc_id} ({n:,} chars)")

    print(f"\n[ingest] Done.")
    print(f"  Inserted:          {inserted}")
    print(f"  Already existed:   {skipped}")
    print(f"  Failed:            {len(failures)}")
    if failures:
        print(f"\n  Failure details:")
        for title, sid, err in failures:
            print(f"    - {title!r} ({sid}): {err}")

    _record_step("ingest")

    conn = _connect()
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    # Force a checkpoint so the WAL merges into the main DB file — otherwise a
    # crash or hard-kill between process exit and DuckDB's lazy checkpoint drops
    # all un-merged inserts.
    conn.execute("CHECKPOINT")
    conn.commit()
    conn.close()

    # Drop the process-wide connection cleanly so the file lock is released
    # and a follow-up --idempotency check can open the DB immediately.
    from db.schema import close_shared_connection
    close_shared_connection()

    print(f"\n  Total documents in DB: {total_docs}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
