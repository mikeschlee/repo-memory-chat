import json
import pymupdf4llm
import trafilatura
from groq import Groq
from dotenv import load_dotenv

# Must load .env before importing db — db.py reads DATABASE_URL at import time
load_dotenv()

from db import insert_document, insert_concept, get_concept_types
from embeddings import embed_concept, embed_texts
import prompts
client = Groq()
MODEL = "llama-3.3-70b-versatile"

# Max characters of markdown text sent to LLM for concept extraction.
# arxiv papers average ~60k chars; we cap at 80k to stay well within context.
MAX_TEXT_CHARS = 80_000


def pdf_to_markdown(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)


def url_to_markdown(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to fetch URL: {url}")
    markdown = trafilatura.extract(downloaded, output_format="markdown", include_links=False)
    if not markdown:
        raise ValueError(f"Failed to extract content from URL: {url}")
    return markdown


def extract_concepts(text, title, existing_types=None):
    """
    Send document text to LLM and receive back a summary + list of enriched concepts.
    Returns (summary, concepts) where concepts is a list of dicts.
    """
    prompt = prompts.concept_extraction(title, text, MAX_TEXT_CHARS, existing_types)

    last_error = None
    for attempt in range(1, 4):  # up to 3 attempts
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            last_error = ValueError("No JSON object found in response")
            print(f"  Attempt {attempt} failed: no JSON object — retrying...")
            continue
        try:
            result = json.loads(raw[start:end])
            return result.get("summary"), result.get("concepts", [])
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Attempt {attempt} failed: {e} — retrying...")

    raise last_error


def ingest_document(markdown, title, source_url, filename=None):
    """Core ingestion — takes markdown text, extracts concepts, stores to DB."""
    char_count = len(markdown)
    print(f"  {char_count:,} characters extracted")

    existing_types = get_concept_types()
    print(f"  Existing concept types: {existing_types or '(none yet)'}")

    print(f"  Sending to LLM for concept extraction...")
    summary, concepts = extract_concepts(markdown, title, existing_types)
    print(f"  {len(concepts)} concepts identified")
    if summary:
        print(f"  Summary: {summary[:80]}...")

    print(f"  Generating embeddings for {len(concepts)} concepts...")
    texts = [f"{c['concept_title']}\n\n{c['understanding']}" for c in concepts]
    embeddings = embed_texts(texts, input_type="document")

    doc_id = insert_document(title, source_url, filename, summary=summary)
    for concept, embedding in zip(concepts, embeddings):
        insert_concept(
            doc_id,
            concept["concept_title"],
            concept["understanding"],
            concept_type=concept.get("concept_type"),
            importance=concept.get("importance"),
            section=concept.get("section"),
            embedding=embedding,
        )
        type_tag = f"[{concept.get('concept_type', '?')}]"
        score_tag = f"importance={concept.get('importance', '?')}"
        print(f"    + {concept['concept_title']} {type_tag} {score_tag}")

    print(f"  Done — {len(concepts)} concepts stored.")
    return doc_id


def ingest_pdf(pdf_path, title, source_url):
    print(f"\n[ingest] {title}")
    print(f"  Converting PDF to markdown...")
    markdown = pdf_to_markdown(pdf_path)
    return ingest_document(markdown, title, source_url, filename=pdf_path)


def ingest_url(url, title):
    print(f"\n[ingest] {title}")
    print(f"  Fetching and converting {url} to markdown...")
    markdown = url_to_markdown(url)
    return ingest_document(markdown, title, source_url=url, filename=None)
