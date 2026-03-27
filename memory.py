import json
import pdfplumber
import anthropic
from dotenv import load_dotenv

# Must load .env before importing db — db.py reads DATABASE_URL at import time
load_dotenv()

from db import insert_document, insert_concept, get_concept_types
import prompts
client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"  # cost-efficient for bulk ingestion

# Max characters of PDF text sent to Claude for concept extraction.
# arxiv papers average ~60k chars; we cap at 80k to stay well within context.
MAX_TEXT_CHARS = 80_000


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_concepts(text, title, existing_types):
    """
    Send document text to Claude and receive back a summary + list of enriched concepts.
    Returns (summary, concepts) where concepts is a list of dicts.
    """
    prompt = prompts.concept_extraction(title, text, MAX_TEXT_CHARS, existing_types)

    last_error = None
    for attempt in range(1, 4):  # up to 3 attempts
        response = client.messages.create(
            model=MODEL,
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
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


def ingest_pdf(pdf_path, title, source_url):
    print(f"\n[ingest] {title}")
    print(f"  Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    char_count = len(text)
    print(f"  {char_count:,} characters extracted")

    existing_types = get_concept_types()
    print(f"  Existing concept types: {existing_types or '(none yet)'}")

    print(f"  Sending to Claude for concept extraction...")
    summary, concepts = extract_concepts(text, title, existing_types)
    print(f"  {len(concepts)} concepts identified")
    if summary:
        print(f"  Summary: {summary[:80]}...")

    doc_id = insert_document(title, source_url, pdf_path, summary=summary)
    for concept in concepts:
        insert_concept(
            doc_id,
            concept["concept_title"],
            concept["understanding"],
            concept_type=concept.get("concept_type"),
            importance=concept.get("importance"),
            section=concept.get("section"),
        )
        type_tag = f"[{concept.get('concept_type', '?')}]"
        score_tag = f"importance={concept.get('importance', '?')}"
        print(f"    + {concept['concept_title']} {type_tag} {score_tag}")

    print(f"  Done — {len(concepts)} concepts stored.")
    return doc_id
