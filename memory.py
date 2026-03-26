import json
import pdfplumber
import anthropic
from dotenv import load_dotenv

# Must load .env before importing db — db.py reads DATABASE_URL at import time
load_dotenv()

from db import insert_document, insert_concept
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


def extract_concepts(text, title):
    """
    Send document text to Claude and receive back a list of core concepts,
    each with a dense semantic understanding paragraph.
    """
    prompt = prompts.concept_extraction(title, text, MAX_TEXT_CHARS)

    last_error = None
    for attempt in range(1, 4):  # up to 3 attempts
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            last_error = ValueError("No JSON array found in response")
            print(f"  Attempt {attempt} failed: no JSON array — retrying...")
            continue
        try:
            concepts = json.loads(raw[start:end])
            return concepts
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

    print(f"  Sending to Claude for concept extraction...")
    concepts = extract_concepts(text, title)
    print(f"  {len(concepts)} concepts identified")

    doc_id = insert_document(title, source_url, pdf_path)
    for concept in concepts:
        insert_concept(doc_id, concept["concept_title"], concept["understanding"])
        print(f"    + {concept['concept_title']}")

    print(f"  Done — {len(concepts)} concepts stored.")
    return doc_id
