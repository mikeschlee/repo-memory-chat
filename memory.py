import json
import pdfplumber
import anthropic
from dotenv import load_dotenv
from db import insert_document, insert_concept

load_dotenv()
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
    prompt = f"""You are processing a research paper titled: "{title}"

Below is the full paper text (truncated if very long):
---
{text[:MAX_TEXT_CHARS]}
---

Your task is to identify the 8–15 most important core concepts in this paper.

For each concept write a rich semantic understanding — a dense paragraph that captures:
- What the concept is
- Why it matters in the context of this paper
- How it relates to the paper's main contribution
- Key technical details, mechanisms, or nuances a researcher would want to know

These understandings will be stored in a semantic memory database and searched later
to answer questions about this paper WITHOUT re-reading it. Make them thorough and precise.

Return ONLY a valid JSON array with no extra text:
[
  {{
    "concept_title": "Short descriptive title (5-10 words)",
    "understanding": "Dense semantic paragraph (100-200 words)..."
  }},
  ...
]"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    start = raw.find("[")
    end = raw.rfind("]") + 1
    concepts = json.loads(raw[start:end])
    return concepts


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
