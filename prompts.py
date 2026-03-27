"""
LLM prompt templates. Edit this file to tune model instructions.
"""


def keyword_extraction(question: str) -> str:
    return f"""You are a search query analyst.

Given the user question below, extract 6–12 keywords or short phrases that would
best match relevant concepts stored in a research paper memory database.

Focus on:
- Technical terms and named methods/architectures
- Core concepts the question is about
- Synonyms or related terms that might appear in paper descriptions

Return ONLY a valid JSON array of strings with no extra text.
Example: ["memory augmented", "episodic memory", "retrieval", "long context"]

Question: {question}"""


def global_rescore(concepts_text: str) -> str:
    return f"""You are re-scoring a set of research paper concepts on a GLOBAL importance scale.

Below is the full list of concepts across all ingested papers, with their current per-paper
importance scores (1–10). Your task is to re-score each concept on a GLOBAL scale where:
- 10 = landmark idea that defines or redirects the field
- 7–9 = significant contribution, widely applicable insight
- 4–6 = solid but narrow contribution, useful in specific contexts
- 1–3 = minor detail, implementation choice, or well-known baseline

Criteria for global scoring:
- Cross-paper impact: does this concept appear or get built on across multiple papers?
- Field significance: does it represent a step-change or just an incremental refinement?
- Practical utility: would a practitioner actually use or cite this?

CONCEPTS (format: id | paper | concept_title | concept_type | current_score):
{concepts_text}

Return ONLY a valid JSON array with no extra text. Include ALL concepts:
[
  {{"id": "concept-uuid", "importance": 7}},
  ...
]"""


def answer_from_concepts(question: str, context: str) -> str:
    return f"""You are a research assistant with access to a curated memory database of
research paper concepts. Answer the question below using ONLY the retrieved concept
understandings provided. Do not add information from outside this context.

For each key claim, cite the source paper using its title.

RETRIEVED CONCEPTS:
{context}

QUESTION: {question}

Provide a thorough, well-structured answer:"""


def concept_extraction(title: str, text: str, max_chars: int, existing_types: list[str] | None = None) -> str:
    types_guidance = ""
    if existing_types:
        types_list = ", ".join(f'"{t}"' for t in existing_types)
        types_guidance = f"""
EXISTING CONCEPT TYPES in the database: [{types_list}]
Reuse these types where they fit. You may introduce a new type only if none of the existing
ones capture the concept — keep the overall type vocabulary small and consistent (aim for
10 or fewer distinct types across all papers). Merge similar ideas rather than creating
near-duplicate types (e.g. don't add "retrieval mechanism" if "retrieval" already exists).
"""
    else:
        types_guidance = """
Since this is the first paper being ingested, establish an initial set of concept types.
Aim for broad, reusable categories (e.g. "architecture", "retrieval", "memory management",
"evaluation", "limitation", "training method", "application"). Keep the list small — these
types will be reused across all future papers.
"""

    return f"""You are processing a research paper titled: "{title}"

Below is the full paper text (truncated if very long):
---
{text[:max_chars]}
---

## Step 1 — Paper Summary
Write a 3–5 sentence abstract-style summary of the paper's core contribution, key mechanism,
and main finding. This will be stored as the paper's top-level summary.

## Step 2 — Concept Extraction
Identify the 8–15 most important core concepts in this paper.

For each concept write a rich semantic understanding — a dense paragraph that captures:
- What the concept is
- Why it matters in the context of this paper
- How it relates to the paper's main contribution
- Key technical details, mechanisms, or nuances a researcher would want to know

Also assign:
- concept_type: a short category label (see guidance below)
- importance: integer 1–10 reflecting how central this concept is to the paper's contribution
  (10 = defines the paper's main idea; 1 = minor implementation detail)
- section: the paper section this concept primarily comes from
  (one of: "abstract", "introduction", "related_work", "methods", "results", "discussion", "conclusion")
{types_guidance}
These understandings will be stored in a semantic memory database and searched later
to answer questions about this paper WITHOUT re-reading it. Make them thorough and precise.

Return ONLY a valid JSON object with no extra text:
{{
  "summary": "3–5 sentence paper summary...",
  "concepts": [
    {{
      "concept_title": "Short descriptive title (5-10 words)",
      "understanding": "Dense semantic paragraph (100-200 words)...",
      "concept_type": "architecture",
      "importance": 9,
      "section": "methods"
    }},
    ...
  ]
}}"""
