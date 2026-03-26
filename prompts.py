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


def answer_from_concepts(question: str, context: str) -> str:
    return f"""You are a research assistant with access to a curated memory database of
research paper concepts. Answer the question below using ONLY the retrieved concept
understandings provided. Do not add information from outside this context.

For each key claim, cite the source paper using its title.

RETRIEVED CONCEPTS:
{context}

QUESTION: {question}

Provide a thorough, well-structured answer:"""


def concept_extraction(title: str, text: str, max_chars: int) -> str:
    return f"""You are processing a research paper titled: "{title}"

Below is the full paper text (truncated if very long):
---
{text[:max_chars]}
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
