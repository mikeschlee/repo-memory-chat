"""
Search orchestration — runs keyword and vector searches in parallel,
merges results, and scores them with a deterministic weighted formula.

Scoring formula:
    final_score = WEIGHT_VECTOR   * vector_similarity   (0–1)
                + WEIGHT_KEYWORD  * keyword_hit          (0 or 1)
                + WEIGHT_IMPORTANCE * normalized_importance (0–1)

Weights are tunable constants. Higher vector weight = trust semantic match more.
Higher keyword weight = ensure exact-term matches don't get buried.
"""

import json
import os
from dataclasses import dataclass, field

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ.setdefault(_key, _val)
except Exception:
    pass

from db import search_concepts_by_keyword, search_concepts_by_vector
from embeddings import embed_query
import prompts

# ---------------------------------------------------------------------------
# Scoring weights — adjust to tune retrieval behaviour
# ---------------------------------------------------------------------------
WEIGHT_VECTOR = 0.5
WEIGHT_KEYWORD = 0.3
WEIGHT_IMPORTANCE = 0.2

# Max results returned to the answer LLM
TOP_K_FINAL = 12
# Max results fetched from each search before merging
TOP_K_VECTOR = 30
MAX_PER_PAPER = 3  # diversity cap: at most N concepts per paper in final results

groq_client = Groq()
QUERY_MODEL = "llama-3.3-70b-versatile"


@dataclass
class ConceptResult:
    concept_id: str
    concept_title: str
    understanding: str
    concept_type: str | None
    importance: int | None
    section: str | None
    paper_title: str
    source_url: str
    vector_similarity: float = 0.0
    keyword_hit: bool = False

    @property
    def final_score(self) -> float:
        norm_importance = (self.importance or 5) / 10.0
        return (
            WEIGHT_VECTOR * self.vector_similarity
            + WEIGHT_KEYWORD * (1.0 if self.keyword_hit else 0.0)
            + WEIGHT_IMPORTANCE * norm_importance
        )


def understand_query(question: str) -> tuple[list[str], str]:
    """
    Ask Claude to extract keywords and generate a semantic answer hypothesis.
    Returns (keywords, semantic_answer).
    Falls back to treating the question as a single keyword if parsing fails.
    """
    prompt = prompts.query_understanding(question)
    response = groq_client.chat.completions.create(
        model=QUERY_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            parsed = json.loads(raw[start:end])
            keywords = parsed.get("keywords", [])
            semantic_answer = parsed.get("semantic_answer", question)
            return keywords, semantic_answer
        except json.JSONDecodeError:
            pass
    # Fallback
    return [question], question


def run_search(question: str) -> tuple[list[ConceptResult], list[str], str]:
    """
    Full search pipeline:
      1. Understand query → keywords + semantic answer
      2. Run keyword search + vector search in parallel (sequential here, fast enough)
      3. Merge, score, diversity-cap, return top results

    Returns (results, keywords, semantic_answer) for display in the UI.
    """
    keywords, semantic_answer = understand_query(question)

    # --- Keyword search ---
    keyword_rows = search_concepts_by_keyword(keywords)
    # rows: (id, concept_title, understanding, concept_type, importance, section, paper_title, source_url)

    # --- Vector search ---
    query_embedding = embed_query(semantic_answer)
    vector_rows = search_concepts_by_vector(query_embedding, top_k=TOP_K_VECTOR)
    # rows: (id, concept_title, understanding, concept_type, importance, section, paper_title, source_url, similarity)

    # --- Build concept map ---
    concepts: dict[str, ConceptResult] = {}

    for row in keyword_rows:
        cid = row[0]
        if cid not in concepts:
            concepts[cid] = ConceptResult(
                concept_id=cid,
                concept_title=row[1],
                understanding=row[2],
                concept_type=row[3],
                importance=row[4],
                section=row[5],
                paper_title=row[6],
                source_url=row[7],
            )
        concepts[cid].keyword_hit = True

    for row in vector_rows:
        cid = row[0]
        similarity = row[8]
        if cid not in concepts:
            concepts[cid] = ConceptResult(
                concept_id=cid,
                concept_title=row[1],
                understanding=row[2],
                concept_type=row[3],
                importance=row[4],
                section=row[5],
                paper_title=row[6],
                source_url=row[7],
            )
        concepts[cid].vector_similarity = max(concepts[cid].vector_similarity, similarity)

    # --- Score and rank ---
    ranked = sorted(concepts.values(), key=lambda c: c.final_score, reverse=True)

    # --- Diversity cap: max MAX_PER_PAPER concepts per paper ---
    paper_counts: dict[str, int] = {}
    final: list[ConceptResult] = []
    for concept in ranked:
        count = paper_counts.get(concept.paper_title, 0)
        if count < MAX_PER_PAPER:
            final.append(concept)
            paper_counts[concept.paper_title] = count + 1
        if len(final) >= TOP_K_FINAL:
            break

    return final, keywords, semantic_answer
