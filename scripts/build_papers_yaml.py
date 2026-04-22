"""
One-off utility: build /papers.yaml from

  1. /tmp/claude/corpus/merged.yaml  — VERIFIED section (65 entries expected)
  2. the legacy PAPERS list in the old top-level ingest.py (25 arxiv seeds)

Output schema per spec §3.1:
    title: str
    source_type: arxiv | url
    source_id_or_url: str
    authors: str        (optional)
    year: int           (optional)
    tier: str           (optional: core | strong | supplementary)
    rationale: str      (optional)

Run:
    python scripts/build_papers_yaml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGED_YAML = Path("/tmp/claude/corpus/merged.yaml")
OUTPUT = REPO_ROOT / "papers.yaml"

# Legacy seeds — copied verbatim from the old ingest.py PAPERS list (25 entries).
# All arxiv IDs; no authors/year/tier/rationale in the old schema.
LEGACY_SEEDS: list[dict] = [
    {"title": "MemGPT: Towards LLMs as Operating Systems", "arxiv_id": "2310.08560"},
    {"title": "Generative Agents: Interactive Simulacra of Human Behavior", "arxiv_id": "2304.03442"},
    {"title": "Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory", "arxiv_id": "2311.08719"},
    {"title": "ReMeMBer: A Robust Memory Module for Enhancing LLMs with External Information", "arxiv_id": "2404.10774"},
    {"title": "A-MEM: Agentic Memory for LLM Agents", "arxiv_id": "2502.12110"},
    {"title": "MemoRAG: Moving towards Next-Gen RAG via Memory-Inspired Knowledge Discovery", "arxiv_id": "2409.05591"},
    {"title": "Cognitive Architectures for Language Agents (CoALA)", "arxiv_id": "2309.02427"},
    {"title": "MemoryBank: Enhancing Large Language Models with Long-Term Memory", "arxiv_id": "2305.10250"},
    {"title": "RecurrentGPT: Interactive Generation of Arbitrarily Long Text", "arxiv_id": "2305.13304"},
    {"title": "Reflexion: Language Agents with Verbal Reinforcement Learning", "arxiv_id": "2303.11366"},
    {"title": "SCM: A Self-Controlled Memory System for LLMs", "arxiv_id": "2304.13343"},
    {"title": "Recurrent Memory Transformer", "arxiv_id": "2207.06881"},
    {"title": "LongMem: Enabling LLMs to Memorize Long-term Dependencies", "arxiv_id": "2306.07174"},
    {"title": "SPRING: Studying Papers and Reasoning to Play Games", "arxiv_id": "2305.15486"},
    {"title": "ExpeL: LLM Agents Are Experiential Learners", "arxiv_id": "2308.10144"},
    {"title": "Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-term Conversations", "arxiv_id": "2402.11975"},
    {"title": "LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration", "arxiv_id": "2402.11550"},
    {"title": "HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models", "arxiv_id": "2405.14831"},
    {"title": "Larimar: Large Language Models with Episodic Memory Control", "arxiv_id": "2403.11901"},
    {"title": "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models", "arxiv_id": "2308.15022"},
    {"title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", "arxiv_id": "2404.16130"},
    {"title": "MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents", "arxiv_id": "2601.03236"},
    {"title": "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents", "arxiv_id": "2407.04363"},
    {"title": "Zep: A Temporal Knowledge Graph Architecture for Agent Memory", "arxiv_id": "2501.13956"},
    {"title": "Graph-based Agent Memory: Taxonomy, Techniques, and Applications", "arxiv_id": "2602.05665"},
]


def load_verified_entries(merged_path: Path) -> list[dict]:
    """Parse merged.yaml and return only the VERIFIED prefix (everything before the UNVERIFIED header)."""
    if not merged_path.exists():
        raise FileNotFoundError(f"merged corpus not found: {merged_path}")
    text = merged_path.read_text()
    marker = "# UNVERIFIED"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"UNVERIFIED section marker not found in {merged_path}")
    verified_text = text[:idx]
    entries = yaml.safe_load(verified_text) or []
    return entries


def normalize_verified(entry: dict) -> dict:
    """Keep only schema-allowed keys; drop `confidence`, `sources` (provenance-only)."""
    allowed = {"title", "authors", "year", "source_type", "source_id_or_url", "tier", "rationale"}
    out = {k: v for k, v in entry.items() if k in allowed}
    # sanity — we expect every verified entry to have these three at minimum
    missing = [k for k in ("title", "source_type", "source_id_or_url") if not out.get(k)]
    if missing:
        raise ValueError(f"verified entry missing required keys {missing}: {entry!r}")
    return out


def normalize_legacy(seed: dict) -> dict:
    return {
        "title": seed["title"],
        "source_type": "arxiv",
        "source_id_or_url": seed["arxiv_id"],
        "tier": "core",
        "rationale": "Legacy seed from repo-memory-chat v0 ingest.py PAPERS list.",
    }


def main() -> int:
    verified = load_verified_entries(MERGED_YAML)
    verified_count = len(verified)

    # Schema-normalize each verified entry.
    verified_norm = [normalize_verified(e) for e in verified]

    # Legacy seeds go first (preserves prior ordering from ingest.py), then verified entries.
    legacy_norm = [normalize_legacy(s) for s in LEGACY_SEEDS]

    # Dedup on (source_type, source_id_or_url).
    seen: set[tuple[str, str]] = set()
    combined: list[dict] = []
    dropped_dupes = 0
    for entry in legacy_norm + verified_norm:
        key = (entry["source_type"], str(entry["source_id_or_url"]).strip())
        if key in seen:
            dropped_dupes += 1
            print(f"  dedup: dropping duplicate {key[0]} {key[1]} ({entry['title']!r})")
            continue
        seen.add(key)
        combined.append(entry)

    OUTPUT.write_text(
        "# papers.yaml — corpus manifest for LTM expert ingest\n"
        f"# {len(legacy_norm)} legacy seeds + {verified_count} verified entries "
        f"- {dropped_dupes} dedups = {len(combined)} total\n\n"
        + yaml.safe_dump(combined, sort_keys=False, allow_unicode=True, width=120)
    )

    print(f"Wrote {OUTPUT}")
    print(f"  legacy seeds: {len(legacy_norm)}")
    print(f"  verified entries (from merged.yaml): {verified_count}")
    print(f"  duplicates dropped: {dropped_dupes}")
    print(f"  total: {len(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
