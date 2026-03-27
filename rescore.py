"""
Global importance re-scoring script.

Loads all concepts from the database and asks Claude to re-score them on a
global scale (1–10) relative to each other across all papers.

Run this manually when:
  - You've added a significant batch of new papers
  - A major new approach has shifted what counts as "important"
  - Per-paper scores feel inconsistent across the corpus

Usage:
    python rescore.py           # preview diff only (dry run)
    python rescore.py --apply   # write updated scores to the database
"""

import argparse
import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

from db import get_all_concepts_for_rescore, update_concept_importance
import prompts

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"  # ranking task — Haiku is sufficient

# Max concepts per scoring batch to stay within context limits
BATCH_SIZE = 150


def build_concepts_text(concepts):
    lines = []
    for concept_id, title, understanding, concept_type, importance, paper in concepts:
        short = understanding[:120].replace("\n", " ")
        lines.append(f"{concept_id} | {paper} | {title} | {concept_type or '?'} | {importance or '?'}")
    return "\n".join(lines)


def rescore_batch(concepts):
    """Send a batch to Claude and return {concept_id: new_importance} mapping."""
    concepts_text = build_concepts_text(concepts)
    prompt = prompts.global_rescore(concepts_text)

    last_error = None
    for attempt in range(1, 4):
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
            results = json.loads(raw[start:end])
            return {r["id"]: r["importance"] for r in results}
        except (json.JSONDecodeError, KeyError) as e:
            last_error = e
            print(f"  Attempt {attempt} failed: {e} — retrying...")

    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write scores to DB (default: dry run)")
    args = parser.parse_args()

    print("Loading all concepts...")
    concepts = get_all_concepts_for_rescore()
    print(f"  {len(concepts)} concepts loaded across all papers\n")

    if not concepts:
        print("No concepts found. Run ingest.py first.")
        return

    # Process in batches
    all_new_scores = {}
    for i in range(0, len(concepts), BATCH_SIZE):
        batch = concepts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(concepts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Scoring batch {batch_num}/{total_batches} ({len(batch)} concepts)...")
        new_scores = rescore_batch(batch)
        all_new_scores.update(new_scores)
        print(f"  Done.")

    # Build diff
    concept_map = {c[0]: c for c in concepts}
    changes = []
    for concept_id, new_score in all_new_scores.items():
        if concept_id not in concept_map:
            continue
        _, title, _, concept_type, old_score, paper = concept_map[concept_id]
        delta = (new_score - old_score) if old_score is not None else None
        changes.append((concept_id, paper, title, old_score, new_score, delta))

    # Sort by magnitude of change
    changes.sort(key=lambda x: abs(x[5] or 0), reverse=True)

    print(f"\n{'─'*70}")
    print(f"RESCORE DIFF — {len(changes)} concepts")
    print(f"{'─'*70}")

    significant = [c for c in changes if c[5] is not None and abs(c[5]) >= 2]
    unchanged = [c for c in changes if c[5] is not None and abs(c[5]) < 2]
    new_entries = [c for c in changes if c[5] is None]

    if significant:
        print(f"\nSignificant changes (±2 or more):")
        for concept_id, paper, title, old, new, delta in significant:
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {arrow}{abs(delta):+d}  [{old}→{new}]  {title}  ({paper})")

    if new_entries:
        print(f"\nNew scores (no previous score):")
        for concept_id, paper, title, old, new, delta in new_entries:
            print(f"  [{new}]  {title}  ({paper})")

    print(f"\n{len(unchanged)} concepts unchanged (delta < 2).")
    print(f"{'─'*70}")

    if args.apply:
        print("\nWriting scores to database...")
        for concept_id, new_score in all_new_scores.items():
            update_concept_importance(concept_id, new_score)
        print(f"  {len(all_new_scores)} concepts updated.")
    else:
        print("\nDry run — no changes written. Use --apply to commit scores.")


if __name__ == "__main__":
    main()
