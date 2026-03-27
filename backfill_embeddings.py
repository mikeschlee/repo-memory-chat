"""
One-time backfill script — generates and stores embeddings for any concepts
that were ingested before vector search was added.

Usage:
    python backfill_embeddings.py           # preview count only
    python backfill_embeddings.py --apply   # generate and write embeddings

Processes in batches of 128 to stay within Voyage AI rate limits.
Safe to re-run — skips concepts that already have embeddings.
"""

import argparse
import os
import time

from dotenv import load_dotenv

load_dotenv()

# Inject Streamlit secrets into env if present
try:
    import streamlit as st
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ.setdefault(_key, _val)
except Exception:
    pass

from db import get_concepts_without_embeddings, update_concept_embedding, init_db
from embeddings import embed_texts

BATCH_SIZE = 128  # Voyage AI batch limit
RATE_LIMIT_DELAY = 0.5  # seconds between batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write embeddings to DB (default: dry run)")
    args = parser.parse_args()

    init_db()
    concepts = get_concepts_without_embeddings()
    print(f"{len(concepts)} concepts without embeddings.")

    if not concepts:
        print("Nothing to backfill.")
        return

    if not args.apply:
        print("Dry run — use --apply to generate and store embeddings.")
        return

    total = len(concepts)
    stored = 0

    for i in range(0, total, BATCH_SIZE):
        batch = concepts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} concepts)...", end=" ", flush=True)

        texts = [f"{c[1]}\n\n{c[2]}" for c in batch]  # title + understanding
        embeddings = embed_texts(texts, input_type="document")

        for (concept_id, _, _), embedding in zip(batch, embeddings):
            update_concept_embedding(concept_id, embedding)
            stored += 1

        print(f"done. ({stored}/{total} total)")

        if i + BATCH_SIZE < total:
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nBackfill complete — {stored} embeddings stored.")


if __name__ == "__main__":
    main()
