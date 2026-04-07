"""
Seed script — downloads the 20 research papers from arxiv and ingests them
into the semantic memory database.

Usage:
    python ingest.py                          # ingest all papers
    python ingest.py --skip N                 # skip first N papers (resume after failure)
    python ingest.py --url URL --title TITLE  # ingest a single web article
"""

import argparse
import os
import time

import requests
from dotenv import load_dotenv

# Must load .env before importing db — db.py reads DATABASE_URL at import time
load_dotenv()

from db import document_exists, init_db
from memory import ingest_pdf, ingest_url

PAPERS = [
    {
        "title": "MemGPT: Towards LLMs as Operating Systems",
        "arxiv_id": "2310.08560",
    },
    {
        "title": "Generative Agents: Interactive Simulacra of Human Behavior",
        "arxiv_id": "2304.03442",
    },
    {
        "title": "Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory",
        "arxiv_id": "2311.08719",
    },
    {
        "title": "ReMeMBer: A Robust Memory Module for Enhancing LLMs with External Information",
        "arxiv_id": "2404.10774",
    },
    {
        "title": "A-MEM: Agentic Memory for LLM Agents",
        "arxiv_id": "2502.12110",
    },
    {
        "title": "MemoRAG: Moving towards Next-Gen RAG via Memory-Inspired Knowledge Discovery",
        "arxiv_id": "2409.05591",
    },
    {
        "title": "Cognitive Architectures for Language Agents (CoALA)",
        "arxiv_id": "2309.02427",
    },
    {
        "title": "MemoryBank: Enhancing Large Language Models with Long-Term Memory",
        "arxiv_id": "2305.10250",
    },
    {
        "title": "RecurrentGPT: Interactive Generation of Arbitrarily Long Text",
        "arxiv_id": "2305.13304",
    },
    {
        "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "arxiv_id": "2303.11366",
    },
    {
        "title": "SCM: A Self-Controlled Memory System for LLMs",
        "arxiv_id": "2304.13343",
    },
    {
        "title": "Recurrent Memory Transformer",
        "arxiv_id": "2207.06881",
    },
    {
        "title": "LongMem: Enabling LLMs to Memorize Long-term Dependencies",
        "arxiv_id": "2306.07174",
    },
    {
        "title": "SPRING: Studying Papers and Reasoning to Play Games",
        "arxiv_id": "2305.15486",
    },
    {
        "title": "ExpeL: LLM Agents Are Experiential Learners",
        "arxiv_id": "2308.10144",
    },
    {
        "title": "Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-term Conversations",
        "arxiv_id": "2402.11975",
    },
    {
        "title": "LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration",
        "arxiv_id": "2402.11550",
    },
    {
        "title": "HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models",
        "arxiv_id": "2405.14831",
    },
    {
        "title": "Larimar: Large Language Models with Episodic Memory Control",
        "arxiv_id": "2403.11901",
    },
    {
        "title": "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models",
        "arxiv_id": "2308.15022",
    },
    {
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "arxiv_id": "2404.16130",
    },
    {
        "title": "MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents",
        "arxiv_id": "2601.03236",
    },
    {
        "title": "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents",
        "arxiv_id": "2407.04363",
    },
    {
        "title": "Zep: A Temporal Knowledge Graph Architecture for Agent Memory",
        "arxiv_id": "2501.13956",
    },
    {
        "title": "Graph-based Agent Memory: Taxonomy, Techniques, and Applications",
        "arxiv_id": "2602.05665",
    },
]


def download_pdf(arxiv_id, output_path):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  Downloading {url}")
    headers = {"User-Agent": "Mozilla/5.0 (research/memory-chat)"}
    r = requests.get(url, stream=True, headers=headers, timeout=60)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  Saved {size_kb} KB → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0, help="Skip first N papers")
    parser.add_argument("--limit", type=int, default=None, help="Ingest at most N papers")
    parser.add_argument("--url", type=str, default=None, help="Ingest a single web article by URL")
    parser.add_argument("--title", type=str, default=None, help="Title for the web article (required with --url)")
    args = parser.parse_args()

    init_db()

    # Web article mode
    if args.url:
        if not args.title:
            parser.error("--title is required when using --url")
        if document_exists(args.url):
            print(f"Already ingested — skipping: {args.url}")
            return
        ingest_url(args.url, args.title)
        return

    os.makedirs("papers", exist_ok=True)

    papers = PAPERS[args.skip:]
    if args.limit is not None:
        papers = papers[:args.limit]
    print(f"Ingesting {len(papers)} papers (skipping first {args.skip})...\n")

    for i, paper in enumerate(papers, start=args.skip + 1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        pdf_path = f"papers/{arxiv_id}.pdf"
        source_url = f"https://arxiv.org/abs/{arxiv_id}"

        print(f"[{i}/{len(PAPERS)}] {title}")

        if document_exists(source_url):
            print("  Already ingested — skipping.\n")
            continue

        if not os.path.exists(pdf_path):
            try:
                download_pdf(arxiv_id, pdf_path)
            except Exception as e:
                print(f"  ERROR downloading: {e} — skipping.\n")
                continue
            time.sleep(3)  # polite delay between arxiv requests

        try:
            ingest_pdf(pdf_path, title, source_url)
        except Exception as e:
            print(f"  ERROR during ingestion: {e}\n")
            continue

        print()

    print("All done.")


if __name__ == "__main__":
    main()
