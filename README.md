# Repo Memory Chat

A chat interface for querying research papers using LLM-generated semantic memory — no RAG, no vector embeddings. Seeded with 20 arxiv papers on LLM memory research (303 concepts).

## How it works

1. **Ingest**: Claude reads each PDF, extracts 8–15 core concepts, and writes a dense semantic understanding for each one. These understandings are stored in SQLite.
2. **Query**: When a user asks a question, Claude extracts search keywords from it. Those keywords are matched against the stored understandings using substring search. Matching understandings are loaded into context and Claude answers from them — never from the raw documents.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Seed the database

Downloads and ingests 20 research papers on LLM memory approaches:

```bash
python ingest.py
```

Takes ~20–40 minutes. Papers download from arxiv into `papers/` (gitignored). Resume after failure with `--skip N`.

## Run the app

```bash
streamlit run app.py
```

## Third-party providers

| Provider | Purpose | Where configured |
|---|---|---|
| [Anthropic](https://www.anthropic.com) | Claude API (`claude-opus-4-6`) — keyword extraction + answering | `ANTHROPIC_API_KEY` env var |
| [Streamlit Cloud](https://streamlit.io/cloud) | Web UI framework + public hosting/deployment | `.streamlit/secrets.toml` on the dashboard |
| [Supabase](https://supabase.com) | Managed PostgreSQL database (production) | `DATABASE_URL` env var (falls back to SQLite locally) |
| [arXiv](https://arxiv.org) | Source for research paper PDFs ingested into the database | Hardcoded URLs in `ingest.py` |

## Papers included

20 research papers on LLM memory approaches beyond RAG, including MemGPT, Generative Agents, Think-in-Memory, A-MEM, MemoRAG, CoALA, MemoryBank, Reflexion, HippoRAG, Larimar, and more.
