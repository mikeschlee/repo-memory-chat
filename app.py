import json
import os

import streamlit as st
from dotenv import load_dotenv

# 1. Load .env for local development (no-op on Streamlit Cloud)
load_dotenv()

# 2. Inject Streamlit secrets into env so db.py and anthropic pick them up.
#    On Streamlit Cloud, secrets are set in the dashboard and available via
#    st.secrets. Locally, .env covers it. setdefault means .env wins locally.
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ.setdefault(_key, _val)
except Exception:
    pass

# 3. Import project modules AFTER env is populated (db.py reads DATABASE_URL
#    at import time to choose SQLite vs PostgreSQL backend)
import anthropic
from db import concept_count, init_db, list_documents, search_concepts

client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def extract_keywords(question: str) -> list[str]:
    """Ask Claude to derive search keywords from the user's question."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"""You are a search query analyst.

Given the user question below, extract 6–12 keywords or short phrases that would
best match relevant concepts stored in a research paper memory database.

Focus on:
- Technical terms and named methods/architectures
- Core concepts the question is about
- Synonyms or related terms that might appear in paper descriptions

Return ONLY a valid JSON array of strings with no extra text.
Example: ["memory augmented", "episodic memory", "retrieval", "long context"]

Question: {question}""",
            }
        ],
    )
    raw = response.content[0].text
    start = raw.find("[")
    end = raw.rfind("]") + 1
    return json.loads(raw[start:end])


def answer_with_context(question: str, concepts: list) -> str:
    """Generate an answer grounded only in the retrieved concept understandings."""
    if not concepts:
        return (
            "No relevant concepts were found in the memory database for your question. "
            "Try rephrasing, or make sure documents have been ingested by running `python ingest.py`."
        )

    context_blocks = []
    for i, (concept_title, understanding, doc_title, source_url) in enumerate(concepts, 1):
        context_blocks.append(
            f"[{i}] Concept: {concept_title}\n"
            f"    Source: {doc_title} ({source_url})\n"
            f"    Understanding: {understanding}"
        )
    context = "\n\n".join(context_blocks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""You are a research assistant with access to a curated memory database of
research paper concepts. Answer the question below using ONLY the retrieved concept
understandings provided. Do not add information from outside this context.

For each key claim, cite the source paper using its title.

RETRIEVED CONCEPTS:
{context}

QUESTION: {question}

Provide a thorough, well-structured answer:""",
            }
        ],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Repo Memory Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Repo Memory Chat")
    st.caption("Semantic memory search over research papers")
    st.divider()

    docs = list_documents()
    n_concepts = concept_count()

    st.metric("Papers loaded", len(docs))
    st.metric("Concepts in memory", n_concepts)
    st.divider()

    if docs:
        st.subheader("Loaded Papers")
        for _, title, _, processed_at in docs:
            date = processed_at[:10] if processed_at else ""
            st.markdown(f"- **{title}**  \n  <small>{date}</small>", unsafe_allow_html=True)
    else:
        st.warning("No papers ingested yet.\n\nRun:\n```\npython ingest.py\n```")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Main chat area ---------------------------------------------------------
st.header("Ask about LLM memory research")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Memory scan details"):
                st.markdown(msg["meta"])

# Input
if prompt := st.chat_input("e.g. How do generative agents manage long-term memory?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Scanning memory...", expanded=False)

        # Step 1: keyword extraction
        status.update(label="Extracting search keywords...")
        keywords = extract_keywords(prompt)

        # Step 2: search concept DB
        status.update(label=f"Searching memory for: {', '.join(keywords[:5])}...")
        concepts = search_concepts(keywords)

        # Step 3: generate answer
        status.update(label=f"Found {len(concepts)} concept(s) — generating answer...")
        answer = answer_with_context(prompt, concepts)

        status.update(label=f"Done — {len(concepts)} concept(s) used", state="complete")

        st.markdown(answer)

        # Build metadata string for expandable details
        meta_lines = [f"**Keywords searched:** {', '.join(f'`{k}`' for k in keywords)}", ""]
        if concepts:
            meta_lines.append(f"**{len(concepts)} concept(s) retrieved:**")
            for concept_title, _, doc_title, source_url in concepts:
                meta_lines.append(f"- [{concept_title}]({source_url}) — *{doc_title}*")
        else:
            meta_lines.append("*No matching concepts found.*")
        meta = "\n".join(meta_lines)

        with st.expander("Memory scan details"):
            st.markdown(meta)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "meta": meta}
        )
