import os

import streamlit as st
from dotenv import load_dotenv

# 1. Load .env for local development (no-op on Streamlit Cloud)
load_dotenv()

# 2. Inject Streamlit secrets into env so db.py and anthropic pick them up.
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ.setdefault(_key, _val)
except Exception:
    pass

# 3. Import project modules AFTER env is populated
from groq import Groq
from db import concept_count, init_db, list_documents
import prompts
from search import run_search

client = Groq()
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def answer_with_context(question: str, concepts) -> str:
    """Generate an answer grounded only in the retrieved concept understandings."""
    if not concepts:
        return (
            "No relevant concepts were found in the memory database for your question. "
            "Try rephrasing, or make sure documents have been ingested by running `python ingest.py`."
        )

    context_blocks = []
    for i, concept in enumerate(concepts, 1):
        context_blocks.append(
            f"[{i}] Concept: {concept.concept_title}\n"
            f"    Type: {concept.concept_type or '?'}  |  "
            f"Importance: {concept.importance or '?'}  |  "
            f"Score: {concept.final_score:.2f}\n"
            f"    Source: {concept.paper_title} ({concept.source_url})\n"
            f"    Understanding: {concept.understanding}"
        )
    context = "\n\n".join(context_blocks)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "user", "content": prompts.answer_from_concepts(question, context)}
        ],
    )
    return response.choices[0].message.content


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

    import db as _db
    if _db.DATABASE_URL:
        st.success("DB: PostgreSQL ✓")
    else:
        st.error("DB: SQLite (DATABASE_URL not set!)")

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

        # Step 1: understand query + run both searches + score
        status.update(label="Understanding query and scanning memory...")
        concepts, keywords, semantic_answer = run_search(prompt)

        # Step 2: generate answer
        status.update(label=f"Found {len(concepts)} concept(s) — generating answer...")
        answer = answer_with_context(prompt, concepts)

        state = "complete" if concepts else "error"
        status.update(label=f"Done — {len(concepts)} concept(s) used", state=state)

        st.markdown(answer)

        # Build metadata string for expandable details
        meta_lines = [
            f"**Keywords searched:** {', '.join(f'`{k}`' for k in keywords)}",
            f"**Semantic query:** _{semantic_answer[:120]}..._" if len(semantic_answer) > 120 else f"**Semantic query:** _{semantic_answer}_",
            "",
        ]
        if concepts:
            meta_lines.append(f"**{len(concepts)} concept(s) retrieved:**")
            for concept in concepts:
                keyword_tag = "🔑" if concept.keyword_hit else "〰"
                meta_lines.append(
                    f"- {keyword_tag} [{concept.concept_title}]({concept.source_url}) "
                    f"— *{concept.paper_title}* "
                    f"| score={concept.final_score:.2f} importance={concept.importance or '?'}"
                )
        else:
            meta_lines.append("*No matching concepts found.*")
        meta = "\n".join(meta_lines)

        with st.expander("Memory scan details"):
            st.markdown(meta)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "meta": meta}
        )
