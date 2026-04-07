"""
MCP server exposing the repo-memory-chat database as tools for Claude.

Tools:
  search_memory(query)  — search concept understandings
  list_papers()         — list all ingested research papers
"""

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from db import init_db, list_documents
from search import run_search

init_db()
mcp = FastMCP("repo-memory")


@mcp.tool()
def search_memory(query: str) -> str:
    """
    Search the LLM memory research database for concepts matching the query.
    Uses hybrid keyword + vector search with LLM query understanding.
    Returns relevant concept understandings with source paper citations.
    """
    results, keywords, _ = run_search(query)
    if not results:
        return f"No concepts found for query: {query}"

    lines = [f"Found {len(results)} concept(s) for '{query}' (keywords: {', '.join(keywords)}):\n"]
    for r in results:
        lines.append(f"## {r.concept_title}")
        lines.append(f"**Source:** [{r.paper_title}]({r.source_url})")
        if r.concept_type:
            lines.append(f"**Type:** {r.concept_type} | **Importance:** {r.importance}/10")
        lines.append(f"{r.understanding}\n")
    return "\n".join(lines)


@mcp.tool()
def list_papers() -> str:
    """List all research papers ingested into the memory database."""
    rows = list_documents()
    if not rows:
        return "No papers ingested yet."
    lines = [f"{len(rows)} papers in memory:\n"]
    for _, title, _, processed_at in rows:
        date = processed_at[:10] if processed_at else "unknown"
        lines.append(f"- {title} ({date})")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
