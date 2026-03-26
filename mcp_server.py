"""
MCP server exposing the repo-memory-chat database as tools for Claude.

Tools:
  search_memory(query)  — search concept understandings
  list_papers()         — list all ingested research papers
"""

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from db import concept_count, init_db, list_documents, search_concepts

init_db()
mcp = FastMCP("repo-memory")


@mcp.tool()
def search_memory(query: str) -> str:
    """
    Search the LLM memory research database for concepts matching the query.
    Returns relevant concept understandings with source paper citations.
    """
    keywords = [kw for kw in query.lower().split() if len(kw) > 2]
    if not keywords:
        return "Query too short — please provide meaningful search terms."

    rows = search_concepts(keywords)
    if not rows:
        return f"No concepts found for query: {query}"

    lines = [f"Found {len(rows)} concept(s) for '{query}':\n"]
    for concept_title, understanding, doc_title, source_url in rows:
        lines.append(f"## {concept_title}")
        lines.append(f"**Source:** [{doc_title}]({source_url})")
        lines.append(f"{understanding}\n")
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
