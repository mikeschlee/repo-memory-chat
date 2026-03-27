# Code Review Guidelines

## Always check
- Database queries use parameterized statements (no f-string SQL)
- API keys and secrets are read from environment variables, never hardcoded
- PDF ingestion handles corrupt or empty files without crashing
- Semantic search results are ranked and returned correctly
- Memory entries are deduplicated before insertion

## Skip
- Formatting, whitespace, or import ordering (handled by linters)
- Cosmetic Streamlit UI changes with no logic impact
- Changes to `memory.db` or `papers/` (generated/data files)

## Nit-level (flag but not blocking)
- Missing type hints on new functions
- Overly broad `except Exception` clauses
