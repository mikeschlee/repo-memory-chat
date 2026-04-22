"""
Centralized configuration for the LTM expert rebuild.

Single source of truth for storage + retrieval tunables. Mirrors the shape of
ltm-benchmarking's `core/config.py` with eval/rescore keys stripped.

Do NOT import anything from db/, retrieval/, or pipeline/ here.
"""

# ── Pipeline defaults ──────────────────────────────────────────────────────────
INGEST_LIMIT: int = 0            # 0 = no limit
EXTRACT_WORKERS: int = 1         # parallel workers for concept extraction

# Extraction parameters
MAX_TEXT_CHARS: int = 80_000     # truncate raw document text at this length
RAG_CHUNK_SIZE: int = 500        # character width of RAG overlap chunks
RAG_CHUNK_OVERLAP: int = 100     # character overlap between adjacent RAG chunks

# ── Concept Clustering (UMAP + HDBSCAN) ───────────────────────────────────────
UMAP_N_COMPONENTS: int = 12
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.0
HDBSCAN_MIN_CLUSTER_SIZE: int = 7
HDBSCAN_MIN_SAMPLES: int = 2

# ── Storage pipeline models (wired by Agent B via env vars) ────────────────────
# Env-var stubs per spec §9 — the names below are the defaults; callers should
# read os.environ overrides in core/llm.py (Agent B).
INGEST_MODEL:  str = "openrouter/qwen/qwen3-235b-a22b-2507"   # Step 2 concept extraction
CLUSTER_MODEL: str = "openrouter/qwen/qwen3-32b"              # Step 3 meta-concept + triples
DEDUP_MODEL:   str = "openrouter/qwen/qwen3-32b"              # Steps 4/5 entity + predicate dedup

# ── Retrieval pipeline models ──────────────────────────────────────────────────
SEARCH_MODEL:  str = "openrouter/meta-llama/llama-3.3-70b-instruct"  # query understanding
ANSWER_MODEL:  str = "openrouter/meta-llama/llama-3.3-70b-instruct"  # answer generation

# Embeddings
EMBEDDING_MODEL: str = "voyage-3"   # 1024d; see db.schema.EMBEDDING_DIM

# ── Retrieval fusion (RRF) ─────────────────────────────────────────────────────
RRF_K:         int   = 60
RRF_W_VECTOR:  float = 1.0
RRF_W_KEYWORD: float = 1.0
RRF_W_GRAPH:   float = 1.0

# ── Retrieval limits ───────────────────────────────────────────────────────────
TOP_K_FINAL:   int = 12
TOP_K_VECTOR:  int = 30
MAX_PER_PAPER: int = 3

# ── BM25 ───────────────────────────────────────────────────────────────────────
BM25_K1:        float = 1.2
BM25_B:         float = 0.75
BM25_SCORE_CAP: float = 20.0

# ── Personalized PageRank ──────────────────────────────────────────────────────
PPR_ALPHA:        float = 0.85
PPR_MAX_ITER:     int   = 50
PPR_TOL:          float = 1e-6
PPR_TOP_VERTICES: int   = 100
