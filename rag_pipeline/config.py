"""Central configuration — change env vars to switch local ↔ AWS."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "structured_schemes"
INDEX_DIR = BASE_DIR / "rag_index"
INDEX_DIR.mkdir(exist_ok=True)

# ── Embedding ─────────────────────────────────────────────────────────────────
# Local: sentence-transformers downloads the model automatically.
# AWS:   set SAGEMAKER_EMBED_ENDPOINT and optionally USE_SAGEMAKER=true.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")   # multilingual; great for Indian schemes
EMBED_DIM = 1024
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_LENGTH = 512

# ── Qdrant Vector Store ───────────────────────────────────────────────────────
# Local:  leave QDRANT_URL empty → files stored at QDRANT_LOCAL_PATH.
# AWS:    set QDRANT_URL=http://<ec2-ip>:6333  (Qdrant running in Docker on EC2)
#         or QDRANT_URL=https://<cluster>.aws.cloud.qdrant.io  (Qdrant Cloud)
QDRANT_LOCAL_PATH = str(INDEX_DIR / "qdrant_db")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "gov_schemes"

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = str(INDEX_DIR / "bm25_index.pkl")
CORPUS_PATH = str(INDEX_DIR / "corpus.pkl")

# ── Retrieval knobs ───────────────────────────────────────────────────────────
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "50"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "50"))
RRF_K = int(os.getenv("RRF_K", "60"))           # standard RRF constant
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
MAX_CHUNKS_PER_SCHEME = int(os.getenv("MAX_CHUNKS_PER_SCHEME", "2"))  # diversity

# ── Chunk type boosts (applied on top of RRF score) ───────────────────────────
CHUNK_TYPE_BOOST: dict[str, float] = {
    "qa":          1.5,   # synthetic Q&A chunks rank highest
    "eligibility": 1.2,   # eligibility text matches queries well
    "summary":     1.2,
    "benefits":    1.1,
    "full":        1.0,
    "application": 0.9,
}

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ── LLM ───────────────────────────────────────────────────────────────────────
# Supported providers: groq | openai | sagemaker
# Groq is the fastest and cheapest option — free tier handles production load.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SAGEMAKER_LLM_ENDPOINT = os.getenv("SAGEMAKER_LLM_ENDPOINT", "")
SAGEMAKER_EMBED_ENDPOINT = os.getenv("SAGEMAKER_EMBED_ENDPOINT", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
USE_SAGEMAKER_EMBED = os.getenv("USE_SAGEMAKER_EMBED", "false").lower() == "true"

# ── Query processing flags ────────────────────────────────────────────────────
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "true").lower() == "true"
ENABLE_QUERY_EXPAND = os.getenv("ENABLE_QUERY_EXPAND", "true").lower() == "true"
ENABLE_AUTO_FILTER = os.getenv("ENABLE_AUTO_FILTER", "true").lower() == "true"
NUM_MULTI_QUERIES = int(os.getenv("NUM_MULTI_QUERIES", "2"))
