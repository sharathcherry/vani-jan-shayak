"""rag.py — RAG answer retrieval with a 3-layer cache.

Retrieves a government scheme answer for an English query using three
cache layers to minimise latency and cost:

  L1 — In-memory LRU  (~0 ms)   Survives across warm Lambda invocations.
  L2 — S3 object      (~30 ms)  Survives cold starts; 24-hour TTL.
  L3 — RAG Lambda     (~1–8 s)  Full Qdrant hybrid search + Cohere rerank
                                 + Amazon Bedrock answer generation.

Also provides intent detection and URL extraction utilities used by the
voice pipeline to guard against off-topic queries.
"""

import hashlib
import json
import re
import time

from config import s3, lambda_client, S3_BUCKET_IN, CACHE_TTL_SECONDS, RAG_LAMBDA_NAME
from cache import answer_cache


# ---------------------------------------------------------------------------
# Intent guard
# ---------------------------------------------------------------------------
# If NONE of these keywords appear in the translated English query, we return
# a clarification prompt instead of guessing a potentially irrelevant answer.
_INTENT_KEYWORDS = {
    "scheme", "schemes", "yojana", "scholarship", "loan", "subsidy",
    "benefit", "benefits", "apply", "application", "eligible", "eligibility",
    "government", "pension", "insurance", "grant", "allowance", "ration",
    "card", "certificate", "income", "caste", "student", "farmer", "woman",
    "disability", "welfare", "help", "support", "fee", "free", "money",
    "what", "how", "when", "where", "which", "is there", "are there",
    "can i", "do i", "tell me", "list", "?",
}

_URL_RE = re.compile(r'https?://[^\s,)>"\']+')


def has_scheme_intent(english_text: str) -> bool:
    """Return True if the query looks like a government scheme question."""
    lower = english_text.lower()
    return any(k in lower for k in _INTENT_KEYWORDS)


def extract_urls(text: str) -> list[str]:
    """Return all URLs found in text, deduplicated, order preserved."""
    seen: set[str] = set()
    result: list[str] = []
    for url in _URL_RE.findall(text):
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def strip_urls(text: str) -> str:
    """Remove URLs from text so TTS does not read them aloud."""
    return _URL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _normalize_query(q: str) -> str:
    """Lowercase, collapse whitespace, and strip session context prefix."""
    if "New question:" in q:
        q = q.split("New question:")[-1]
    return " ".join(q.lower().split())


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def get_rag_answer(english_query: str) -> str:
    """Retrieve a government scheme answer with 3-layer caching.

    Cache key is a SHA-256 hash of the normalised query so semantically
    identical questions (different whitespace / capitalisation) share a
    cache entry.
    """
    normalized = _normalize_query(english_query)
    cache_key  = "wa-cache/" + hashlib.sha256(normalized.encode()).hexdigest() + ".txt"

    # -- L1: in-memory LRU (fastest) --------------------------------------
    mem_hit = answer_cache.get(normalized)
    if mem_hit is not None:
        print(f"[cache-L1] HIT (in-memory) key={cache_key[-12:]}")
        return mem_hit

    # -- L2: S3 persistent cache (survives cold starts) -------------------
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_IN, Key=cache_key)
        age = time.time() - obj["LastModified"].timestamp()
        if age < CACHE_TTL_SECONDS:
            cached = obj["Body"].read().decode("utf-8")
            print(f"[cache-L2] HIT (S3, age={age/3600:.1f}h) key={cache_key[-12:]}")
            answer_cache.set(normalized, cached)   # promote to L1
            return cached
        print(f"[cache-L2] EXPIRED (age={age/3600:.1f}h)")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception:
        pass

    # -- L3: RAG Lambda (Qdrant hybrid search → Cohere rerank → Bedrock) -
    print("[cache] MISS — invoking RAG Lambda")
    response = lambda_client.invoke(
        FunctionName=RAG_LAMBDA_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps({
            "rawPath": "/debug/query",
            "body": json.dumps({"query": english_query}),
            "headers": {"content-type": "application/json"},
        }),
    )
    result = json.loads(response["Payload"].read())
    body   = json.loads(result.get("body", "{}"))
    answer = body.get("answer", "No information found.")

    # Write-through to both cache layers (best-effort, never fail the request)
    answer_cache.set(normalized, answer)
    try:
        s3.put_object(
            Bucket=S3_BUCKET_IN,
            Key=cache_key,
            Body=answer.encode("utf-8"),
        )
        print(f"[cache-L2] STORED key={cache_key[-12:]}")
    except Exception:
        pass

    return answer
