"""Hybrid BM25 + Qdrant vector retriever with RRF, Qdrant pre-filtering, and dedup.

Every query runs this pipeline:
  1.  Build Qdrant Filter from metadata dict  → server-side O(log n) pre-filter
  2.  BM25 sparse search  (keyword / exact-match)
  3.  Qdrant dense search (semantic / vector) with the pre-filter applied
  4.  RRF fusion over the union of both result sets
  5.  Chunk-type boost   (QA 1.5 > eligibility/summary 1.2 > benefits 1.1 > application 0.9)
  6.  Scheme-level deduplication  (max MAX_CHUNKS_PER_SCHEME per scheme_id)
  7.  Return top-K candidates for the cross-encoder reranker

Multi-query: run steps 2-6 for every query variant produced by the LLM,
then keep the best-scoring chunk per chunk_id before final dedup + top-K.

Key design decisions:
  - Text IS stored in Qdrant payload (not just metadata) → no BM25 corpus needed
    for text retrieval; BM25 corpus kept only for the sparse scoring pass.
  - Qdrant payload indexes created at ingest time → filtered queries fast.
  - parent_docs.pkl loaded once at init for O(1) parent-doc lookup.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    ScoredPoint,
)

from rag.config import (
    BM25_INDEX_PATH,
    BM25_TOP_K,
    CHUNK_TYPE_BOOST,
    CORPUS_PATH,
    INDEX_DIR,
    MAX_CHUNKS_PER_SCHEME,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    RERANK_TOP_K,
    RRF_K,
    VECTOR_TOP_K,
)

PARENT_DOCS_PATH = str(INDEX_DIR / "parent_docs.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    id:          str
    scheme_id:   str
    scheme_name: str
    chunk_type:  str
    text:        str
    score:       float
    payload:     dict = field(default_factory=dict)

    def get_parent_doc(self, parent_docs: dict[str, dict]) -> dict:
        """
        Return the full structured parent document for this scheme.
        Falls back to sparse payload reconstruction if the scheme_id
        is not in parent_docs (should not happen in normal operation).
        """
        doc = parent_docs.get(self.scheme_id)
        if doc:
            return doc
        p = self.payload
        return {
            "scheme_id":       self.scheme_id,
            "scheme_name":     self.scheme_name,
            "state_or_ut":     p.get("state_or_ut", ""),
            "scheme_category": p.get("scheme_category", ""),
            "ministry":        p.get("ministry", ""),
            "summary":         "",
            "eligibility":     "",
            "benefits":        "",
            "application":     "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Production-grade hybrid retriever.
    Instantiate once; thread-safe for concurrent reads.
    """

    def __init__(self) -> None:
        self._qdrant      = _make_qdrant_client()
        self._bm25, self._corpus = _load_bm25_and_corpus()
        # O(1) lookup: chunk_id → corpus item (for sparse text + filter data)
        self._corpus_idx  = {item["id"]: item for item in self._corpus}
        self._parent_docs = _load_parent_docs()

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text:   str,
        metadata_filter: dict | None = None,
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """Single-query hybrid retrieval → RRF → boost → dedup → top-K."""
        qdrant_filter = _build_qdrant_filter(metadata_filter)

        # ── Sparse: BM25 ──────────────────────────────────────────────────────
        bm25_ranked = self._bm25_search(query_text, BM25_TOP_K, metadata_filter)
        bm25_rank   = {cid: rank for rank, (cid, _) in enumerate(bm25_ranked)}

        # ── Dense: Qdrant ANN with server-side pre-filter ─────────────────────
        vec_hits    = self._vector_search(query_vector, VECTOR_TOP_K, qdrant_filter)
        vec_rank    = {str(h.id): rank for rank, h in enumerate(vec_hits)}
        # payload from Qdrant includes text (stored at ingest time)
        vec_payload = {str(h.id): (h.payload or {}) for h in vec_hits}

        # ── RRF fusion ────────────────────────────────────────────────────────
        all_ids = set(bm25_rank) | set(vec_rank)
        fused: list[tuple[str, float]] = []
        for cid in all_ids:
            score = 0.0
            if cid in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[cid] + 1)
            if cid in vec_rank:
                score += 1.0 / (RRF_K + vec_rank[cid] + 1)
            fused.append((cid, score))
        fused.sort(key=lambda x: x[1], reverse=True)

        # ── Enrich + boost ────────────────────────────────────────────────────
        results: list[RetrievedChunk] = []
        for cid, rrf_score in fused:
            # Prefer Qdrant payload (has text + full metadata);
            # fall back to BM25 corpus item for chunks that only appeared in BM25.
            payload    = vec_payload.get(cid) or self._corpus_idx.get(cid, {})
            chunk_type = payload.get("chunk_type", "full")
            boost      = CHUNK_TYPE_BOOST.get(chunk_type, 1.0)

            # Text: Qdrant payload first, then BM25 corpus
            text = payload.get("text") or self._corpus_idx.get(cid, {}).get("text", "")

            results.append(RetrievedChunk(
                id          = cid,
                scheme_id   = payload.get("scheme_id", ""),
                scheme_name = payload.get("scheme_name", ""),
                chunk_type  = chunk_type,
                text        = text,
                score       = rrf_score * boost,
                payload     = payload,
            ))

        # ── Dedup → top-K ────────────────────────────────────────────────────
        return _deduplicate(results, MAX_CHUNKS_PER_SCHEME)[:top_k]

    def retrieve_multi(
        self,
        query_vectors: np.ndarray,
        query_texts:   list[str],
        metadata_filter: dict | None = None,
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Multi-query retrieval — run retrieve() for each (vector, text) pair,
        keep the best score per chunk_id, then final dedup + top-K.
        """
        merged: dict[str, RetrievedChunk] = {}
        for vec, text in zip(query_vectors, query_texts):
            for chunk in self.retrieve(vec, text, metadata_filter=metadata_filter, top_k=top_k):
                prev = merged.get(chunk.id)
                if prev is None or chunk.score > prev.score:
                    merged[chunk.id] = chunk

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return _deduplicate(ranked, MAX_CHUNKS_PER_SCHEME)[:top_k]

    # ── Private ───────────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[tuple[str, float]]:
        """BM25 keyword search with Python-side post-filter. Returns [(id, score)]."""
        tokens      = query.lower().split()
        scores: np.ndarray = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0.0:
                break
            item = self._corpus[idx]
            if metadata_filter and not _passes_bm25_filter(item, metadata_filter):
                continue
            results.append((item["id"], float(scores[idx])))
        return results

    def _vector_search(
        self,
        vector: np.ndarray,
        top_k: int,
        qdrant_filter: Filter | None,
    ) -> list[ScoredPoint]:
        """Qdrant ANN search with optional server-side pre-filter."""
        return self._qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant filter builder  (server-side — O(log n) with payload indexes)
# ─────────────────────────────────────────────────────────────────────────────

def _build_qdrant_filter(meta: dict | None) -> Filter | None:
    """
    Build a Qdrant Filter from our internal metadata dict.
    Returns None if no filters active (full collection search).

    Supported keys:
      state         → state_or_ut  (exact string match)
      category      → scheme_category  (exact string match)
      for_women     → is_for_women = true
      for_farmers   → is_for_farmers = true
      for_disabled  → is_for_disabled = true
      for_sc_st     → is_for_sc_st = true
      for_students  → is_for_students = true

    All these fields have Qdrant payload indexes (created in indexer.py),
    so filtering is O(log n), not O(n).
    """
    if not meta:
        return None

    must: list = []

    if state := meta.get("state"):
        must.append(FieldCondition(key="state_or_ut",     match=MatchValue(value=state)))
    if category := meta.get("category"):
        must.append(FieldCondition(key="scheme_category", match=MatchValue(value=category)))

    for meta_key, payload_key in (
        ("for_women",    "is_for_women"),
        ("for_farmers",  "is_for_farmers"),
        ("for_disabled", "is_for_disabled"),
        ("for_sc_st",    "is_for_sc_st"),
        ("for_students", "is_for_students"),
    ):
        if meta.get(meta_key):
            must.append(FieldCondition(key=payload_key, match=MatchValue(value=True)))

    return Filter(must=must) if must else None


# ─────────────────────────────────────────────────────────────────────────────
# BM25 post-filter  (mirrors Qdrant filter for the sparse path)
# ─────────────────────────────────────────────────────────────────────────────

def _passes_bm25_filter(item: dict, meta: dict) -> bool:
    if state := meta.get("state"):
        if item.get("state_or_ut", "").lower() != state.lower():
            return False
    if category := meta.get("category"):
        if item.get("scheme_category", "").lower() != category.lower():
            return False
    for meta_key, payload_key in (
        ("for_women",    "is_for_women"),
        ("for_farmers",  "is_for_farmers"),
        ("for_disabled", "is_for_disabled"),
        ("for_sc_st",    "is_for_sc_st"),
        ("for_students", "is_for_students"),
    ):
        if meta.get(meta_key) and not item.get(payload_key):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(chunks: list[RetrievedChunk], max_per_scheme: int) -> list[RetrievedChunk]:
    """
    Keep at most max_per_scheme chunks per scheme_id.
    Chunks must already be sorted by score descending.
    """
    counts: dict[str, int] = defaultdict(int)
    out: list[RetrievedChunk] = []
    for chunk in chunks:
        if counts[chunk.scheme_id] < max_per_scheme:
            out.append(chunk)
            counts[chunk.scheme_id] += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _make_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    return QdrantClient(path=QDRANT_LOCAL_PATH)


def _load_bm25_and_corpus():
    bm25_p   = Path(BM25_INDEX_PATH)
    corpus_p = Path(CORPUS_PATH)
    if not bm25_p.exists() or not corpus_p.exists():
        raise FileNotFoundError(
            "BM25 index not found. Run `python ingest.py` to build the index first."
        )
    with open(bm25_p, "rb") as f:
        bm25 = pickle.load(f)
    with open(corpus_p, "rb") as f:
        corpus = pickle.load(f)
    return bm25, corpus


def _load_parent_docs() -> dict[str, dict]:
    p = Path(PARENT_DOCS_PATH)
    if not p.exists():
        raise FileNotFoundError(
            "parent_docs.pkl not found. Run `python ingest.py` to build the index first."
        )
    with open(p, "rb") as f:
        return pickle.load(f)
