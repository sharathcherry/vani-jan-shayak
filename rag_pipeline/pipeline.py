"""Full RAG pipeline orchestrator.

Query → Expand → Rewrite + Multi-query (1 LLM call) → Embed → Hybrid Retrieve
→ RRF + Boost + Filter + Dedup → Rerank → Parent-doc → Generate Answer

All stages are configurable via env vars (see config.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.config import (
    ENABLE_AUTO_FILTER,
    ENABLE_MULTI_QUERY,
    ENABLE_QUERY_EXPAND,
    ENABLE_QUERY_REWRITE,
    FINAL_TOP_K,
    NUM_MULTI_QUERIES,
    RERANK_TOP_K,
)
from rag.embedder import Embedder
from rag.llm_client import LLMClient
from rag.query_processor import (
    build_query_prompt,
    expand_query,
    extract_metadata_filters,
    parse_query_variants,
)
from rag.reranker import Reranker
from rag.retriever import RetrievedChunk, Retriever


@dataclass
class RAGAnswer:
    question: str
    answer: str
    schemes: list[dict]               # Parent documents sent to LLM
    retrieved_chunks: list[dict]       # Debug: chunks after reranking
    query_variants: list[str]          # All queries used for retrieval
    metadata_filter: dict              # Filters auto-extracted from query
    latency_ms: float
    stage_latencies: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """
    Production RAG pipeline.  Load once; call answer() for every request.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.answer("schemes for women farmers in Punjab")
        print(result.answer)
    """

    def __init__(self) -> None:
        print("[pipeline] Loading embedding model ...")
        self.embedder = Embedder()
        print("[pipeline] Loading BM25 + Qdrant ...")
        self.retriever = Retriever()
        print("[pipeline] Loading reranker ...")
        self.reranker = Reranker()
        print("[pipeline] Loading LLM client ...")
        self.llm = LLMClient()
        print("[pipeline] Ready.")

    # ── Public ────────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        metadata_filter: dict | None = None,
        top_k: int = FINAL_TOP_K,
    ) -> RAGAnswer:
        """
        End-to-end RAG: question → answer.

        Args:
            question:        User's natural-language question.
            metadata_filter: Optional dict to hard-filter by state/category/flags.
                             If None and ENABLE_AUTO_FILTER=true, extracted from question.
            top_k:           Number of schemes to use for answer generation.
        """
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # ── 1. Query expansion (rule-based, ~0ms) ────────────────────────────
        t = time.perf_counter()
        expanded = expand_query(question) if ENABLE_QUERY_EXPAND else question
        timings["expand_ms"] = (time.perf_counter() - t) * 1000

        # ── 2. Auto-extract metadata filters ─────────────────────────────────
        t = time.perf_counter()
        if metadata_filter is None and ENABLE_AUTO_FILTER:
            metadata_filter = extract_metadata_filters(question)
        metadata_filter = metadata_filter or {}
        timings["filter_extract_ms"] = (time.perf_counter() - t) * 1000

        # ── 3. LLM: rewrite + multi-query (single API call) ──────────────────
        t = time.perf_counter()
        primary_query = expanded
        variant_queries: list[str] = []

        if ENABLE_QUERY_REWRITE or ENABLE_MULTI_QUERY:
            n_variants = NUM_MULTI_QUERIES if ENABLE_MULTI_QUERY else 0
            prompt = build_query_prompt(expanded, n_variants)
            try:
                llm_out = self.llm.rewrite_and_expand_queries(expanded, n_variants, prompt)
                rewritten, variants = parse_query_variants(llm_out, n_variants)
                if ENABLE_QUERY_REWRITE and rewritten:
                    primary_query = rewritten
                if ENABLE_MULTI_QUERY:
                    variant_queries = variants
            except Exception as exc:
                # LLM call failed: continue with expanded query only
                print(f"[pipeline] Query LLM call failed ({exc}); using expanded query.")

        all_queries = [primary_query] + variant_queries
        timings["query_llm_ms"] = (time.perf_counter() - t) * 1000

        # ── 4. Embed all queries ──────────────────────────────────────────────
        t = time.perf_counter()
        query_vectors = self.embedder.embed_queries(all_queries)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        # ── 5. Hybrid retrieval (multi-query with RRF) ────────────────────────
        t = time.perf_counter()
        chunks: list[RetrievedChunk] = self.retriever.retrieve_multi(
            query_vectors=query_vectors,
            query_texts=all_queries,
            metadata_filter=metadata_filter if metadata_filter else None,
            top_k=RERANK_TOP_K,
        )
        timings["retrieve_ms"] = (time.perf_counter() - t) * 1000

        # ── 6. Rerank ─────────────────────────────────────────────────────────
        t = time.perf_counter()
        # Rerank against the original question for best semantic alignment
        chunks = self.reranker.rerank(question, chunks, top_k=top_k)
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        # ── 7. Parent-document reconstruction ────────────────────────────────
        t = time.perf_counter()
        parent_docs = [c.get_parent_doc(self.retriever._parent_docs) for c in chunks]
        timings["parent_doc_ms"] = (time.perf_counter() - t) * 1000

        # ── 8. LLM answer generation ──────────────────────────────────────────
        t = time.perf_counter()
        if parent_docs:
            answer_text = self.llm.generate_answer(question, parent_docs)
        else:
            answer_text = (
                "No relevant government schemes were found for your query. "
                "Please try rephrasing or providing more details about your situation."
            )
        timings["gen_ms"] = (time.perf_counter() - t) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        return RAGAnswer(
            question=question,
            answer=answer_text,
            schemes=parent_docs,
            retrieved_chunks=[
                {
                    "scheme_name": c.scheme_name,
                    "chunk_type":  c.chunk_type,
                    "score":       round(c.score, 4),
                    "text":        c.text[:300] + "..." if len(c.text) > 300 else c.text,
                }
                for c in chunks
            ],
            query_variants=all_queries,
            metadata_filter=metadata_filter,
            latency_ms=round(total_ms, 1),
            stage_latencies={k: round(v, 1) for k, v in timings.items()},
        )
