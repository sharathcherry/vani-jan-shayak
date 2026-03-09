"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Takes a user query + candidate chunks → returns chunks re-ordered by
relevance score.  This corrects ordering errors from approximate retrieval.

Model notes:
  - bge-reranker-v2-m3 is multilingual, works well for Hindi terms mixed in.
  - Runs on CPU; 20-chunk batch ≈ 200-500ms on a modern CPU.
  - For GPU, torch automatically detects CUDA if available.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.config import RERANKER_MODEL, FINAL_TOP_K
from rag.retriever import RetrievedChunk


class Reranker:
    """Cross-encoder reranker.  Shared single instance per process."""

    def __init__(self) -> None:
        # trust_remote_code=False is fine for bge-reranker-v2-m3
        self._model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
            automodel_args={"torch_dtype": "auto"},
        )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = FINAL_TOP_K,
    ) -> list[RetrievedChunk]:
        """Re-score all chunks and return top_k sorted by reranker score."""
        if not chunks:
            return []

        # Build (query, passage) pairs — use the chunk text directly
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach reranker scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]
