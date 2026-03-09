"""Embedding model wrapper — sentence-transformers locally, SageMaker on AWS.

Swap backends by setting USE_SAGEMAKER_EMBED=true and SAGEMAKER_EMBED_ENDPOINT
in your .env.  No code changes needed.
"""
from __future__ import annotations

import numpy as np

from rag.config import (
    EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_MAX_LENGTH,
    USE_SAGEMAKER_EMBED,
    SAGEMAKER_EMBED_ENDPOINT,
    AWS_REGION,
)

# BGE-large variants need an instruction prefix for query embedding.
# BGE-M3 does NOT need a prefix (handles it internally).
_NEEDS_QUERY_PREFIX = "bge-large" in EMBED_MODEL.lower()
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


import torch, multiprocessing as _mp

# CPU thread tuning (no-op when CUDA is available)
if not torch.cuda.is_available():
    _cpu_cores = _mp.cpu_count()
    torch.set_num_threads(_cpu_cores)
    torch.set_num_interop_threads(max(1, _cpu_cores // 2))

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Embedder:
    """Thread-safe embedding wrapper.  A single instance is shared per process."""

    def __init__(self) -> None:
        self._use_sagemaker = USE_SAGEMAKER_EMBED and bool(SAGEMAKER_EMBED_ENDPOINT)
        if not self._use_sagemaker:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=_DEVICE)
            self._model.max_seq_length = EMBED_MAX_LENGTH

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed document passages (no instruction prefix)."""
        return self._encode(texts, is_query=False)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self._encode([query], is_query=True)[0]

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed multiple queries at once. Returns shape (N, dim)."""
        return self._encode(queries, is_query=True)

    # ── Private ───────────────────────────────────────────────────────────────

    def _encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if self._use_sagemaker:
            return self._sagemaker_encode(texts)
        if is_query and _NEEDS_QUERY_PREFIX:
            texts = [_QUERY_PREFIX + t for t in texts]
        return self._model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _sagemaker_encode(self, texts: list[str]) -> np.ndarray:
        """Embed via a SageMaker endpoint that returns a list of floats per text."""
        import boto3
        import json

        runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        vectors: list[list[float]] = []
        for text in texts:
            resp = runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_EMBED_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps({"inputs": text}),
            )
            payload = json.loads(resp["Body"].read())
            # Handle both [[...]] and [...] response shapes
            vec = payload[0] if isinstance(payload[0], list) else payload
            vectors.append(vec)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)
