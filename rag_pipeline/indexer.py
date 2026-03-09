"""Build BM25 + Qdrant indexes from bedrock_chunks/ pre-made chunk files.

Chunk inventory (all have question-augmented text for better retrieval):
  summary/      1919 chunks  — "What is X, who is it for, what does it do?"
  eligibility/  1919 chunks  — "Who is eligible, what are the criteria?"
  benefits/     1919 chunks  — "What benefits, how much money, subsidy %?"
  application/  1919 chunks  — "How to apply, what documents needed?"
  qa/          14611 chunks  — Individual Q&A pairs (avg 7.6 per scheme)
  ─────────────────────────────
  TOTAL        22287 chunks

Every chunk has a "Question this chunk answers: ..." prefix baked in —
question-augmented embedding aligns chunk space with user query space.

What this indexer does:
  1. Loads structured_schemes/*.json → parent_docs.pkl (full scheme text per scheme_id)
  2. Walks bedrock_chunks/*/  → 22k chunk texts + metadata
  3. Embeds with BAAI/bge-m3, upserts into Qdrant (on-disk)
  4. Creates Qdrant PAYLOAD INDEXES on all filterable fields → O(log n) filters
  5. Builds BM25Okapi index over all chunk texts → saved to disk

Run once:      python ingest.py
Force rebuild: python ingest.py --force
"""
from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.config import (
    BM25_INDEX_PATH,
    CORPUS_PATH,
    DATA_DIR,
    EMBED_DIM,
    INDEX_DIR,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
)
from rag.embedder import Embedder

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "bedrock_chunks"
PARENT_DOCS_PATH = str(INDEX_DIR / "parent_docs.pkl")

CHUNK_SUBDIRS = ["summary", "eligibility", "benefits", "application", "qa", "new"]


# ── Qdrant payload fields to index for O(log n) filtering ────────────────────
# keyword → exact-match string fields
_KEYWORD_FIELDS = [
    "state_or_ut",
    "scheme_category",
    "scheme_level",
    "chunk_type",
    "ministry",
]
# bool → boolean fields
_BOOL_FIELDS = [
    "is_for_women",
    "is_for_disabled",
    "is_for_sc_st",
    "is_for_students",
    "is_for_farmers",
]
# float → numeric range fields
_FLOAT_FIELDS = [
    "max_amount_inr",
    "min_amount_inr",
    "max_investment_inr",
    "data_completeness",
]


def _make_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    return QdrantClient(path=QDRANT_LOCAL_PATH)


def _load_parent_docs() -> dict[str, dict]:
    """Build scheme_id → full parent doc dict from structured_schemes/."""
    parent_docs: dict[str, dict] = {}
    for path in DATA_DIR.glob("*.json"):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = s.get("scheme_id", "")
        if not sid:
            continue
        parent_docs[sid] = {
            "scheme_id":       sid,
            "scheme_name":     s.get("scheme_name", ""),
            "state_or_ut":     s.get("state_or_ut", ""),
            "scheme_category": s.get("scheme_category", ""),
            "scheme_level":    s.get("scheme_level", ""),
            "ministry":        s.get("ministry_or_department", ""),
            "summary":         s.get("text_summary", ""),
            "eligibility":     s.get("text_eligibility", ""),
            "benefits":        s.get("text_benefits", ""),
            "application":     s.get("text_application", ""),
            "objective":       s.get("objective", ""),
        }
    return parent_docs


def _load_all_chunks() -> list[dict]:
    """
    Walk bedrock_chunks/<subdir>/*.txt + *.metadata.json.
    Deduplicates on (scheme_id, chunk_type, qa_index).
    Returns fully-populated chunk dicts.
    """
    seen: set[str] = set()
    chunks: list[dict] = []

    for subdir in CHUNK_SUBDIRS:
        subdir_path = CHUNKS_DIR / subdir
        if not subdir_path.exists():
            continue

        for txt_path in sorted(f for f in subdir_path.iterdir() if f.suffix == ".txt"):
            meta_path = Path(str(txt_path) + ".metadata.json")
            if not meta_path.exists():
                continue
            try:
                text     = txt_path.read_text(encoding="utf-8").strip()
                meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not text:
                continue

            attrs       = meta_raw.get("metadataAttributes", {})
            scheme_id   = attrs.get("scheme_id", "")
            scheme_name = attrs.get("scheme_name", "")
            chunk_type  = attrs.get("chunk_type", subdir)
            qa_index    = attrs.get("qa_index", -1)

            if not scheme_id:
                continue

            dedup_key = f"{scheme_id}::{chunk_type}::{qa_index}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{scheme_id}::{chunk_type}::{qa_index}",
            ))

            def _b(key: str) -> bool:
                v = attrs.get(key, False)
                return v if isinstance(v, bool) else str(v).lower() == "true"

            def _f(key: str):
                v = attrs.get(key)
                if v is None or v in ("", "null", "None"):
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            chunks.append({
                # Core
                "id":              point_id,
                "text":            text,              # stored in BOTH Qdrant + BM25 corpus
                "chunk_type":      chunk_type,
                "scheme_id":       scheme_id,
                "scheme_name":     scheme_name,
                "qa_index":        qa_index,
                # Keyword filter fields
                "state_or_ut":     attrs.get("state_or_ut", ""),
                "scheme_category": attrs.get("scheme_category", ""),
                "scheme_level":    attrs.get("scheme_level", ""),
                "ministry":        attrs.get("ministry", ""),
                # Boolean beneficiary flags
                "is_for_women":    _b("is_for_women"),
                "is_for_disabled": _b("is_for_disabled"),
                "is_for_sc_st":    _b("is_for_sc_st"),
                "is_for_students": _b("is_for_students"),
                "is_for_farmers":  _b("is_for_farmers"),
                # Numeric filter fields
                "max_amount_inr":     _f("meta_max_amount_inr"),
                "min_amount_inr":     _f("meta_min_amount_inr"),
                "max_investment_inr": _f("meta_max_investment_inr"),
                "data_completeness":  _f("data_completeness"),
            })

    return chunks


def _create_payload_indexes(client: QdrantClient) -> None:
    """
    Create Qdrant payload indexes so filtered queries run in O(log n).

    Without these, every filtered search scans ALL 22k payloads in Python.
    With indexes, Qdrant narrows candidates before ANN search.
    """
    print("[indexer] Creating Qdrant payload indexes ...")

    for field_name in _KEYWORD_FIELDS:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    for field_name in _BOOL_FIELDS:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=PayloadSchemaType.BOOL,
        )

    for field_name in _FLOAT_FIELDS:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=PayloadSchemaType.FLOAT,
        )

    print(
        f"[indexer] Payload indexes created:\n"
        f"  keyword : {_KEYWORD_FIELDS}\n"
        f"  bool    : {_BOOL_FIELDS}\n"
        f"  float   : {_FLOAT_FIELDS}"
    )


def build_index(force: bool = False) -> None:
    client = _make_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    bm25_exists        = Path(BM25_INDEX_PATH).exists()
    parent_docs_exist  = Path(PARENT_DOCS_PATH).exists()

    if QDRANT_COLLECTION in existing and bm25_exists and parent_docs_exist and not force:
        print("[indexer] Index already built. Pass --force to rebuild.")
        return

    # ── 1. Parent docs ────────────────────────────────────────────────────
    print("[indexer] Loading parent documents from structured_schemes/ ...")
    parent_docs = _load_parent_docs()
    print(f"[indexer] {len(parent_docs)} parent docs loaded.")
    with open(PARENT_DOCS_PATH, "wb") as f:
        pickle.dump(parent_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── 2. Load all pre-made chunks ───────────────────────────────────────
    print("[indexer] Scanning bedrock_chunks/ ...")
    chunks = _load_all_chunks()

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    for ctype, cnt in sorted(by_type.items()):
        print(f"  {ctype:15}: {cnt:5} chunks")
    print(f"  {'TOTAL':15}: {len(chunks):5} chunks")

    # ── 3. (Re)create Qdrant collection ───────────────────────────────────
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE,
            on_disk=True,
        ),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200, on_disk=True),
        on_disk_payload=True,
    )

    # ── 4. Create payload indexes BEFORE upserting ────────────────────────
    # Must happen before data lands so Qdrant indexes as data arrives.
    _create_payload_indexes(client)

    # ── 5. Embed + upsert ─────────────────────────────────────────────────
    # sentence-transformers handles internal batching via EMBED_BATCH_SIZE.
    # We embed everything in one call, then upsert everything in one call.
    print("[indexer] Embedding all chunks (sentence-transformers batches internally) ...")
    embedder = Embedder()
    vectors = embedder.embed_passages([c["text"] for c in chunks])

    print("[indexer] Upserting all points into Qdrant ...")
    points = [
        PointStruct(
            id      = c["id"],
            vector  = v.tolist(),
            payload = {k: c[k] for k in c if k != "id"},   # text IS stored
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=False)

    # Trigger HNSW index build
    client.update_collection(
        collection_name=QDRANT_COLLECTION,
        optimizer_config={"indexing_threshold": 0},
    )
    print("[indexer] Qdrant upsert complete.")

    # ── 6. BM25 index ─────────────────────────────────────────────────────
    print("[indexer] Building BM25Okapi index ...")
    from rank_bm25 import BM25Okapi

    corpus = [
        {
            "id":              c["id"],
            "scheme_id":       c["scheme_id"],
            "scheme_name":     c["scheme_name"],
            "chunk_type":      c["chunk_type"],
            "text":            c["text"],
            "state_or_ut":     c["state_or_ut"],
            "scheme_category": c["scheme_category"],
            "is_for_women":    c["is_for_women"],
            "is_for_disabled": c["is_for_disabled"],
            "is_for_sc_st":    c["is_for_sc_st"],
            "is_for_students": c["is_for_students"],
            "is_for_farmers":  c["is_for_farmers"],
            "max_amount_inr":  c["max_amount_inr"],
        }
        for c in chunks
    ]

    tokenized = [item["text"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"\n[indexer] Done.\n"
        f"  Chunks  : {len(chunks)}\n"
        f"  Schemes : {len(parent_docs)}\n"
        f"  Qdrant  : {QDRANT_LOCAL_PATH if not QDRANT_URL else QDRANT_URL}\n"
        f"  BM25    : {BM25_INDEX_PATH}\n"
        f"  Parents : {PARENT_DOCS_PATH}"
    )
