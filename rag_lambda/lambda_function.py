"""
lambda_function.py
-------------------
AWS Lambda handler for:
  Twilio Voice/WhatsApp → AWS Transcribe → Qdrant Cloud → Bedrock → TTS reply

Environment variables (set in Lambda console):
  QDRANT_CLOUD_URL        https://126e5307-....eu-central-1-0.aws.cloud.qdrant.io:6333
  QDRANT_CLOUD_API_KEY    eyJ...
  QDRANT_COLLECTION       schemes_hybrid
  BEDROCK_LLM_MODEL       eu.amazon.nova-lite-v1:0
  AWS_REGION_NAME         eu-central-1
  TRANSCRIBE_S3_BUCKET    government-scheme
  SAGEMAKER_ENDPOINT      bge-large-en-endpoint   (optional, for GPU embeddings)
  TWILIO_ACCOUNT_SID      ACxxx
  TWILIO_AUTH_TOKEN       xxx
"""

from __future__ import annotations
import json
import os
import re
import shutil
import time
import threading
import urllib.parse
import urllib.request
import base64
import hashlib
import hmac
import boto3
import httpx

# ── Model Cache Setup (runs once per cold start) ──────────────────────────────
# /var/task is READ-ONLY in Lambda. fastembed needs WRITE access for lock files.
# Copy pre-baked models from /var/task → /tmp (writable) at container startup.
_MODEL_SRC = "/var/task/fastembed_cache"
_MODEL_DST = "/tmp/fastembed_cache"

def _setup_model_cache():
    if os.path.exists(_MODEL_DST):
        print("[startup] Models already in /tmp — warm container.")
        return
    if os.path.exists(_MODEL_SRC):
        print(f"[startup] Copying models {_MODEL_SRC} → {_MODEL_DST} ...")
        shutil.copytree(_MODEL_SRC, _MODEL_DST)
        print("[startup] ✅ Models ready.")
    else:
        print("[startup] ⚠ No pre-baked models. Will download on first query.")
        os.makedirs(_MODEL_DST, exist_ok=True)
    os.environ["FASTEMBED_CACHE_DIR"] = _MODEL_DST
    os.environ["HF_HOME"] = "/tmp/hf_home"
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

_setup_model_cache()  # Runs once at container init (cold start only)

# ── Config ────────────────────────────────────────────────────────────────
QDRANT_URL       = os.environ["QDRANT_CLOUD_URL"].rstrip("/")
QDRANT_API_KEY   = os.environ["QDRANT_CLOUD_API_KEY"]
COLLECTION       = os.environ.get("QDRANT_COLLECTION", "schemes_hybrid")
LLM_MODEL        = os.environ.get("BEDROCK_LLM_MODEL", "eu.amazon.nova-lite-v1:0")
AWS_REGION       = os.environ.get("AWS_REGION_NAME", "eu-central-1")
S3_BUCKET        = os.environ.get("TRANSCRIBE_S3_BUCKET", "government-scheme")
SM_ENDPOINT      = os.environ.get("SAGEMAKER_ENDPOINT", "")   # optional
TWILIO_SID       = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN     = os.environ.get("TWILIO_AUTH_TOKEN", "")
CANDIDATE_K      = 20
TOP_K            = 5
AZURE_RERANK_URL = os.environ.get("AZURE_RERANK_URL", "")  # Azure AI Foundry Cohere rerank endpoint
AZURE_RERANK_KEY = os.environ.get("AZURE_RERANK_KEY", "")  # Azure AI Foundry API key
RERANK_K         = int(os.environ.get("RERANK_K", "5"))

_LANG_NAMES = {
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "en-IN": "English",
}

# Unicode ranges for Indian scripts — used to detect language from Twilio SpeechResult text
# te-IN is #1 priority — checked first in both script detection and Azure LID.
_SCRIPT_DETECT = [
    ("te-IN", '\u0c00', '\u0c7f'),  # Telugu   ← #1 PRIORITY
    ("kn-IN", '\u0c80', '\u0cff'),  # Kannada  ← PRIORITY
    ("hi-IN", '\u0900', '\u097f'),  # Devanagari / Hindi ← PRIORITY
    ("ta-IN", '\u0b80', '\u0bff'),  # Tamil
    ("ml-IN", '\u0d00', '\u0d7f'),  # Malayalam
    ("mr-IN", '\u0900', '\u097f'),  # Marathi (shares Devanagari — checked after hi-IN)
    ("gu-IN", '\u0a80', '\u0aff'),  # Gujarati
    ("pa-IN", '\u0a00', '\u0a7f'),  # Gurmukhi (Punjabi)
    ("or-IN", '\u0b00', '\u0b7f'),  # Odia
    ("bn-IN", '\u0980', '\u09ff'),  # Bengali
]


def _detect_lang_from_script(text: str, fallback: str = "en-IN") -> str:
    """Detect language from Unicode script characters in the transcript.

    Returns the language code with the highest character density (\u226515% threshold),
    or `fallback` if no Indian script is detected (e.g. Latin/English).
    """
    if not text:
        return fallback
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return fallback
    total = len(non_space)
    best_lang, best_count = fallback, 0
    seen = set()
    for lang, lo, hi in _SCRIPT_DETECT:
        if lang in seen:
            continue
        seen.add(lang)
        count = sum(1 for c in non_space if lo <= c <= hi)
        if count > best_count and (count / total) >= 0.15:
            best_count, best_lang = count, lang
    return best_lang


# ── Query metadata extraction ─────────────────────────────────────────────────
# Ported from github_upload/rag/query_processor.py + retriever.py
# These run in the Lambda with zero LLM cost (pure regex + dict lookup).

# State values in Qdrant that represent national/central schemes.
# These must ALWAYS be included alongside any state filter so central schemes
# (PM Kisan, PM Awas Yojana, etc.) are not excluded for state-specific queries.
_NATIONAL_STATE_LABELS = frozenset({
    "Central", "All States/UTs", "PAN India", "All",
    "National", "India", "All states and UTs", "All Indian states",
})

# Canonical state name map — lowercase user input → official Qdrant stored value
_STATE_MAP: dict[str, str] = {
    "andhra pradesh": "Andhra Pradesh", "ap": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh", "hp": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha", "orissa": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "tn": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal", "wb": "West Bengal",
    "andaman": "Andaman and Nicobar Islands",
    "chandigarh": "Chandigarh",
    "dadra": "Dadra and Nagar Haveli and Daman and Diu",
    "daman": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi": "Delhi",
    "jammu": "Jammu & Kashmir", "kashmir": "Jammu & Kashmir", "j&k": "Jammu & Kashmir",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "puducherry": "Puducherry", "pondicherry": "Puducherry",
}

# Domain synonym expansion — adds related terms to improve BM25 recall
_EXPANSION_DICT: dict[str, str] = {
    "farmer":         "kisan agriculture crop subsidy farming",
    "kisan":          "farmer agriculture crop subsidy farming",
    "weaver":         "handloom bunkar textile artisan weaving",
    "fisherman":      "fisheries fishing fisher marine coastal",
    "artisan":        "craftsman handicraft skill traditional",
    "widow":          "widowed destitute women pension bereaved",
    "disabled":       "disability handicapped divyang differently abled",
    "divyang":        "disabled handicapped disability differently abled",
    "student":        "education scholarship school college university",
    "youth":          "young unemployed skill development",
    "women":          "woman female girl mahila",
    "mahila":         "women woman female girl",
    "senior citizen": "elderly old age pension",
    "scholarship":    "stipend financial assistance education grant",
    "pension":        "monthly allowance retirement old age financial support",
    "subsidy":        "financial assistance grant incentive reduction",
    "loan":           "credit finance borrowing interest",
    "sc":             "scheduled caste dalit backward",
    "st":             "scheduled tribe tribal adivasi",
    "obc":            "other backward class backward community",
    "bpl":            "below poverty line poor low income",
    "health":         "medical healthcare hospital insurance treatment",
    "housing":        "house shelter accommodation awas pradhan mantri",
    "skill":          "training vocational apprenticeship",
    "pm":             "pradhan mantri prime minister central government",
    "pmay":           "pradhan mantri awas yojana housing scheme",
    "pmkvy":          "pradhan mantri kaushal vikas yojana skill training",
    "pm kisan":       "pradhan mantri kisan samman nidhi farmer income support",
    "msme":           "micro small medium enterprise startup business",
    "startup":        "new business enterprise entrepreneurship msme",
}

# Beneficiary flag patterns — (qdrant_bool_field, [regex_patterns_on_query])
_BENEFICIARY_PATTERNS: list[tuple[str, list[str]]] = [
    ("is_for_farmers",  ["farmer", "kisan", "agricultur", "farming", "crop"]),
    ("is_for_women",    ["women", "woman", "female", "girl", "mahila", r"\bwidow"]),
    ("is_for_disabled", ["disab", "handicap", "divyang", "differently abled"]),
    ("is_for_sc_st",    [r"\bsc\b", r"\bst\b", "scheduled caste", "scheduled tribe",
                         "dalit", "tribal", "adivasi"]),
    ("is_for_students", ["student", "scholarship", "school", "college",
                         "universit", "educat", "b\.?tech", "btech", "mbbs",
                         "b\.sc", "m\.tech", "phd"]),
]


def _expand_query(query: str) -> str:
    """Append domain synonyms to improve BM25 sparse recall. Zero latency."""
    q_lower = query.lower()
    extras = [exp for kw, exp in _EXPANSION_DICT.items() if kw in q_lower]
    return query + " " + " ".join(extras) if extras else query


def _extract_metadata_filters(query: str) -> dict:
    """
    Parse structured metadata filters from free-text English query.
    Returns a dict with keys: state, is_for_sc_st, is_for_students, etc.
    Empty dict = no filters (search everything).
    Pure regex — no LLM call, ~0ms latency.
    """
    q_lower = query.lower()
    filters: dict = {}

    # Longest-match state detection (prevents "up" matching "support")
    for state_key in sorted(_STATE_MAP, key=len, reverse=True):
        if re.search(r"\b" + re.escape(state_key) + r"\b", q_lower):
            filters["state"] = _STATE_MAP[state_key]
            break

    # Beneficiary flags
    for payload_key, patterns in _BENEFICIARY_PATTERNS:
        for pat in patterns:
            if re.search(pat, q_lower):
                filters[payload_key] = True
                break

    return filters


def _build_qdrant_filter(meta: dict) -> dict | None:
    """
    Convert extracted metadata dict → Qdrant REST API filter JSON.
    Beneficiary booleans go in MUST (hard filter).
    State goes in MUST as an ANY match that always includes Central/national
    labels so schemes like PM Kisan are never excluded for state queries.
    Returns None when no filters apply (full collection search).
    """
    if not meta:
        return None
    must: list = []

    # Hard filters: beneficiary flags
    for key in ("is_for_women", "is_for_farmers", "is_for_disabled",
                "is_for_sc_st", "is_for_students"):
        if meta.get(key):
            must.append({"key": key, "match": {"value": True}})

    # State filter: user's state + all national/central labels (always inclusive)
    if state := meta.get("state"):
        state_values = [state] + sorted(_NATIONAL_STATE_LABELS)
        must.append({"key": "state_or_ut", "match": {"any": state_values}})

    return {"must": must} if must else None


# ── AWS clients (reused across warm invocations)
_bedrock    = None
_transcribe = None
_s3         = None
_sm_runtime = None

# ── Fastembed model singletons (loaded once per container, not per call) ───
_dense_emb  = None
_sparse_emb = None

def bedrock():
    global _bedrock
    if not _bedrock:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock

def transcribe_client():
    global _transcribe
    if not _transcribe:
        _transcribe = boto3.client("transcribe", region_name=AWS_REGION)
    return _transcribe

def s3_client():
    global _s3
    if not _s3:
        _s3 = boto3.client("s3", region_name=AWS_REGION)
    return _s3

def sm_runtime():
    global _sm_runtime
    if not _sm_runtime:
        _sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    return _sm_runtime


# ── Embedding ─────────────────────────────────────────────────────────────

def _get_dense_emb():
    """Singleton dense embedder — loaded once per Lambda container lifetime."""
    global _dense_emb
    if _dense_emb is None:
        from fastembed import TextEmbedding
        cache = os.environ.get("FASTEMBED_CACHE_DIR", "/var/task/fastembed_cache")
        _dense_emb = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", cache_dir=cache)
    return _dense_emb


def _get_sparse_emb():
    """Singleton sparse embedder — loaded once per Lambda container lifetime."""
    global _sparse_emb
    if _sparse_emb is None:
        from fastembed import SparseTextEmbedding
        cache = os.environ.get("FASTEMBED_CACHE_DIR", "/var/task/fastembed_cache")
        _sparse_emb = SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=cache)
    return _sparse_emb


def embed_query(text: str) -> list[float]:
    """Dense embedding — reuses the container-level singleton (no reload cost)."""
    return list(_get_dense_emb().embed([text]))[0].tolist()


def embed_sparse(text: str) -> dict:
    """BM25 sparse embedding — reuses the container-level singleton."""
    sp = list(_get_sparse_emb().embed([text]))[0]
    return {"indices": sp.indices.tolist(), "values": sp.values.tolist()}



# ── Qdrant Cloud Query (direct REST API — no qdrant-client package needed) ─

def qdrant_hybrid_search(query: str) -> tuple[list[dict], dict]:
    """Hybrid search via Qdrant Cloud REST API with metadata pre-filtering.

    Returns (chunks, meta) where:
      chunks — list of dicts with text, scheme_id, state_or_ut, scheme_category, etc.
      meta   — extracted filter dict (passed downstream to reranker / generator)
    """
    # 1. Extract metadata filters from the query (regex, ~0ms)
    meta = _extract_metadata_filters(query)
    qdrant_filter = _build_qdrant_filter(meta)

    # 2. Expand query for better BM25 sparse recall
    expanded_query = _expand_query(query)

    dense_result: list = [None]
    sparse_result: list = [None]

    def _dense():  dense_result[0] = embed_query(expanded_query)
    def _sparse(): sparse_result[0] = embed_sparse(expanded_query)

    t1 = threading.Thread(target=_dense)
    t2 = threading.Thread(target=_sparse)
    t1.start(); t2.start()
    t1.join();  t2.join()

    dense_vec  = dense_result[0]
    sparse_vec = sparse_result[0]

    # 3. Build Qdrant prefetch payload with filter applied in both arms
    prefetch_dense  = {"query": dense_vec,  "using": "dense",  "limit": CANDIDATE_K}
    prefetch_sparse = {"query": sparse_vec, "using": "sparse", "limit": CANDIDATE_K}
    if qdrant_filter:
        prefetch_dense["filter"]  = qdrant_filter
        prefetch_sparse["filter"] = qdrant_filter

    payload = {
        "prefetch": [prefetch_dense, prefetch_sparse],
        "query": {"fusion": "rrf"},
        "limit": CANDIDATE_K,
        "with_payload": True,
    }
    if qdrant_filter:
        payload["filter"] = qdrant_filter

    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/query"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # 4. Extract full payload (not just text + scheme_id)
    chunks = []
    for point in data.get("result", {}).get("points", []):
        p = point.get("payload", {})
        chunks.append({
            "text":            p.get("text", ""),
            "scheme_id":       p.get("scheme_id", ""),
            "scheme_name":     p.get("scheme_name", ""),
            "state_or_ut":     p.get("state_or_ut", ""),
            "scheme_category": p.get("scheme_category", ""),
            "chunk_type":      p.get("chunk_type", ""),
            "is_for_sc_st":    p.get("is_for_sc_st", False),
            "is_for_students": p.get("is_for_students", False),
            "is_for_women":    p.get("is_for_women", False),
            "is_for_farmers":  p.get("is_for_farmers", False),
            "is_for_disabled": p.get("is_for_disabled", False),
            "score":           point.get("score", 0.0),
        })

    filter_desc = (
        f"state={meta.get('state', 'any')} "
        + " ".join(k for k in meta if k != "state" and meta[k])
        if meta else "none"
    )
    print(f"[qdrant] {len(chunks)} chunks | filter: {filter_desc}")
    return chunks, meta


# ── Azure Cohere Reranker ────────────────────────────────────────────────────

def rerank_chunks(query: str, chunks: list[dict], meta: dict | None = None, top_k: int = RERANK_K) -> list[dict]:
    """Rerank using Cohere rerank-v4.0-fast on Azure AI Foundry.
    Builds an enriched rerank query from extracted metadata so the reranker
    knows the user's state / beneficiary constraints explicitly.
    Falls back to RRF order on any error so the pipeline never breaks.
    """
    if not chunks or len(chunks) <= top_k:
        return chunks[:top_k]
    if not AZURE_RERANK_URL or not AZURE_RERANK_KEY:
        print("[rerank] Azure env vars not set, skipping rerank")
        return chunks[:top_k]

    # Build a richer rerank query that makes constraints explicit
    constraints: list[str] = []
    if meta:
        if state := meta.get("state"):
            constraints.append(f"available in {state} or nationally")
        if meta.get("is_for_sc_st"):
            constraints.append("for SC/ST beneficiaries")
        if meta.get("is_for_students"):
            constraints.append("for students")
        if meta.get("is_for_women"):
            constraints.append("for women")
        if meta.get("is_for_farmers"):
            constraints.append("for farmers")
        if meta.get("is_for_disabled"):
            constraints.append("for persons with disability")
    rerank_query = (
        f"{query}. Requirements: {', '.join(constraints)}"
        if constraints else query
    )

    try:
        docs = [c["text"][:2000] for c in chunks]
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                AZURE_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {AZURE_RERANK_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": "rerank-v4-fast", "query": rerank_query,
                      "documents": docs, "top_n": top_k},
            )
            resp.raise_for_status()
        result = resp.json()
        ranked = sorted(result["results"], key=lambda r: r["relevance_score"], reverse=True)
        reranked = [chunks[r["index"]] for r in ranked]
        print(f"[rerank] {len(reranked)} chunks after Cohere rerank (query: '{rerank_query[:80]}')")
        return reranked
    except Exception as e:
        print(f"[rerank] failed ({e}), falling back to RRF order")
        return chunks[:top_k]


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate an English answer given an English query and English context chunks."""
    context_parts = []
    for i, c in enumerate(chunks):
        state   = c.get("state_or_ut", "") or "Central/National"
        cat     = c.get("scheme_category", "")
        name    = c.get("scheme_name", c["scheme_id"])
        header  = f"[{i+1}] {name} | State: {state} | Category: {cat}"
        context_parts.append(f"{header}\n{c['text']}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant for Indian government schemes.
Answer the question using ONLY the context below. Follow these rules strictly:

1. Recommend the single BEST scheme that most closely matches the user's situation.
2. Include: scheme name, key benefit (amount or type), and one eligibility point.
3. If the user mentions they are currently studying in India, do NOT recommend overseas
   or foreign study scholarships unless the user explicitly asked about studying abroad.
4. If the user mentions a specific state, prefer state-specific schemes over central ones.
5. If no scheme clearly matches, say: "I couldn't find an exact match. Please visit myscheme.gov.in or call 14555."
6. Keep your TOTAL answer under 100 words. Use simple language suitable for spoken voice.

Context:
{context}

Question: {query}

Answer:"""

    resp = bedrock().converse(
        modelId=LLM_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 400},
    )
    return resp["output"]["message"]["content"][0]["text"]


# ── AWS Transcribe ────────────────────────────────────────────────────────

def transcribe_audio(audio_url: str, call_sid: str) -> str:
    """Download Twilio audio → S3 → AWS Transcribe → text."""
    # Download audio from Twilio with basic auth
    auth = base64.b64encode(f"{TWILIO_SID}:{TWILIO_TOKEN}".encode()).decode()
    req = urllib.request.Request(
        f"{audio_url}.wav",
        headers={"Authorization": f"Basic {auth}"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        audio_bytes = r.read()

    # Upload to S3
    s3_key = f"voice-tmp/{call_sid}.wav"
    s3_client().put_object(
        Bucket=S3_BUCKET, Key=s3_key,
        Body=audio_bytes, ContentType="audio/wav"
    )
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

    # Start transcription
    job_name = f"lambda-rag-{call_sid}-{int(time.time())}"
    transcribe_client().start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        MediaFormat="wav",
        LanguageCode="en-IN",
    )

    # Poll (max 60s for short voice clips)
    for _ in range(30):
        status = transcribe_client().get_transcription_job(
            TranscriptionJobName=job_name
        )["TranscriptionJob"]
        if status["TranscriptionJobStatus"] == "COMPLETED":
            uri = status["Transcript"]["TranscriptFileUri"]
            with httpx.Client(timeout=10) as c:
                data = c.get(uri).json()
            return data["results"]["transcripts"][0]["transcript"]
        elif status["TranscriptionJobStatus"] == "FAILED":
            print("[transcribe] FAILED:", status)
            return ""
        time.sleep(2)

    return ""


# ── Azure Speech (TTS) ────────────────────────────────────────────────────
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")

AZURE_VOICE_MAP = {
    "en-IN": ("en-IN", "en-IN-NeerjaNeural"),
    "hi-IN": ("hi-IN", "hi-IN-SwaraNeural"),
    "te-IN": ("te-IN", "te-IN-ShrutiNeural"),
    "ta-IN": ("ta-IN", "ta-IN-PallaveNeural"),
    "kn-IN": ("kn-IN", "kn-IN-SapnaNeural"),
    "ml-IN": ("ml-IN", "ml-IN-SobhanaNeural"),
    "mr-IN": ("mr-IN", "mr-IN-AarohiNeural"),
    "bn-IN": ("bn-IN", "bn-IN-TanishaaNeural"),
    "gu-IN": ("gu-IN", "gu-IN-DhwaniNeural"),
}

def synthesize_speech_azure(text: str, message_sid: str, lang_code: str = "en-IN") -> str:
    """Convert AI text to an MP3 voice note using Azure Speech REST API."""
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY environment variable is not set.")

    xml_lang, voice_name = AZURE_VOICE_MAP.get(lang_code, ("en-IN", "en-IN-NeerjaNeural"))
    print(f"[TTS] Azure Speech: lang={lang_code}, voice={voice_name}")

    safe_text = text[:3000].replace("&", "and").replace("<", "").replace(">", "")
    ssml = f"""<speak version='1.0' xml:lang='{xml_lang}'>
    <voice name='{voice_name}'>{safe_text}</voice>
</speak>"""

    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
        "User-Agent": "GovSchemes-VoiceBot/1.0",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, headers=headers, content=ssml.encode("utf-8"))
    resp.raise_for_status()
    audio_bytes = resp.content

    s3_key = f"wa-out/{message_sid}.mp3"
    s3_client().put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=audio_bytes,
        ContentType="audio/mpeg"
    )

    presigned_url = s3_client().generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': S3_BUCKET, 'Key': s3_key},
        ExpiresIn=3600
    )
    return presigned_url

# ── TwiML Helpers ─────────────────────────────────────────────────────────

def twiml_greet() -> str:
    """TwiML to greet caller and collect speech."""
    # Since it's a synchronous prompt, we can't easily run Azure TTS on-the-fly 
    # without a webhook transition. We'll use Twilio's basic Polly just for the initial greeting
    # since it's hardcoded text, or we can use generic Twilio voice.
    return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" action="/prod/voice/answer" method="POST"
          language="en-IN" speechTimeout="auto" speechModel="phone_call"
          enhanced="true">
    <Say voice="Polly.Aditi" language="en-IN">
      Welcome to the Government Schemes helpline.
      Please speak your question now.
    </Say>
  </Gather>
  <Say voice="Polly.Aditi" language="en-IN">
    We did not hear a question. Please call again.
  </Say>
</Response>"""


def twiml_answer(audio_url: str) -> str:
    """TwiML to play the generated Azure audio back to the caller."""
    audio_url = audio_url.replace("&", "&amp;")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{audio_url}</Play>
</Response>"""


def twiml_error() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Aditi" language="en-IN">
    Sorry, I could not understand your question. Please try again.
  </Say>
</Response>"""


# ── Main Lambda Handler ───────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Handles two routes via API Gateway:
      POST /voice/incoming  → greet + record
      POST /voice/answer    → transcribe + retrieve + respond
    """
    path = event.get("rawPath") or event.get("path", "")
    body_raw = event.get("body", "")
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode()

    # Parse URL-encoded Twilio form body
    params = dict(urllib.parse.parse_qsl(body_raw))
    print(f"[lambda] path={path} params_keys={list(params.keys())}")

    # ── Route: Greet incoming call ────────────────────────────────────────
    if path.endswith("/voice/incoming"):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/xml"},
            "body": twiml_greet(),
        }

    # ── Route: Process recording + answer ────────────────────────────────
    if path.endswith("/voice/answer"):
        call_sid = params.get("CallSid", "unknown")

        # Fast path: Twilio <Gather speech> already transcribed the audio
        query = params.get("SpeechResult", "").strip()

        if not query:
            # Slow fallback: <Record> webhook — download + Transcribe + poll (~20s)
            recording_url = params.get("RecordingUrl", "")
            if recording_url:
                query = transcribe_audio(recording_url, call_sid)

        print(f"[lambda] Transcribed/SpeechResult: {query!r}")

        if not query.strip():
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/xml"},
                "body": twiml_error(),
            }

        # 2. Retrieve from Qdrant Cloud
        chunks, meta = qdrant_hybrid_search(query)
        chunks = rerank_chunks(query, chunks, meta=meta)

        # 3. Generate answer
        if chunks:
            answer = generate_answer(query, chunks)
        else:
            answer = (
                "I could not find a relevant government scheme for your question. "
                "Please visit myscheme.gov.in or call 1800-11-1555 for help."
            )

        print(f"[lambda] Answer: {answer[:100]}...")

        # Detect the user's language from the transcript script so TTS replies in the same language
        lang_code = _detect_lang_from_script(query, fallback="en-IN")
        print(f"[lambda] Detected lang for TTS: {lang_code}")

        # 4. Synthesize speech using Azure
        try:
            audio_url = synthesize_speech_azure(answer, call_sid, lang_code=lang_code)
            body_xml = twiml_answer(audio_url)
        except Exception as e:
            print(f"[lambda] TTS Error: {e}")
            body_xml = twiml_error()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/xml"},
            "body": body_xml,
        }

    # ── Debug/Query endpoint (called by existing Lambda via invoke) ───────
    if path.endswith("/debug/query"):
        try:
            body_str = event.get("body", "{}")
            if event.get("isBase64Encoded"):
                body_str = base64.b64decode(body_str).decode()
            body_json = json.loads(body_str) if body_str else {}
            query = body_json.get("query", "").strip()
            if not query:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "Missing 'query' field in request body"}),
                }
            # Query arrives in English; retrieve and answer in English
            chunks, meta = qdrant_hybrid_search(query)
            chunks  = rerank_chunks(query, chunks, meta=meta)
            answer  = generate_answer(query, chunks) if chunks else (
                "I could not find a relevant government scheme for your question. "
                "Please visit myscheme.gov.in or call 1800-11-1555 for help."
            )
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "answer":     answer,
                    "num_chunks": len(chunks),
                    "filters":    meta,
                    "contexts": [
                        {
                            "scheme_name":     c.get("scheme_name", c["scheme_id"]),
                            "state_or_ut":     c.get("state_or_ut", ""),
                            "scheme_category": c.get("scheme_category", ""),
                            "text":            c["text"],
                        }
                        for c in chunks
                    ],
                }),
            }
        except Exception as e:
            print(f"[debug/query] ERROR: {e}")
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    # ── Health check ──────────────────────────────────────────────────────
    if path.endswith("/health"):
        # Quick Qdrant Cloud ping
        try:
            url = f"{QDRANT_URL}/collections/{COLLECTION}"
            with httpx.Client(timeout=5) as c:
                r = c.get(url, headers={"api-key": QDRANT_API_KEY})
            pts = r.json()["result"]["points_count"]
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "ok", "points": pts}),
            }
        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 404, "body": "Not found"}
