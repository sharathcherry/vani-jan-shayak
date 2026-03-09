# Vani Jan Sahayak — System Architecture

A multilingual WhatsApp + Voice bot that answers Indian government scheme queries using RAG (Retrieval-Augmented Generation). Users send a voice note or text in any Indian language; the bot replies in the same language.

---

## High-Level Flow

```
User (WhatsApp voice note or text)
        │
        ▼
  Twilio Webhook
        │
        ▼
┌───────────────────────────────┐
│  Lambda 1: vani-jan-webhook   │  (main.py)
│  - Download audio             │
│  - Azure STT → transcript     │
│  - Language detection         │
│  - Intent guard               │
│  - Translate to English       │
│  - Call Lambda 2 for RAG      │
│  - Answer cache check         │
│  - Translate answer back      │
│  - Azure TTS → voice note     │
│  - Reply via Twilio           │
└───────────────────────────────┘
        │
        │ invoke (RequestResponse)
        ▼
┌───────────────────────────────────┐
│  Lambda 2: gov-schemes-voice-rag  │  (lambda_voice_rag/lambda_function.py)
│  - Embed query (dense + sparse)   │
│  - Qdrant hybrid search           │
│  - Cohere reranker (Azure)        │
│  - Bedrock LLM → English answer   │
└───────────────────────────────────┘
```

---

## Lambda 1 — `vani-jan-webhook` (`main.py`)

**Runtime:** Python 3.11 | **Region:** eu-north-1 | **Deployed as:** zip (`vani_webhook.zip`)

This is the **entry point** for all messages from Twilio. It owns the entire user-facing conversation pipeline.

### What it does

| Step | Description |
|------|-------------|
| **Receive** | Twilio POSTs the WhatsApp message as a URL-encoded form body. Parsed by `parse_body()`. |
| **Voice Path** | If a `MediaUrl0` exists (voice note), it immediately acks Twilio ("Got your voice note...") and re-invokes itself asynchronously (`InvocationType="Event"`) so the 15-second Twilio timeout is never hit. The async half runs `_handle_voice_async()`. |
| **Text Path** | If the message is typed text, it goes straight to language detect → translate → RAG → translate back → reply. |
| **Join keyword** | If user sends "join ..." or "hi/hello/start/help", returns the branded Vani Jan Sahayak welcome message. |

### Voice pipeline inside `_handle_voice_async()`

1. **Parallel A** — Session S3 read and Twilio audio download happen in parallel (saves ~1–2s).
2. **Azure STT** (`_azure_stt()`) — Converts audio to WAV via ffmpeg, sends to Azure Speech REST API with multi-language LID. Returns transcript + detected language.
3. **Language detection** — 3-layer strategy (see Language Detection section below).
4. **Intent guard** (`_has_scheme_intent()`) — If the English translation contains none of the scheme-related keywords, sends a clarification message and stops. Prevents nonsense RAG lookups.
5. **RAG answer** (`get_rag_answer()`) — Checks cache first (L1 in-memory, L2 S3), then calls Lambda 2.
6. **Parallel B** — Session S3 write and answer translation happen in parallel (saves ~1s).
7. **Azure TTS** (`synthesize_speech()`) — Converts the native-language answer to MP3, uploads to S3, sends presigned URL back to Twilio as a voice note.
8. **Links** — Any URLs extracted from the English answer are sent as a separate text message (not read by TTS).

---

## Lambda 2 — `gov-schemes-voice-rag` (`lambda_voice_rag/lambda_function.py`)

**Runtime:** Python 3.11 | **Region:** eu-north-1 | **Deployed as:** Docker container on ECR

This is the **RAG engine**. Lambda 1 invokes it synchronously over `/debug/query`. It also has routes for Twilio Voice phone calls (`/voice/incoming`, `/voice/answer`).

### What it does

| Step | Description |
|------|-------------|
| **Embed** | Generates a dense vector (BAAI/bge-large-en-v1.5 via fastembed) and a sparse BM25 vector in parallel threads. |
| **Qdrant hybrid search** | Sends prefetch requests for both vectors to Qdrant Cloud, fused with RRF (Reciprocal Rank Fusion). Retrieves top 20 candidates. |
| **Rerank** | Sends the 20 candidates to Cohere rerank-v4-fast on Azure AI Foundry. Returns top 5 most relevant chunks. Falls back to RRF order on any error. |
| **Generate answer** | Sends the top 5 chunks + English query to Amazon Bedrock (Nova Lite). Returns a ≤80 word answer in English. |
| **TTS (voice calls)** | For Twilio voice phone calls, synthesizes the answer to MP3 using Azure Speech TTS and plays it back via TwiML. |

### Cold Start Optimization
The Docker image contains pre-baked fastembed model files at `/var/task/fastembed_cache`. On cold start, `_setup_model_cache()` copies them to `/tmp` (the only writable directory in Lambda). Warm containers skip this copy entirely.

---

## Azure Speech Services

Both Lambdas use **Azure Cognitive Services Speech** in the `centralindia` region.

| Use | Where | Function |
|-----|-------|----------|
| **STT (Speech-to-Text)** | Lambda 1 (`main.py`) | `_azure_stt()` — transcribes voice notes |
| **TTS (Text-to-Speech)** | Lambda 1 (`main.py`) | `synthesize_speech()` — WhatsApp voice replies |
| **TTS (Text-to-Speech)** | Lambda 2 (`lambda_voice_rag/lambda_function.py`) | `synthesize_speech_azure()` — Twilio phone call replies |

### Voice Map (AZURE_VOICE_MAP)

| Language | Voice | Style |
|----------|-------|-------|
| Telugu (`te-IN`) | `te-IN-ShrutiNeural` | #1 priority |
| Kannada (`kn-IN`) | `kn-IN-SapnaNeural` | Priority |
| Hindi (`hi-IN`) | `hi-IN-SwaraNeural` | Priority |
| English (`en-IN`) | `en-IN-NeerjaNeural` | Priority |
| Tamil (`ta-IN`) | `ta-IN-PallaveNeural` | |
| Malayalam (`ml-IN`) | `ml-IN-SobhanaNeural` | |
| Marathi (`mr-IN`) | `mr-IN-AarohiNeural` | |
| Bengali (`bn-IN`) | `bn-IN-TanishaaNeural` | |
| Gujarati (`gu-IN`) | `gu-IN-DhwaniNeural` | |
| Punjabi (`pa-IN`) | `pa-IN-VaaniNeural` | |
| Odia (`or-IN`) | `or-IN-SubhasiniNeural` | |
| Assamese (`as-IN`) | `as-IN-YashicaNeural` | |

TTS audio is uploaded to S3 and delivered to Twilio as a **presigned URL** (expires in 1 hour).

---

## Language Detection (3-Layer Strategy)

Runs in `_azure_stt()` in Lambda 1 every time a voice note arrives.

```
Layer 1 — Azure LID result
  Azure returns a detected language (e.g. "te-IN") alongside the transcript.
  If it's one of the 4 priority langs (te/kn/hi/en), trust it unconditionally.
  If it's different from the session hint, trust it (genuine override).

Layer 2 — Unicode script scan (_detect_lang_from_script)
  If Azure's result was unreliable (empty or echoed back the hint),
  scan the transcript characters for Indian Unicode ranges.
  Language with ≥15% character density wins.
  te-IN is checked first (index 0 in _SCRIPT_DETECT).

Layer 3 — Session fallback
  If transcript has no Indian script chars at all (Latin/English),
  trust Azure's result. If Azure has nothing, default to te-IN.
```

`_PRIORITY_LANGS = {"te-IN", "kn-IN", "hi-IN", "en-IN"}` — these four are always trusted from Azure LID even when they match the session hint, because Azure natively distinguishes them well.

**Language persistence:** After every successful query, the detected language is saved to the user's S3 session file as `LANG:{code}`. Next turn, Lambda 1 reads this and uses it as the Azure STT primary hint, so Azure outputs native script instead of romanized Latin.

---

## Qdrant Vector Database

**Hosted on:** Qdrant Cloud (`eu-central-1` region, AWS)
**Collection:** `schemes_hybrid`
**Accessed from:** Lambda 2 via direct REST API calls (no qdrant-client package needed)

### Search Strategy

- **Dense search** — BAAI/bge-large-en-v1.5 semantic embeddings (1024-dim)
- **Sparse search** — BM25 keyword embeddings (Qdrant/bm25 via fastembed)
- **Fusion** — RRF (Reciprocal Rank Fusion) across both results
- **Candidate limit** — Top 20 chunks retrieved, then reranked to top 5
- Dense and sparse embeddings are generated in **parallel threads** to cut latency

The database contains chunked text from Indian government scheme PDFs, indexed with both vector types for hybrid retrieval.

---

## Cohere Reranker (Azure AI Foundry)

**Model:** `rerank-v4-fast` (Cohere rerank-v4.0-fast)
**Hosted on:** Azure AI Foundry (configured via `AZURE_RERANK_URL` + `AZURE_RERANK_KEY` env vars)
**Where:** Lambda 2, function `rerank_chunks()`

Takes the 20 RRF-fused candidates from Qdrant and scores each chunk's relevance to the original query. Returns the top 5 most relevant. If Azure is unreachable or the env vars are missing, it silently falls back to the RRF ordering (pipeline never breaks).

---

## Answer Caching (2 Layers)

**Where:** Lambda 1 (`main.py`), function `get_rag_answer()`

Caching avoids hitting Lambda 2 (and Qdrant + Bedrock) for repeated questions.

| Layer | Storage | Speed | Lifetime |
|-------|---------|-------|----------|
| **L1 — In-memory LRU** | `_LRUCache` (OrderedDict, max 256 entries) | ~0 ms | Survives across warm invocations of the same container |
| **L2 — S3 persistent** | `wa-cache/{sha256_of_query}.txt` in `whatsapp-voice-messages` bucket | ~20–50 ms | 24 hours |
| **L3 — RAG Lambda** | Lambda 2 (Qdrant + Bedrock) | ~3–8 s | Not cached yet |

Cache key = SHA-256 of the normalized English query (lowercased, whitespace collapsed). An L2 hit is also promoted to L1 so subsequent in-container calls are instant.

---

## Session Storage

**Where:** S3 bucket `whatsapp-voice-messages`, key prefix `wa-sessions/`
**Managed by:** Lambda 1

Format of session file:
```
LANG:te-IN
Q: What scholarships are available for SC students?
A: The Post Matric Scholarship for SC students...
```

- `LANG:` line is read next turn to set the Azure STT primary language hint
- Q/A lines are stored but **not injected** into RAG queries (every query is standalone)
- Session expires after **3 hours** and is deleted on next access
- Default language for new/expired sessions: **`te-IN`**

---

## S3 Buckets

| Bucket | Region | Contents |
|--------|--------|----------|
| `whatsapp-voice-messages` | eu-north-1 | Incoming audio (`wa-in/`), sessions (`wa-sessions/`), answer cache (`wa-cache/`), RAG transcribe temp (`voice-tmp/`) |
| `whatsapp-voice-responses` | eu-north-1 | TTS output MP3 files (`wa-out/`) served to Twilio via presigned URL |
| `government-scheme` | eu-central-1 | Source PDFs, Lambda 2's transcribe temp files, TTS output for phone calls |

---

## Bedrock LLM

**Model:** `eu.amazon.nova-lite-v1:0` (Amazon Nova Lite)
**Region:** eu-central-1
**Used for:**
- Answering scheme queries (Lambda 2, `generate_answer()`)
- Translating queries and answers between Indian languages and English (Lambda 1, `translate_text()`)

---

## Deployment

| Component | Method | Command |
|-----------|--------|---------|
| Lambda 1 (webhook) | Zip file | `python build_lambda.py` → `aws lambda update-function-code --function-name vani-jan-webhook --zip-file fileb://vani_webhook.zip` |
| Lambda 2 (RAG) | Docker → ECR | `.\deploy_lambda.ps1` (includes `--provenance=false` for Lambda manifest compatibility) |
| ffmpeg | Lambda Layer | `ffmpeg_layer.zip` attached to Lambda 1 at `/opt/bin/ffmpeg` |

---

## Environment Variables Summary

### Lambda 1 (`vani-jan-webhook`)
| Variable | Purpose |
|----------|---------|
| `AZURE_SPEECH_KEY` | Azure Speech API key (STT + TTS) |
| `AZURE_SPEECH_REGION` | `centralindia` |
| `TWILIO_ACCOUNT_SID` | Twilio auth |
| `TWILIO_AUTH_TOKEN` | Twilio auth |
| `S3_VOICE_INPUT_BUCKET` | `whatsapp-voice-messages` |
| `S3_VOICE_OUTPUT_BUCKET` | `whatsapp-voice-responses` |

### Lambda 2 (`gov-schemes-voice-rag`)
| Variable | Purpose |
|----------|---------|
| `QDRANT_CLOUD_URL` | Qdrant Cloud endpoint |
| `QDRANT_CLOUD_API_KEY` | Qdrant auth |
| `QDRANT_COLLECTION` | `schemes_hybrid` |
| `BEDROCK_LLM_MODEL` | `eu.amazon.nova-lite-v1:0` |
| `AWS_REGION_NAME` | `eu-central-1` |
| `AZURE_SPEECH_KEY` | Azure TTS (phone call path) |
| `AZURE_SPEECH_REGION` | `centralindia` |
| `AZURE_RERANK_URL` | Azure AI Foundry Cohere rerank endpoint |
| `AZURE_RERANK_KEY` | Azure AI Foundry API key |
| `RERANK_K` | Number of chunks after rerank (default 5) |
| `TRANSCRIBE_S3_BUCKET` | `government-scheme` |
| `TWILIO_ACCOUNT_SID` | Twilio auth (phone call path) |
| `TWILIO_AUTH_TOKEN` | Twilio auth (phone call path) |
