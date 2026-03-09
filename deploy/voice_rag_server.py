"""
voice_rag_server.py
--------------------
Twilio Voice → AWS Transcribe → Qdrant Hybrid RAG → Bedrock → Twilio TTS

Flow:
  1. User calls Twilio number
  2. Twilio records voice message and hits /voice/incoming
  3. Recording webhook hits /voice/transcribe with audio URL
  4. AWS Transcribe converts audio → text
  5. Qdrant hybrid retrieval finds relevant scheme chunks
  6. Bedrock Nova generates a concise answer
  7. Server returns TwiML with <Say> (TTS) response to caller

Run:
  uvicorn voice_rag_server:app --host 0.0.0.0 --port 8000

Expose via ngrok for local dev:
  ngrok http 8000
  Set Twilio webhook → https://<ngrok-url>/voice/incoming
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Prefetch, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding
from twilio.twiml.voice_response import VoiceResponse, Record, Gather
from twilio.rest import Client as TwilioClient

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="GovSchemes Voice RAG", version="1.0.0")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_LOCAL_PATH = os.getenv("QDRANT_PATH", "qdrant_hybrid_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "schemes_hybrid")
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
CANDIDATE_K = 20
TOP_K = 5

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "eu.amazon.nova-lite-v1:0")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# S3 bucket for Transcribe (needs a bucket in same region)
TRANSCRIBE_S3_BUCKET = os.getenv("BEDROCK_S3_BUCKET", "government-scheme")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None
_dense_emb: TextEmbedding | None = None
_sparse_emb: SparseTextEmbedding | None = None
_bedrock_client = None
_transcribe_client = None
_s3_client = None
_twilio_client: TwilioClient | None = None


def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        cloud_url = os.getenv("QDRANT_CLOUD_URL")
        cloud_key = os.getenv("QDRANT_CLOUD_API_KEY")
        if cloud_url and cloud_key:
            log.info("Connecting to Qdrant Cloud: %s", cloud_url)
            _qdrant_client = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=30)
        else:
            log.info("Using local Qdrant at: %s", QDRANT_LOCAL_PATH)
            _qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
    return _qdrant_client


def get_dense_embedder():
    global _dense_emb
    if _dense_emb is None:
        log.info("Loading dense embedder: %s", DENSE_MODEL)
        # Auto-detect GPU
        try:
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        except ImportError:
            providers = ["CPUExecutionProvider"]
        _dense_emb = TextEmbedding(model_name=DENSE_MODEL, providers=providers)
        log.info("Dense embedder loaded (providers: %s)", providers)
    return _dense_emb


def get_sparse_embedder():
    global _sparse_emb
    if _sparse_emb is None:
        _sparse_emb = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_emb


def get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def get_transcribe():
    global _transcribe_client
    if _transcribe_client is None:
        _transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
    return _transcribe_client


def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def get_twilio():
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


# ---------------------------------------------------------------------------
# Warm up models on startup (prevents first-call hang)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def warmup():
    log.info("Warming up models...")
    get_dense_embedder()
    get_sparse_embedder()
    get_qdrant()
    log.info("Models ready.")


# ---------------------------------------------------------------------------
# Core: Encoding
# ---------------------------------------------------------------------------

def encode_dense(text: str) -> list[float]:
    return list(get_dense_embedder().embed([text]))[0].tolist()


def encode_sparse(text: str) -> SparseVector:
    sp = list(get_sparse_embedder().embed([text]))[0]
    return SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())


# ---------------------------------------------------------------------------
# Core: Hybrid Retrieval
# ---------------------------------------------------------------------------

def hybrid_retrieve(query: str) -> list[dict]:
    """Hybrid dense+sparse retrieval with RRF fusion."""
    dense_vec = encode_dense(query)
    sparse_vec = encode_sparse(query)

    results = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=CANDIDATE_K),
            Prefetch(query=sparse_vec, using="sparse", limit=CANDIDATE_K),
        ],
        query=FusionQuery(fusion=FusionQuery.Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        chunks.append({
            "text": point.payload.get("text", ""),
            "scheme_id": point.payload.get("scheme_id", ""),
            "score": point.score or 0.0,
        })

    log.info("Retrieved %d chunks for query: %s...", len(chunks), query[:60])
    return chunks


# ---------------------------------------------------------------------------
# Core: Answer Generation (Bedrock)
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate a concise spoken answer using Bedrock Nova."""
    context = "\n\n".join(
        [f"Scheme: {c['scheme_id']}\n{c['text']}" for c in chunks]
    )

    prompt = f"""You are a helpful assistant for Indian government schemes. 
A citizen has asked a question via voice call. Answer in simple, clear language 
suitable for text-to-speech. Keep it under 100 words. If you don't know, say so.

Context:
{context}

Question: {query}

Answer (keep it short and spoken-friendly):"""

    response = get_bedrock().converse(
        modelId=BEDROCK_LLM_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 256},
    )

    answer = response["output"]["message"]["content"][0]["text"]
    log.info("Generated answer (%d chars)", len(answer))
    return answer


# ---------------------------------------------------------------------------
# Core: Voice Transcription (AWS Transcribe)
# ---------------------------------------------------------------------------

def transcribe_audio_url(audio_url: str, call_sid: str) -> str:
    """
    Download audio from Twilio URL and transcribe with AWS Transcribe.
    Returns transcribed text.
    """
    transcribe = get_transcribe()
    s3 = get_s3()

    # 1. Download audio from Twilio (requires auth)
    log.info("Downloading audio from Twilio: %s", audio_url)
    resp = httpx.get(
        audio_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        timeout=30.0,
    )
    resp.raise_for_status()

    # 2. Upload to S3 (Transcribe needs S3 URI)
    s3_key = f"voice-transcribe/{call_sid}.wav"
    s3.put_object(
        Bucket=TRANSCRIBE_S3_BUCKET,
        Key=s3_key,
        Body=resp.content,
        ContentType="audio/wav",
    )
    s3_uri = f"s3://{TRANSCRIBE_S3_BUCKET}/{s3_key}"
    log.info("Uploaded audio to S3: %s", s3_uri)

    # 3. Start transcription job
    job_name = f"voice-rag-{call_sid}-{int(time.time())}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        MediaFormat="wav",
        LanguageCode="en-IN",  # Indian English; change to "hi-IN" for Hindi
        Settings={"ShowSpeakerLabels": False},
    )

    # 4. Poll until complete (max 60s for short voice messages)
    for attempt in range(30):
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status == "COMPLETED":
            transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            # Download transcript JSON
            transcript_resp = httpx.get(transcript_uri, timeout=10.0)
            data = transcript_resp.json()
            text = data["results"]["transcripts"][0]["transcript"]
            log.info("Transcription: %s", text)
            return text
        elif job_status == "FAILED":
            log.error("Transcription failed: %s", status)
            return ""
        time.sleep(2)

    log.error("Transcription timed out for job: %s", job_name)
    return ""


# ---------------------------------------------------------------------------
# Twilio Webhook Endpoints
# ---------------------------------------------------------------------------

@app.post("/voice/incoming", response_class=PlainTextResponse)
async def voice_incoming(request: Request):
    """
    Step 1: User calls → Twilio hits this endpoint.
    We greet the user and start recording their question.
    """
    response = VoiceResponse()
    response.say(
        "Welcome to the Government Schemes helpline. "
        "Please speak your question after the beep, then press any key or stay silent for 3 seconds.",
        voice="Polly.Aditi",  # Indian English voice
        language="en-IN",
    )
    # Record up to 30 seconds; Twilio will POST to /voice/recording_ready
    response.record(
        max_length=30,
        timeout=3,
        transcribe=False,  # We use AWS Transcribe instead
        action="/voice/recording_ready",
        method="POST",
        play_beep=True,
    )

    log.info("Incoming call — greeting sent")
    return PlainTextResponse(str(response), media_type="application/xml")


@app.post("/voice/recording_ready", response_class=PlainTextResponse)
async def voice_recording_ready(
    request: Request,
    background_tasks: BackgroundTasks,
    RecordingUrl: str = Form(...),
    CallSid: str = Form(...),
    RecordingDuration: str = Form(""),
):
    """
    Step 2: Recording is ready.
    Transcribe + retrieve + generate answer, then respond with TTS.
    """
    log.info("Recording ready: CallSid=%s Duration=%ss", CallSid, RecordingDuration)
    audio_url = f"{RecordingUrl}.wav"

    # Transcribe audio → text
    query_text = transcribe_audio_url(audio_url, CallSid)

    if not query_text.strip():
        response = VoiceResponse()
        response.say(
            "Sorry, I couldn't understand your question. Please try calling again.",
            voice="Polly.Aditi",
            language="en-IN",
        )
        return PlainTextResponse(str(response), media_type="application/xml")

    log.info("Query: %s", query_text)

    # Retrieve from Qdrant
    chunks = hybrid_retrieve(query_text)

    # Generate answer
    if chunks:
        answer = generate_answer(query_text, chunks)
    else:
        answer = (
            "I'm sorry, I couldn't find any relevant government scheme for your query. "
            "Please try rephrasing or call the helpline at 1800-11-1555."
        )

    # Respond with TTS
    response = VoiceResponse()
    response.say(answer, voice="Polly.Aditi", language="en-IN")
    response.say(
        "Thank you for using the Government Schemes helpline. Goodbye!",
        voice="Polly.Aditi",
        language="en-IN",
    )

    log.info("Responded to CallSid=%s", CallSid)
    return PlainTextResponse(str(response), media_type="application/xml")


# ---------------------------------------------------------------------------
# WhatsApp Voice Note Endpoint (Twilio WhatsApp)
# ---------------------------------------------------------------------------

@app.post("/whatsapp/incoming", response_class=PlainTextResponse)
async def whatsapp_incoming(
    request: Request,
    Body: str = Form(""),
    MediaUrl0: str = Form(""),
    MediaContentType0: str = Form(""),
    From: str = Form(""),
    To: str = Form(""),
    MessageSid: str = Form(""),
):
    """
    Handles WhatsApp messages (text or voice notes).
    Twilio webhook for WhatsApp incoming messages.
    """
    from twilio.twiml.messaging_response import MessagingResponse

    log.info("WhatsApp from=%s type=%s", From, MediaContentType0)

    query_text = ""

    # Voice note (audio/ogg, audio/mpeg, etc.)
    if MediaUrl0 and "audio" in MediaContentType0:
        log.info("Transcribing WhatsApp voice note: %s", MediaUrl0)
        query_text = transcribe_audio_url(MediaUrl0, MessageSid)
    elif Body.strip():
        # Text message — treat directly as query
        query_text = Body.strip()
        log.info("Text query: %s", query_text)

    response = MessagingResponse()

    if not query_text:
        response.message("Sorry, I couldn't understand your message. Please send a text or voice note.")
        return PlainTextResponse(str(response), media_type="application/xml")

    # Retrieve + generate
    chunks = hybrid_retrieve(query_text)
    if chunks:
        answer = generate_answer(query_text, chunks)
    else:
        answer = "Sorry, no relevant scheme found. Try rephrasing your question."

    response.message(f"🏛️ *GovSchemes Answer:*\n\n{answer}")
    return PlainTextResponse(str(response), media_type="application/xml")


# ---------------------------------------------------------------------------
# Debug / Health endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "collection": COLLECTION_NAME, "qdrant": QDRANT_LOCAL_PATH}


@app.post("/debug/query")
async def debug_query(request: Request):
    """Test retrieval directly with a text query (no voice)."""
    body = await request.json()
    query = body.get("query", "")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)

    chunks = hybrid_retrieve(query)
    answer = generate_answer(query, chunks) if chunks else "No results found."
    return {
        "query": query,
        "answer": answer,
        "chunks_count": len(chunks),
        "top_chunk": chunks[0] if chunks else None,
    }
