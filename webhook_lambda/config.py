"""config.py — Centralised configuration for the Gov Schemes Voice Bot.

All environment variables, AWS clients, and shared constants live here.
Every other module imports from this file, so there is exactly one place
to change keys, bucket names, or region settings.
"""

import os
import boto3

# ---------------------------------------------------------------------------
# AWS / Infrastructure
# ---------------------------------------------------------------------------
REGION        = "eu-north-1"
S3_BUCKET_IN  = os.environ.get("S3_VOICE_INPUT_BUCKET",  "whatsapp-voice-messages")
S3_BUCKET_OUT = os.environ.get("S3_VOICE_OUTPUT_BUCKET", "whatsapp-voice-responses")
RAG_LAMBDA_NAME = "gov-schemes-voice-rag"

# AWS clients — module-level singletons, reused across warm Lambda invocations
s3            = boto3.client("s3",              region_name=REGION)
lambda_client = boto3.client("lambda",          region_name=REGION)
bedrock       = boto3.client("bedrock-runtime", region_name="eu-central-1")

# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------
TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN",  "")

# ---------------------------------------------------------------------------
# Azure Speech (STT + TTS)
# ---------------------------------------------------------------------------
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY",    "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")

# Azure Neural TTS voice for each supported Indian language
AZURE_VOICE_MAP: dict[str, tuple[str, str]] = {
    "en-IN": ("en-IN", "en-IN-NeerjaNeural"),
    "hi-IN": ("hi-IN", "hi-IN-SwaraNeural"),
    "te-IN": ("te-IN", "te-IN-ShrutiNeural"),
    "ta-IN": ("ta-IN", "ta-IN-PallaveNeural"),
    "kn-IN": ("kn-IN", "kn-IN-SapnaNeural"),
    "ml-IN": ("ml-IN", "ml-IN-SobhanaNeural"),
    "mr-IN": ("mr-IN", "mr-IN-AarohiNeural"),
    "bn-IN": ("bn-IN", "bn-IN-TanishaaNeural"),
    "gu-IN": ("gu-IN", "gu-IN-DhwaniNeural"),
    "pa-IN": ("pa-IN", "pa-IN-VaaniNeural"),
    "or-IN": ("or-IN", "or-IN-SubhasiniNeural"),
    "as-IN": ("as-IN", "as-IN-YashicaNeural"),
}

# ---------------------------------------------------------------------------
# TTL settings
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS = 3  * 60 * 60   # 3 hours  — per-user conversation context
CACHE_TTL_SECONDS   = 24 * 60 * 60   # 24 hours — RAG answer cache

# ---------------------------------------------------------------------------
# Language display names (used in Bedrock translation prompts)
# ---------------------------------------------------------------------------
LANG_NAMES: dict[str, str] = {
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
    "as-IN": "Assamese",
    "en-IN": "English",
}
