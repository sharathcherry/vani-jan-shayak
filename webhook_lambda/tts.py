"""tts.py — Text-to-Speech via Azure Neural TTS.

Converts the translated answer text into an MP3 voice note:
  1. Build an SSML payload with the appropriate Neural voice for the language
  2. POST to Azure Cognitive Services TTS endpoint
  3. Upload the MP3 to S3
  4. Return a 1-hour presigned URL that Twilio can attach to a WhatsApp message

Caching: synthesized MP3s are stored in S3 under wa-tts-cache/ keyed by
SHA-256(sanitised_text|lang_code). Identical text+language combinations
skip the Azure call (~1-2 s) and return a fresh presigned URL for the
already-stored audio.
"""

import hashlib
import time

import httpx

from config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_MAP,
    CACHE_TTL_SECONDS,
    S3_BUCKET_OUT,
    s3,
)

_TTS_CACHE_PREFIX = "wa-tts-cache"


def synthesize_speech(text: str, message_sid: str, lang_code: str = "en-IN") -> str:
    """Azure Neural TTS → S3 → presigned URL for Twilio media attachment.

    Args:
        text:        The answer text to synthesise (URLs already stripped).
        message_sid: Used as the S3 object key to keep audio files unique.
        lang_code:   BCP-47 language tag, e.g. "hi-IN".

    Returns:
        A 1-hour presigned S3 URL that Twilio can serve as a WhatsApp audio message.
    """
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY environment variable is not set.")

    xml_lang, voice_name = AZURE_VOICE_MAP.get(lang_code, ("en-IN", "en-IN-NeerjaNeural"))
    print(f"[TTS] Azure Speech: lang={lang_code}, voice={voice_name}")

    # Sanitise text for SSML — done first so the cache key matches actual content
    safe_text = text[:2000].replace("&", "and").replace("<", "").replace(">", "")

    # Check S3 TTS cache (same text + voice = identical audio)
    cache_key = hashlib.sha256(f"{safe_text}|{lang_code}".encode()).hexdigest()
    cache_s3_key = f"{_TTS_CACHE_PREFIX}/{cache_key}.mp3"
    try:
        head = s3.head_object(Bucket=S3_BUCKET_OUT, Key=cache_s3_key)
        age = time.time() - head["LastModified"].timestamp()
        if age <= CACHE_TTL_SECONDS:
            print(f"[TTS] S3 cache hit — skipping Azure synthesis")
            return s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET_OUT, "Key": cache_s3_key},
                ExpiresIn=3600,
            )
    except Exception:
        pass  # cache miss — fall through to Azure

    ssml = (
        f"<speak version='1.0' xml:lang='{xml_lang}'>"
        f"<voice name='{voice_name}'>{safe_text}</voice>"
        f"</speak>"
    )

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
    print(f"[TTS] Azure returned {len(audio_bytes):,} bytes of MP3")

    # Upload to S3 cache key and generate a presigned URL (valid 1 hour)
    s3.put_object(
        Bucket=S3_BUCKET_OUT,
        Key=cache_s3_key,
        Body=audio_bytes,
        ContentType="audio/mpeg",
    )
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET_OUT, "Key": cache_s3_key},
        ExpiresIn=3600,
    )
