"""stt.py — Speech-to-Text and audio utilities.

Handles the full audio ingestion pipeline:
  1. download_twilio_audio() — fetch audio from Twilio CDN
  2. convert_to_wav()        — transcode any format to 16 kHz WAV via ffmpeg
  3. detect_lang_from_script() — Unicode script scan for Indian language detection
  4. azure_stt()             — Azure Speech STT with 3-layer language detection

Language detection uses three layers in priority order:
  Layer 1 — Azure LID        (up to 4 candidate languages, REST API limit)
  Layer 2 — Unicode script   (definitive for Indian scripts, zero latency)
  Layer 3 — Fallback en-IN
"""

import base64
import os
import subprocess
import tempfile
import urllib.request

import httpx

from config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_MAP,
    TWILIO_SID,
    TWILIO_TOKEN,
)

# Unicode character ranges for each supported Indian script.
# Te-IN is listed first because it is the highest-priority language for
# this deployment (Telangana/AP focus). Checked in order; first script
# whose character density exceeds 15% wins.
_SCRIPT_DETECT = [
    ("te-IN", "\u0c00", "\u0c7f"),   # Telugu        ← #1 priority
    ("kn-IN", "\u0c80", "\u0cff"),   # Kannada
    ("hi-IN", "\u0900", "\u097f"),   # Devanagari (Hindi / Marathi)
    ("ta-IN", "\u0b80", "\u0bff"),   # Tamil
    ("ml-IN", "\u0d00", "\u0d7f"),   # Malayalam
    ("gu-IN", "\u0a80", "\u0aff"),   # Gujarati
    ("pa-IN", "\u0a00", "\u0a7f"),   # Gurmukhi (Punjabi)
    ("or-IN", "\u0b00", "\u0b7f"),   # Odia
    ("bn-IN", "\u0980", "\u09ff"),   # Bengali / Assamese
    ("as-IN", "\u0980", "\u09ff"),
]


class _NoAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Strip the Authorization header when Twilio CDN redirects.

    Twilio media URLs redirect to AWS S3. Sending credentials to S3
    causes a 400 error, so we strip auth on every redirect hop.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newreq = super().redirect_request(req, fp, code, msg, headers, newurl)
        if newreq:
            newreq.headers.pop("Authorization", None)
            newreq.headers.pop("authorization", None)
        return newreq


def download_twilio_audio(media_url: str) -> tuple[bytes, str]:
    """Download a voice note from Twilio CDN.

    Returns (audio_bytes, content_type).
    Authenticates with Basic auth and strips credentials on redirect.
    """
    auth_b64 = base64.b64encode(f"{TWILIO_SID}:{TWILIO_TOKEN}".encode()).decode()
    req = urllib.request.Request(media_url, headers={"Authorization": f"Basic {auth_b64}"})
    opener = urllib.request.build_opener(_NoAuthRedirectHandler)
    with opener.open(req, timeout=10) as r:
        data = r.read()
        content_type = r.headers.get("Content-Type", "audio/ogg; codecs=opus")
    print(f"[download] {len(data)} bytes, Content-Type: {content_type}")
    return data, content_type


def convert_to_wav(audio_bytes: bytes) -> bytes:
    """Transcode any audio format to 16 kHz mono WAV.

    Uses the ffmpeg binary bundled as a Lambda layer at /opt/bin/ffmpeg.
    Azure Speech STT requires 16 kHz mono PCM WAV for non-OGG inputs.
    """
    ffmpeg = "/opt/bin/ffmpeg"
    if not os.path.exists(ffmpeg):
        raise RuntimeError("ffmpeg layer not attached — /opt/bin/ffmpeg not found")

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False, dir="/tmp") as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", in_path, "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", "-f", "wav", out_path],
            capture_output=True,
            timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exit {result.returncode}: {result.stderr.decode()[:200]}"
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def detect_lang_from_script(text: str, fallback: str = "en-IN") -> str:
    """Identify language by Unicode script character density.

    Checks which script block accounts for ≥15% of non-whitespace
    characters. This is the ground-truth layer 2 fallback when Azure LID
    either returns empty or echoes the primary hint without confidence.
    """
    if not text:
        return fallback

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return fallback

    total = len(non_space)
    best_lang, best_count = fallback, 0
    seen: set[str] = set()

    for lang, lo, hi in _SCRIPT_DETECT:
        if lang in seen:
            continue
        seen.add(lang)
        count = sum(1 for c in non_space if lo <= c <= hi)
        if count > best_count and (count / total) >= 0.15:
            best_count, best_lang = count, lang

    return best_lang


def azure_stt(
    audio_bytes: bytes,
    content_type: str,
    preferred_lang: str = "te-IN",
) -> tuple[str, str]:
    """Azure Cognitive Services Speech-to-Text with automatic language detection.

    `preferred_lang` is the language detected in the user's previous turn,
    stored in the S3 session. It is used as the Azure LID primary hint so
    the correct script appears in the transcript on the first try.

    Returns (transcript, detected_lang_code).

    Detection priority:
      1. Azure LID confidently identifies a language different from the hint
      2. Unicode script scan on the returned transcript (Layer 2)
      3. Azure's result when transcript is Latin script
      4. Fallback to en-IN
    """
    try:
        # OGG/Opus can be sent directly; everything else must be transcoded to WAV.
        if "ogg" in content_type or "opus" in content_type or "wav" in content_type:
            audio_to_send    = audio_bytes
            stt_content_type = "audio/ogg; codecs=opus"
        else:
            print(f"[STT] Converting {content_type} -> WAV...")
            audio_to_send    = convert_to_wav(audio_bytes)
            stt_content_type = "audio/wav; codecs=audio/pcm; samplerate=16000"
            print(f"[STT] WAV size: {len(audio_to_send):,} bytes")

        # Azure LID REST API hard limit: maximum 4 candidate languages.
        # Priority covers the most common Indian languages for this use case.
        # Tamil, Malayalam, Marathi etc. are caught by Unicode script scan (Layer 2).
        _LID_PRIORITY = ["te-IN", "kn-IN", "hi-IN", "en-IN"]
        if preferred_lang not in _LID_PRIORITY and preferred_lang in AZURE_VOICE_MAP:
            # Rotate session language into slot 0 so Azure has it as a candidate
            _LID_PRIORITY = [preferred_lang] + _LID_PRIORITY[:3]

        lid_langs    = ",".join(_LID_PRIORITY)
        primary_lang = preferred_lang if preferred_lang in AZURE_VOICE_MAP else "te-IN"

        stt_url = (
            f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={primary_lang}&lid={lid_langs}&format=simple"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": stt_content_type,
        }

        with httpx.Client(timeout=12.0) as client:
            resp = client.post(stt_url, headers=headers, content=audio_to_send)

        result     = resp.json()
        status     = result.get("RecognitionStatus", "Unknown")
        transcript = result.get("DisplayText", "")
        pl         = result.get("PrimaryLanguage", {})
        azure_lang = pl.get("Language", "") if isinstance(pl, dict) else ""
        confidence = pl.get("Confidence", "") if isinstance(pl, dict) else ""

        print(
            f"[STT] status={status} azure_lang={azure_lang!r} "
            f"confidence={confidence!r} text='{transcript[:70]}'"
        )
        if not transcript:
            return "", "en-IN"

        # For the 4 priority languages (te/kn/hi/en), trust Azure even when it
        # matches the primary hint — these are well-supported and reliable.
        # For all other languages, only trust Azure if it actively overrides the hint.
        _PRIORITY_LANGS = {"te-IN", "kn-IN", "hi-IN", "en-IN"}
        if azure_lang and azure_lang in AZURE_VOICE_MAP and (
            azure_lang != primary_lang or azure_lang in _PRIORITY_LANGS
        ):
            detected = azure_lang
            print(f"[STT] lang={detected} (Azure LID, confidence={confidence})")
        else:
            script_lang = detect_lang_from_script(transcript, fallback="en-IN")
            if script_lang != "en-IN":
                detected = script_lang
                print(f"[STT] lang={detected} (Unicode script; azure_lang={azure_lang!r})")
            elif azure_lang and azure_lang in AZURE_VOICE_MAP:
                detected = azure_lang
                print(f"[STT] lang={detected} (Azure; no Indian script in transcript)")
            else:
                detected = "en-IN"
                print("[STT] lang=en-IN (fallback; no reliable detection)")

        return transcript, detected

    except Exception as e:
        print(f"[STT] Azure failed: {e}")
        return "", "en-IN"
