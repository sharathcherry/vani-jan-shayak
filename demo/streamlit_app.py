"""app.py — Vani Jan Sahayak (Streamlit browser UI)

Same pipeline as the WhatsApp Lambda:
  STT → translate → RAG → translate → TTS
rendered in the browser instead of Twilio.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

import boto3
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")
RAG_LAMBDA_NAME     = os.environ.get("RAG_LAMBDA_NAME", "gov-schemes-voice-rag")
BEDROCK_REGION      = os.environ.get("BEDROCK_REGION", "eu-central-1")
LAMBDA_REGION       = os.environ.get("LAMBDA_REGION", "eu-north-1")
WHATSAPP_DEMO_LINK  = "https://wa.me/14155238886?text=join%20angry-habit"

# Azure Neural TTS voices — mirrors config.py / tts.py exactly
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

# Single source of language display names — used in UI labels and translation prompts
_LANG_DISPLAY: dict[str, str] = {
    "hi-IN": "Hindi",    "te-IN": "Telugu",    "ta-IN": "Tamil",
    "kn-IN": "Kannada",  "ml-IN": "Malayalam", "mr-IN": "Marathi",
    "bn-IN": "Bengali",  "gu-IN": "Gujarati",  "pa-IN": "Punjabi",
    "or-IN": "Odia",     "as-IN": "Assamese",  "en-IN": "English",
}

# Unicode script ranges — same priority order as stt.py.
# NOTE: Marathi (mr-IN) shares Devanagari with Hindi, and Assamese (as-IN)
# shares Bengali script. Pure Unicode detection cannot distinguish these pairs;
# Azure LID handles them when voice is used.
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
]

# Intent keywords — ported from rag.py.
# If NONE appear in the English query, the RAG Lambda is not invoked,
# avoiding wasted quota on greetings and off-topic messages.
_INTENT_KEYWORDS = {
    "scheme", "schemes", "yojana", "scholarship", "loan", "subsidy",
    "benefit", "benefits", "apply", "application", "eligible", "eligibility",
    "government", "pension", "insurance", "grant", "allowance", "ration",
    "card", "certificate", "income", "caste", "student", "farmer", "woman",
    "disability", "welfare", "help", "support", "fee", "free", "money",
    "what", "how", "when", "where", "which", "is there", "are there",
    "can i", "do i", "tell me", "list", "?",
}

_URL_RE = re.compile(r"https?://[^\s,)>\"']+")


# ── AWS clients (cached across Streamlit reruns) ───────────────────────────────
@st.cache_resource
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


@st.cache_resource
def _lambda_client():
    return boto3.client("lambda", region_name=LAMBDA_REGION)


# ── Language helpers ───────────────────────────────────────────────────────────

def _detect_lang_from_script(text: str, fallback: str = "en-IN") -> str:
    """Identify language by Unicode script character density (>=15% threshold).

    Checks which script block accounts for the most non-whitespace characters,
    provided it exceeds 15% of the total. This is the Layer 2 fallback when
    Azure LID either returns empty or echoes the primary hint without confidence.
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


def _has_scheme_intent(english_text: str) -> bool:
    """Return True if the English query appears to be about a government scheme.

    Ported from rag.py — guards against invoking the RAG Lambda for greetings
    or completely off-topic messages.
    """
    lower = english_text.lower()
    return any(k in lower for k in _INTENT_KEYWORDS)


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _convert_to_wav(audio_bytes: bytes) -> bytes:
    """Transcode any audio format to 16 kHz mono PCM WAV using ffmpeg.

    Mirrors the logic in stt.py but uses shutil.which() to locate ffmpeg on
    PATH rather than assuming the Lambda layer path /opt/bin/ffmpeg.
    Azure Speech STT requires 16 kHz mono PCM WAV for non-OGG inputs.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it to process non-WAV/OGG audio.\n"
            "  Windows: winget install ffmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   apt install ffmpeg"
        )
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav",
             out_path],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exit {result.returncode}: {result.stderr.decode()[:300]}"
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def _extract_urls(text: str) -> list[str]:
    """Return all URLs found in text, deduplicated, in order of appearance."""
    seen: set[str] = set()
    result = []
    for url in _URL_RE.findall(text):
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _strip_urls(text: str) -> str:
    """Remove URLs from text so TTS does not read them aloud."""
    return _URL_RE.sub("", text).strip()


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def azure_stt(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    preferred_lang: str = "te-IN",
) -> tuple[str, str, str]:
    """Azure Speech STT with 3-layer language detection.

    Mirrors the logic in stt.py but returns a third value (error_detail)
    so the UI can display a clear message instead of a silent failure.

    Layer 1 -- Azure LID (up to 4 candidate languages)
    Layer 2 -- Unicode script scan on the returned transcript
    Layer 3 -- Fallback to en-IN

    Returns:
        (transcript, detected_lang_code, error_detail)
        error_detail is "" on success, human-readable message on failure.
    """
    if not AZURE_SPEECH_KEY:
        return "", "en-IN", "AZURE_SPEECH_KEY is not set. Check your .env file."

    try:
        # OGG/Opus (from WhatsApp or Firefox) goes directly to Azure.
        # st.audio_input returns WAV (Streamlit normalises browser recordings).
        # Everything else is converted to 16 kHz mono WAV via ffmpeg.
        if "ogg" in content_type or "opus" in content_type:
            audio_to_send    = audio_bytes
            stt_content_type = "audio/ogg; codecs=opus"
        else:
            audio_to_send    = _convert_to_wav(audio_bytes)
            stt_content_type = "audio/wav; codecs=audio/pcm; samplerate=16000"

        # Azure LID REST API hard limit: maximum 4 candidate languages.
        # Rotate the session-preferred language into slot 0 when it is not
        # already in the default list so Azure actively considers it.
        _LID_PRIORITY: list[str] = ["te-IN", "kn-IN", "hi-IN", "en-IN"]
        if preferred_lang not in _LID_PRIORITY and preferred_lang in AZURE_VOICE_MAP:
            _LID_PRIORITY = [preferred_lang] + _LID_PRIORITY[:3]
        lid_langs    = ",".join(_LID_PRIORITY)
        primary_lang = preferred_lang if preferred_lang in AZURE_VOICE_MAP else "te-IN"

        stt_url = (
            f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={primary_lang}&lid={lid_langs}&format=detailed"
        )
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                stt_url,
                headers={
                    "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                    "Content-Type": stt_content_type,
                },
                content=audio_to_send,
            )

        if resp.status_code != 200:
            st.session_state["_last_stt_raw"] = {
                "http_status": resp.status_code, "body": resp.text[:500]
            }
            return "", "en-IN", f"Azure HTTP {resp.status_code}: {resp.text[:300]}"

        result     = resp.json()
        # Persist for the debug expander in the voice tab (every call, success or fail)
        st.session_state["_last_stt_raw"] = result
        status     = result.get("RecognitionStatus", "Unknown")
        transcript = result.get("DisplayText", "")

        # With LID + detailed format the transcript lives in NBest[0]["Display"].
        # Fall back to top-level DisplayText for non-LID responses.
        nbest = result.get("NBest", [])
        if not transcript and nbest:
            transcript = nbest[0].get("Display") or nbest[0].get("Lexical", "")

        if not transcript:
            raw_hint   = json.dumps(result, ensure_ascii=False)[:300]
            status_msg = {
                "NoMatch": (
                    "Azure could not match speech -- try speaking more clearly "
                    "or closer to the mic."
                ),
                "InitialSilenceTimeout": (
                    "No speech detected -- the recording may be silent."
                ),
                "BabbleTimeout": (
                    "Too much background noise -- try in a quieter environment."
                ),
                "Error": (
                    "Azure Speech returned an error. Check your key and region."
                ),
                # RecognitionStatus "Success" with empty transcript + Confidence 0.0
                # means Azure received audio but found no recognisable words.
                "Success": (
                    "Audio was received but no words were recognised (confidence 0.0). "
                    "Try speaking louder and closer to the microphone, "
                    "or re-record in a quieter environment."
                ),
            }.get(status, f"Azure status: {status} -- no transcript. Raw: {raw_hint}")
            return "", "en-IN", status_msg

        # ── Language detection (mirrors stt.py priority logic) ────────────
        pl         = result.get("PrimaryLanguage", {})
        azure_lang = pl.get("Language", "") if isinstance(pl, dict) else ""

        # Trust Azure for the 4 high-confidence priority languages.
        # For others, prefer the Unicode script scan (Layer 2).
        _PRIORITY_LANGS = {"te-IN", "kn-IN", "hi-IN", "en-IN"}
        if azure_lang and azure_lang in AZURE_VOICE_MAP and (
            azure_lang != primary_lang or azure_lang in _PRIORITY_LANGS
        ):
            detected = azure_lang
        else:
            script_lang = _detect_lang_from_script(transcript, fallback="en-IN")
            if script_lang != "en-IN":
                detected = script_lang
            elif azure_lang and azure_lang in AZURE_VOICE_MAP:
                detected = azure_lang
            else:
                detected = "en-IN"

        return transcript, detected, ""

    except Exception as e:
        return "", "en-IN", f"STT exception: {e}"


def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> str:
    """Translate between languages using Amazon Bedrock Nova Lite.

    Identical model and prompt to translation.py so responses match the
    WhatsApp pipeline. Returns the original text unchanged on same-language
    pairs, empty input, or Bedrock failures (graceful degradation).
    """
    src = _LANG_DISPLAY.get(source_lang_code, "English")
    tgt = _LANG_DISPLAY.get(target_lang_code, "English")
    if src == tgt or not text.strip():
        return text
    prompt = (
        f"Translate the following {src} text to {tgt}. "
        f"Output ONLY the translation, nothing else.\n\n{text}"
    )
    try:
        resp = _bedrock().converse(
            modelId="eu.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 512},
        )
        return resp["output"]["message"]["content"][0]["text"].strip() or text
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text


def get_rag_answer(english_query: str) -> tuple[str, list[dict], dict]:
    """Invoke the gov-schemes-voice-rag Lambda via the /debug/query endpoint.

    Returns:
        answer   -- generated English answer text
        contexts -- list of source chunks: {scheme_name, state_or_ut,
                    scheme_category, text}  (empty list on failure)
        filters  -- metadata filters applied by the Lambda (state, beneficiary flags)
    """
    try:
        response = _lambda_client().invoke(
            FunctionName=RAG_LAMBDA_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps({
                "rawPath": "/debug/query",
                "body":    json.dumps({"query": english_query}),
                "headers": {"content-type": "application/json"},
            }),
        )
        result   = json.loads(response["Payload"].read())
        body     = json.loads(result.get("body", "{}"))
        answer      = body.get("answer",   "No information found.")
        raw_contexts = body.get("contexts", [])
        filters      = body.get("filters",  {})
        # Normalize contexts: older deployed Lambda returns plain text strings;
        # current lambda_function.py returns dicts with scheme_name etc.
        contexts = [
            c if isinstance(c, dict)
            else {"text": c, "scheme_name": "", "state_or_ut": "", "scheme_category": ""}
            for c in raw_contexts
        ]
        return answer, contexts, filters
    except Exception as e:
        st.error(f"RAG Lambda call failed: {e}")
        return "Sorry, could not retrieve scheme information at this time.", [], {}


def synthesize_speech(text: str, lang_code: str = "en-IN") -> bytes:
    """Azure Neural TTS -- returns MP3 bytes for the Streamlit audio player.

    Unlike tts.py (which uploads to S3 and returns a presigned URL for Twilio),
    this version returns raw bytes that st.audio() can play directly in the browser.
    """
    if not AZURE_SPEECH_KEY:
        raise RuntimeError("AZURE_SPEECH_KEY is not set -- cannot synthesize speech.")
    xml_lang, voice_name = AZURE_VOICE_MAP.get(lang_code, ("en-IN", "en-IN-NeerjaNeural"))
    safe_text = text[:2000].replace("&", "and").replace("<", "").replace(">", "")
    ssml = (
        f"<speak version='1.0' xml:lang='{xml_lang}'>"
        f"<voice name='{voice_name}'>{safe_text}</voice>"
        f"</speak>"
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type":              "application/ssml+xml",
                "X-Microsoft-OutputFormat":  "audio-24khz-160kbitrate-mono-mp3",
                "User-Agent":                "GovSchemes-VoiceBot/1.0",
            },
            content=ssml.encode("utf-8"),
        )
    resp.raise_for_status()
    return resp.content


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(query_text: str, lang_code: str) -> dict:
    """Run the full voice/text pipeline for a single query.

    Steps:
      1. Translate native -> English  (Bedrock Nova Lite)
      2. Intent guard                 (keyword check, no Lambda cost)
      3. RAG                          (gov-schemes-voice-rag Lambda)
      4. Translate English -> native  (Bedrock Nova Lite)
      5. Extract links                (from English answer -- survive translation)
      6. TTS                          (Azure Neural TTS -> MP3 bytes)

    Returns a dict with all intermediate values for display and debug.
    """
    results: dict = {"lang_code": lang_code}

    # ── Step 1: translate to English ──────────────────────────────────────
    if lang_code != "en-IN":
        with st.spinner(f"Translating {_LANG_DISPLAY[lang_code]} -> English..."):
            english_query = translate_text(query_text, lang_code, "en-IN")
    else:
        english_query = query_text
    results["english_query"] = english_query

    # ── Step 2: intent guard -- skip RAG for greetings / off-topic ────────
    if not _has_scheme_intent(english_query):
        clarify = (
            "I didn't catch a specific government scheme question. "
            "Try asking something like: 'What scholarships are available for SC "
            "students?' or 'How do I apply for PM Kisan Yojana?'"
        )
        if lang_code != "en-IN":
            with st.spinner(f"Translating response -> {_LANG_DISPLAY[lang_code]}..."):
                clarify = translate_text(clarify, "en-IN", lang_code)
        results.update({
            "no_intent":     True,
            "native_answer": clarify,
            "links":         [],
            "tts_text":      clarify,
            "contexts":      [],
            "filters":       {},
        })
        return results

    # ── Step 3: RAG ───────────────────────────────────────────────────────
    with st.spinner("Searching 22,000+ government scheme documents..."):
        english_answer, contexts, filters = get_rag_answer(english_query)
    results["english_answer"] = english_answer
    results["contexts"]       = contexts
    results["filters"]        = filters

    # ── Step 4: translate answer back to native language ──────────────────
    if lang_code != "en-IN":
        with st.spinner(f"Translating English -> {_LANG_DISPLAY[lang_code]}..."):
            native_answer = translate_text(english_answer, "en-IN", lang_code)
    else:
        native_answer = english_answer
    results["native_answer"] = native_answer

    # ── Step 5: extract links and strip them from TTS text ────────────────
    # Links are extracted from the English answer because URLs survive
    # translation poorly; the native answer is used for TTS after stripping.
    results["links"]    = _extract_urls(english_answer)
    results["tts_text"] = _strip_urls(native_answer)

    # ── Step 6: TTS ───────────────────────────────────────────────────────
    if AZURE_SPEECH_KEY:
        with st.spinner("Generating voice response..."):
            try:
                results["audio_bytes"] = synthesize_speech(results["tts_text"], lang_code)
            except Exception as e:
                results["tts_error"] = str(e)

    return results


# ── Results display ────────────────────────────────────────────────────────────

def _display_results(results: dict) -> None:
    lang_code  = results["lang_code"]
    lang_label = _LANG_DISPLAY.get(lang_code, lang_code)

    st.divider()

    # ── Answer text ───────────────────────────────────────────────────────
    st.subheader(f"Answer ({lang_label})")
    if results.get("no_intent"):
        st.info(results["native_answer"])
    else:
        st.write(results["native_answer"])

    # ── Audio player ──────────────────────────────────────────────────────
    if "audio_bytes" in results:
        st.subheader("Voice Response")
        st.audio(results["audio_bytes"], format="audio/mp3")
    elif "tts_error" in results:
        st.warning(f"TTS unavailable: {results['tts_error']}")

    # ── Official links ────────────────────────────────────────────────────
    if results.get("links"):
        st.subheader("Official Links")
        for url in results["links"]:
            st.markdown(f"- {url}")

    # ── Source scheme cards (from Lambda /debug/query response) ───────────
    contexts = results.get("contexts", [])
    if contexts:
        with st.expander(f"Source schemes retrieved ({len(contexts)})"):
            for i, ctx in enumerate(contexts, 1):
                name  = ctx.get("scheme_name") or "Unknown Scheme"
                state = ctx.get("state_or_ut") or "Central / National"
                cat   = ctx.get("scheme_category", "")
                label = f"**{i}. {name}** -- {state}"
                if cat:
                    label += f" | _{cat}_"
                st.markdown(label)
                snippet = ctx.get("text", "")[:280].replace("\n", " ").strip()
                if snippet:
                    st.caption(snippet + "...")
                if i < len(contexts):
                    st.divider()

    # ── Pipeline debug details ────────────────────────────────────────────
    with st.expander("Pipeline details"):
        st.markdown(f"**Detected language:** `{lang_code}` ({lang_label})")

        if "english_query" in results:
            st.markdown(f"**English query sent to RAG:** {results['english_query']}")

        filters = results.get("filters", {})
        if filters:
            parts = []
            if state := filters.get("state"):
                parts.append(f"state={state}")
            for flag in ("is_for_sc_st", "is_for_students", "is_for_women",
                         "is_for_farmers", "is_for_disabled"):
                if filters.get(flag):
                    parts.append(flag.replace("is_for_", ""))
            if parts:
                st.markdown(f"**Metadata filters applied:** `{', '.join(parts)}`")

        if "english_answer" in results:
            st.markdown(f"**English answer from RAG:** {results['english_answer']}")

        if results.get("no_intent"):
            st.info("RAG was skipped -- query did not match any scheme intent keywords.")


# ── Service diagnostics ────────────────────────────────────────────────────────

def _diag_azure_stt() -> tuple[bool, str]:
    """Validate the Azure Speech key with a minimal STT request (1-byte WAV header)."""
    if not AZURE_SPEECH_KEY:
        return False, "Key not set"
    # Send a nearly-empty WAV (44-byte header, zero audio data).
    # Azure should return a 200 with RecognitionStatus=InitialSilenceTimeout,
    # which proves the key and endpoint are valid.
    wav_header = bytes([
        0x52,0x49,0x46,0x46, 0x24,0x00,0x00,0x00,  # RIFF....
        0x57,0x41,0x56,0x45, 0x66,0x6d,0x74,0x20,  # WAVEfmt_
        0x10,0x00,0x00,0x00, 0x01,0x00, 0x01,0x00,  # chunk / PCM / 1ch
        0x80,0x3E,0x00,0x00, 0x00,0x7D,0x00,0x00,  # 16000 Hz / byte rate
        0x02,0x00, 0x10,0x00,                        # block align / 16-bit
        0x64,0x61,0x74,0x61, 0x00,0x00,0x00,0x00,  # data chunk, 0 bytes
    ])
    try:
        url = (
            f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
            f"?language=en-IN&format=simple"
        )
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                    "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                },
                content=wav_header,
            )
        if resp.status_code == 401:
            return False, "401 Unauthorized — key is invalid or region is wrong"
        if resp.status_code == 403:
            return False, "403 Forbidden — key quota exhausted or subscription inactive"
        if resp.status_code == 200:
            return True, "OK"
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as e:
        return False, f"Connection error: {e}"


def _diag_bedrock() -> tuple[bool, str]:
    """Call Bedrock Nova Lite with a one-word prompt to confirm auth + model access."""
    try:
        resp = _bedrock().converse(
            modelId="eu.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 5},
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _diag_lambda() -> tuple[bool, str]:
    """Invoke the RAG Lambda with a trivial query and check the response shape."""
    try:
        response = _lambda_client().invoke(
            FunctionName=RAG_LAMBDA_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps({
                "rawPath": "/debug/query",
                "body":    json.dumps({"query": "PM Kisan"}),
                "headers": {"content-type": "application/json"},
            }),
        )
        result      = json.loads(response["Payload"].read())
        status_code = result.get("statusCode", "?")
        if status_code == 200:
            return True, "OK"
        return False, f"Lambda returned {status_code}: {result.get('body','')[:200]}"
    except Exception as e:
        return False, str(e)


def _render_sidebar() -> None:
    """Render the diagnostics panel in the Streamlit sidebar."""
    with st.sidebar:
        st.markdown("## Service Diagnostics")
        st.caption(
            "Tests each backend service with a live call. "
            "Run this to confirm the pipeline is actually connected."
        )

        run_diagnostics = st.button("Run Diagnostics", type="primary", use_container_width=True)
        if not st.session_state.get("_diagnostics_ran", False) or run_diagnostics:
            st.session_state["_diagnostics_ran"] = True
            with st.spinner("Testing Azure Speech..."):
                ok, msg = _diag_azure_stt()
                st.session_state["_diag_azure"] = (ok, msg)

            with st.spinner("Testing Bedrock Nova Lite..."):
                ok, msg = _diag_bedrock()
                st.session_state["_diag_bedrock"] = (ok, msg)

            with st.spinner("Testing Lambda..."):
                ok, msg = _diag_lambda()
                st.session_state["_diag_lambda"] = (ok, msg)

        # Show results if they exist
        for key, label in [
            ("_diag_azure",   "Azure Speech"),
            ("_diag_bedrock", "Bedrock Nova Lite"),
            ("_diag_lambda",  "Lambda"),
        ]:
            if key in st.session_state:
                ok, msg = st.session_state[key]
                icon = "✅" if ok else "❌"
                colour = "#2dba4e" if ok else "#e05252"
                st.markdown(
                    f"<div style='margin:4px 0;padding:6px 10px;"
                    f"border-left:3px solid {colour};border-radius:4px;"
                    f"background:#0d1b2e;font-size:0.85rem;'>"
                    f"<b>{icon} {label}</b><br/>"
                    f"<span style='color:#9ab8d8;word-break:break-word;'>{msg}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )



# ── Streamlit page setup ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vani Jan Sahayak",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

_render_sidebar()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    /* ── Dark navy base -- government / trust tone ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp,
    [data-testid="stHeader"], section.main, .main,
    [data-testid="stSidebar"] {
        background-color: #060d18 !important;
        color: #dce8f5 !important;
        font-family: 'Manrope', sans-serif;
    }

    /* Saffron + green flag-glow at corners */
    .stApp {
        background:
            radial-gradient(ellipse 1200px 450px at -5% -10%,
                rgba(232,135,10,0.07), rgba(232,135,10,0) 60%),
            radial-gradient(ellipse 900px 380px at 105% 5%,
                rgba(19,136,8,0.06), rgba(19,136,8,0) 55%),
            linear-gradient(180deg, #060d18 0%, #060d18 100%) !important;
    }

    [data-testid="stBlock"], [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"], div.block-container {
        background-color: transparent !important;
    }

    textarea, input[type="text"], input[type="number"],
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    .stTextArea textarea, .stTextInput > div > input {
        background-color: #0d1b2e !important;
        color: #dce8f5 !important;
        border: 1px solid #1e3554 !important;
        border-radius: 8px !important;
    }

    [data-testid="stSelectbox"] > div > div,
    .stSelectbox > div > div {
        background-color: #0d1b2e !important;
        color: #dce8f5 !important;
        border: 1px solid #1e3554 !important;
        border-radius: 8px !important;
    }

    hr { border-color: #132038 !important; }

    [data-testid="stTabs"] [role="tab"] { color: #6a8aaa !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #e8870a !important;
        border-bottom-color: #e8870a !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        border-bottom: 1px solid #132038 !important;
        background: transparent !important;
    }

    [data-testid="stAlert"] {
        background-color: #0d1b2e !important;
        border: 1px solid #1e3554 !important;
        color: #dce8f5 !important;
        border-radius: 8px !important;
    }

    [data-testid="stExpander"] {
        background-color: #0d1b2e !important;
        border: 1px solid #132038 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #dce8f5 !important; }

    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #c4700a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }
    button[kind="primary"]:hover { background-color: #e8870a !important; }
    .stButton > button {
        background-color: #0d1b2e !important;
        color: #dce8f5 !important;
        border: 1px solid #1e3554 !important;
        border-radius: 8px !important;
    }

    .stCaption, [data-testid="stCaption"] { color: #4a6a8a !important; }

    [data-testid="stFileUploader"] {
        background-color: #0d1b2e !important;
        border: 1px dashed #1e3554 !important;
        border-radius: 8px !important;
        color: #dce8f5 !important;
    }

    /* ── Prototype notice banner ── */
    .proto-banner {
        background: #0d1b2e;
        border-left: 3px solid #e8870a;
        color: #dce8f5;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.8rem;
        font-size: 0.93rem;
        line-height: 1.65;
        text-align: center;
    }
    .proto-banner strong { font-weight: 700; color: #f0c070; font-size: 1rem; }
    .proto-banner a { color: #e8870a; text-decoration: underline; font-weight: 600; }
    .proto-banner a:hover { color: #ffaa40; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #060d18 0%, #0d2040 55%, #112a50 100%);
        color: #dce8f5;
        border-radius: 14px;
        border: 1px solid #1e3554;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.55);
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: #ffffff;
        text-align: center;
    }
    .hero p { margin: 0; opacity: 0.82; font-size: 1rem; text-align: center; }
    .chip {
        display: inline-block;
        color: #dce8f5;
        margin: 0.2rem 0.35rem 0 0;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 999px;
        padding: 0.18rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ── WhatsApp / judge box ── */
    .judge-box {
        background: #0d1b2e;
        border: 1px solid #1e3554;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-top: 0.4rem;
        color: #9ab8d8;
        font-size: 0.92rem;
    }
    .judge-box a { color: #e8870a; font-weight: 600; }
    .judge-box a:hover { color: #ffaa40; }
    .judge-box code {
        background: rgba(232,135,10,0.12);
        color: #f0c070;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page header ────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="proto-banner">
      <strong>Prototype Notice</strong><br/>
      This browser interface is a <strong>prototype built solely to demonstrate
      the functionality</strong> of our deployed application to judges.
      The actual product runs as a live WhatsApp AI assistant and cannot be
      accessed via a standard web URL (it is a Twilio webhook).<br/><br/>
      <strong>Try the live deployed app on WhatsApp:</strong>&nbsp;
      <a href="{WHATSAPP_DEMO_LINK}" target="_blank">Open Vani Jan Sahayak on WhatsApp</a><br/>
      <small>If WhatsApp asks for a code, send: <strong>join angry-habit</strong></small>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Vani Jan Sahayak</h1>
      <p>AI helpline for Indian Government Schemes -- multilingual voice and text.</p>
      <div style="text-align:center; margin-top:0.5rem;">
        <span class="chip">STT</span>
        <span class="chip">Translation</span>
        <span class="chip">Hybrid RAG</span>
        <span class="chip">Cohere Rerank</span>
        <span class="chip">TTS</span>
        <span class="chip">12 Languages</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="judge-box">
      <b>Live WhatsApp Bot:</b>&nbsp;
      <a href="{WHATSAPP_DEMO_LINK}" target="_blank">Open Vani Jan Sahayak on WhatsApp</a>
      &nbsp;&nbsp;<span style="color:#9a6060;">
      Send <code>join angry-habit</code> if prompted</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not AZURE_SPEECH_KEY:
    st.warning(
        "AZURE_SPEECH_KEY not found in environment -- STT and TTS will not work. "
        "Add it to your .env file and restart."
    )

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_voice, tab_text = st.tabs(["Voice Demo", "Text Demo"])


# ── Voice tab ──────────────────────────────────────────────────────────────────
with tab_voice:
    st.write("Record a voice note or upload an audio file asking about a government scheme.")

    preferred_lang = st.selectbox(
        "Language hint for speech recognition",
        options=list(AZURE_VOICE_MAP.keys()),
        format_func=lambda x: _LANG_DISPLAY.get(x, x),
        index=list(AZURE_VOICE_MAP.keys()).index("te-IN"),
        help=(
            "Sets the primary language hint for Azure STT. "
            "The system still auto-detects via LID and Unicode script scan."
        ),
    )

    # st.audio_input was added in Streamlit 1.31 -- fallback gracefully
    audio_recorded = None
    try:
        audio_recorded = st.audio_input("Record your question")
    except AttributeError:
        st.info("Audio recording requires Streamlit >= 1.31. Use the file uploader below.")

    audio_uploaded = st.file_uploader(
        "Or upload an audio file",
        type=["wav", "mp3", "ogg", "aac", "m4a", "opus", "webm"],
        help="Supported: WAV, MP3, OGG, AAC, M4A, OPUS, WebM (non-WAV/OGG requires ffmpeg).",
    )

    audio_source = audio_recorded or audio_uploaded

    if audio_source and st.button("Process Voice", type="primary"):
        audio_bytes = audio_source.read()

        # Determine content type for azure_stt routing (OGG/Opus -> direct; else -> ffmpeg)
        if audio_recorded and not audio_uploaded:
            # Streamlit normalises browser recordings to WAV internally
            content_type = "audio/wav"
        else:
            fname = getattr(audio_uploaded, "name", "")
            ext   = fname.rsplit(".", 1)[-1].lower() if "." in fname else "wav"
            content_type = {
                "wav":  "audio/wav",
                "ogg":  "audio/ogg; codecs=opus",
                "opus": "audio/ogg; codecs=opus",
                "mp3":  "audio/mpeg",
                "aac":  "audio/aac",
                "m4a":  "audio/mp4",
                "webm": "audio/webm",
            }.get(ext, "audio/wav")

        with st.spinner("Transcribing audio..."):
            transcript, lang_code, stt_error = azure_stt(
                audio_bytes, content_type, preferred_lang
            )

        if not transcript:
            st.error(stt_error or "No transcript returned.")
        else:
            st.success(f"**Transcript:** {transcript}")
            st.info(f"**Detected language:** {_LANG_DISPLAY.get(lang_code, lang_code)}")
            results = run_pipeline(transcript, lang_code)
            _display_results(results)

        # Always show the raw Azure response so it's easy to verify the call happened
        if "_last_stt_raw" in st.session_state:
            with st.expander("Raw Azure STT response (proof of API call)"):
                st.json(st.session_state["_last_stt_raw"])


# ── Text tab ───────────────────────────────────────────────────────────────────
with tab_text:
    st.write("Type your question in any Indian language or English.")

    sample_prompts = [
        "SC student Karnataka B.Tech scholarship",
        "PM Kisan Yojana ke liye kaise apply karein?",
        "తెలంగాణలో రైతులకు ఏ పథకాలు ఉన్నాయి?",
        "Disabled women self-employment schemes in Tamil Nadu",
        "ਪੰਜਾਬ ਵਿੱਚ ਕਿਸਾਨਾਂ ਲਈ ਸਬਸਿਡੀ ਯੋਜਨਾਵਾਂ",
    ]
    sample_choice = st.selectbox(
        "Optional: try a sample query",
        options=["(Use my own question)"] + sample_prompts,
        index=0,
    )

    user_text = st.text_area(
        "Your question",
        placeholder=(
            "e.g. What scholarships are available for SC students doing BTech?\n"
            "or: SC విద్యార్థులకు ఏ స్కాలర్‌షిప్‌లు అందుబాటులో ఉన్నాయి?\n"
            "or: पीएम किसान योजना के लिए कैसे आवेदन करें?"
        ),
        height=130,
        value="" if sample_choice == "(Use my own question)" else sample_choice,
    )

    if st.button("Search Schemes", type="primary"):
        query = user_text.strip()
        if not query:
            st.warning("Please enter a question first.")
        else:
            lang_code = _detect_lang_from_script(query, fallback="en-IN")
            st.info(f"**Detected language:** {_LANG_DISPLAY.get(lang_code, lang_code)}")
            results = run_pipeline(query, lang_code)
            _display_results(results)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Supported languages: Hindi · Telugu · Tamil · Kannada · Malayalam · "
    "Marathi · Bengali · Gujarati · Punjabi · Odia · Assamese · English"
)
