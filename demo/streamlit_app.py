"""
streamlit_app.py -- Vani Jan Sahayak (browser UI)
Same pipeline as the WhatsApp Lambda: STT -> translate -> RAG -> translate -> TTS
but rendered in the browser instead of Twilio.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import tempfile

import boto3
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -- Config -------------------------------------------------------------------
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")
RAG_LAMBDA_NAME     = "gov-schemes-voice-rag"
BEDROCK_REGION      = "eu-central-1"
LAMBDA_REGION       = "eu-north-1"

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

_LANG_NAMES: dict[str, str] = {
    "hi-IN": "Hindi",  "te-IN": "Telugu", "ta-IN": "Tamil",
    "kn-IN": "Kannada","ml-IN": "Malayalam","mr-IN": "Marathi",
    "bn-IN": "Bengali","gu-IN": "Gujarati","pa-IN": "Punjabi",
    "or-IN": "Odia",   "as-IN": "Assamese","en-IN": "English",
}

_LANG_DISPLAY: dict[str, str] = {
    "hi-IN": "Hindi",  "te-IN": "Telugu", "ta-IN": "Tamil",
    "kn-IN": "Kannada","ml-IN": "Malayalam","mr-IN": "Marathi",
    "bn-IN": "Bengali","gu-IN": "Gujarati","pa-IN": "Punjabi",
    "or-IN": "Odia",   "as-IN": "Assamese","en-IN": "English",
}

_SCRIPT_DETECT = [
    ("te-IN", "\u0c00", "\u0c7f"),
    ("kn-IN", "\u0c80", "\u0cff"),
    ("hi-IN", "\u0900", "\u097f"),
    ("ta-IN", "\u0b80", "\u0bff"),
    ("ml-IN", "\u0d00", "\u0d7f"),
    ("mr-IN", "\u0900", "\u097f"),
    ("gu-IN", "\u0a80", "\u0aff"),
    ("pa-IN", "\u0a00", "\u0a7f"),
    ("or-IN", "\u0b00", "\u0b7f"),
    ("bn-IN", "\u0980", "\u09ff"),
    ("as-IN", "\u0980", "\u09ff"),
]

_URL_RE = re.compile(r"https?://[^\s,)>\"']+")

# Words indicating the user explicitly wants overseas / abroad programs
_OVERSEAS_WORDS = {
    "overseas", "abroad", "foreign country", "outside india",
    "international", "out of india", "study abroad", "foreign university",
}


# -- AWS clients (cached across reruns) ---------------------------------------
@st.cache_resource
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


@st.cache_resource
def _lambda_client():
    return boto3.client("lambda", region_name=LAMBDA_REGION)


# -- Pipeline helpers ---------------------------------------------------------

def _detect_lang_from_script(text: str, fallback: str = "en-IN") -> str:
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


def _convert_to_wav(audio_bytes: bytes) -> bytes:
    """Convert any audio format to 16 kHz mono WAV via system ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", "-f", "wav", out_path],
            capture_output=True, timeout=30,
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


def azure_stt(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    preferred_lang: str = "te-IN",
) -> tuple[str, str]:
    """Azure Speech STT. Returns (transcript, detected_lang_code)."""
    if not AZURE_SPEECH_KEY:
        st.error("AZURE_SPEECH_KEY not set. Check your .env file.")
        return "", "en-IN"
    try:
        if "ogg" in content_type or "opus" in content_type:
            audio_to_send = audio_bytes
            stt_content_type = "audio/ogg; codecs=opus"
        elif "wav" in content_type or "x-wav" in content_type:
            audio_to_send = audio_bytes
            stt_content_type = "audio/wav; codecs=audio/pcm; samplerate=16000"
        else:
            audio_to_send = _convert_to_wav(audio_bytes)
            stt_content_type = "audio/wav; codecs=audio/pcm; samplerate=16000"

        _LID_PRIORITY = ["te-IN", "kn-IN", "hi-IN", "en-IN"]
        if preferred_lang not in _LID_PRIORITY and preferred_lang in AZURE_VOICE_MAP:
            _LID_PRIORITY = [preferred_lang] + _LID_PRIORITY[:3]
        lid_langs = ",".join(_LID_PRIORITY)
        primary_lang = preferred_lang if preferred_lang in AZURE_VOICE_MAP else "te-IN"

        stt_url = (
            f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={primary_lang}&lid={lid_langs}&format=simple"
        )
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                stt_url,
                headers={
                    "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                    "Content-Type": stt_content_type,
                },
                content=audio_to_send,
            )
        result = resp.json()
        transcript = result.get("DisplayText", "")
        pl = result.get("PrimaryLanguage", {})
        azure_lang = pl.get("Language", "") if isinstance(pl, dict) else ""

        if not transcript:
            return "", "en-IN"

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

        return transcript, detected
    except Exception as e:
        st.error(f"STT failed: {e}")
        return "", "en-IN"


def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> str:
    """Translate via Bedrock Nova Lite."""
    src = _LANG_NAMES.get(source_lang_code, "English")
    tgt = _LANG_NAMES.get(target_lang_code, "English")
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


def _enrich_query(english_query: str) -> str:
    """Append a domestic-focus constraint unless the user explicitly asked for overseas.

    The Cohere reranker can surface overseas scholarships (e.g. Prabhuddha) for
    queries like 'SC student Karnataka B.Tech scholarship' because they lexically
    match. This suffix steers both retrieval and generation toward domestic schemes.
    """
    lower = english_query.lower()
    if any(word in lower for word in _OVERSEAS_WORDS):
        return english_query  # user wants overseas -- preserve as-is
    return (
        english_query
        + " Focus only on schemes and scholarships available within India"
        " for students currently studying in India. Exclude overseas or international programs."
    )


def get_rag_answer(english_query: str) -> tuple[str, dict, list]:
    """Call the RAG Lambda with a domestically-enriched query.
    Returns (answer, filters_applied, contexts).
    """
    enriched = _enrich_query(english_query)
    try:
        response = _lambda_client().invoke(
            FunctionName=RAG_LAMBDA_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps({
                "rawPath": "/debug/query",
                "body": json.dumps({"query": enriched}),
                "headers": {"content-type": "application/json"},
            }),
        )
        result = json.loads(response["Payload"].read())
        body = json.loads(result.get("body", "{}"))
        answer   = body.get("answer", "No information found.")
        filters  = body.get("filters", {})
        contexts = body.get("contexts", [])
        return answer, filters, contexts
    except Exception as e:
        st.error(f"RAG Lambda call failed: {e}")
        return "Sorry, could not retrieve scheme information at this time.", {}, []


def synthesize_speech(text: str, lang_code: str = "en-IN") -> bytes:
    """Azure Neural TTS -- returns MP3 bytes directly (no S3 needed)."""
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
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
                "User-Agent": "GovSchemes-VoiceBot/1.0",
            },
            content=ssml.encode("utf-8"),
        )
    resp.raise_for_status()
    return resp.content


def _extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls = []
    for url in _URL_RE.findall(text):
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("", text).strip()


def _parse_schemes(answer: str) -> list[str]:
    """Split 'Scheme 1: ... Scheme 2: ...' answer into individual scheme texts.
    Returns a list of scheme description strings, or empty list if not that format.
    """
    parts = re.split(r"\bScheme\s+\d+\s*:", answer, flags=re.IGNORECASE)
    schemes = [p.strip() for p in parts[1:] if p.strip()]
    return schemes


# -- Full pipeline ------------------------------------------------------------

def run_pipeline(query_text: str, lang_code: str) -> dict:
    results: dict = {"lang_code": lang_code}

    if lang_code != "en-IN":
        with st.spinner(f"Translating {_LANG_DISPLAY[lang_code]} to English..."):
            english_query = translate_text(query_text, lang_code, "en-IN")
    else:
        english_query = query_text
    results["english_query"] = english_query

    with st.spinner("Searching 22,000+ scheme documents for top 2 matches..."):
        english_answer, rag_filters, rag_contexts = get_rag_answer(english_query)
    results["english_answer"] = english_answer
    results["rag_filters"]    = rag_filters
    results["rag_contexts"]   = rag_contexts

    if lang_code != "en-IN":
        with st.spinner(f"Translating answer to {_LANG_DISPLAY[lang_code]}..."):
            native_answer = translate_text(english_answer, "en-IN", lang_code)
    else:
        native_answer = english_answer
    results["native_answer"] = native_answer

    results["links"] = _extract_urls(english_answer)
    results["tts_text"] = _strip_urls(native_answer)

    if AZURE_SPEECH_KEY:
        with st.spinner("Generating voice response..."):
            try:
                results["audio_bytes"] = synthesize_speech(results["tts_text"], lang_code)
            except Exception as e:
                results["tts_error"] = str(e)

    return results


def _display_results(results: dict) -> None:
    lang_code      = results["lang_code"]
    lang_label     = _LANG_DISPLAY.get(lang_code, lang_code)
    native_answer  = results["native_answer"]
    english_answer = results["english_answer"]

    native_schemes  = _parse_schemes(native_answer)
    english_schemes = _parse_schemes(english_answer)

    st.divider()

    # ── Scheme cards ──────────────────────────────────────────────────────────
    if len(native_schemes) >= 2:
        st.subheader(f"Top 2 Recommended Schemes  ({lang_label})")
        col_a, col_b = st.columns(2)
        for col, scheme_text, idx in zip(
            [col_a, col_b], native_schemes[:2], [1, 2]
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"##### Scheme {idx}")
                    st.write(scheme_text)
    elif native_schemes:
        st.subheader(f"Recommended Scheme  ({lang_label})")
        with st.container(border=True):
            st.write(native_schemes[0])
    else:
        st.subheader(f"Answer  ({lang_label})")
        with st.container(border=True):
            st.write(native_answer)

    # ── Voice playback ────────────────────────────────────────────────────────
    if "audio_bytes" in results:
        st.subheader("Voice Response")
        st.audio(results["audio_bytes"], format="audio/mp3")
    elif "tts_error" in results:
        st.warning(f"TTS unavailable: {results['tts_error']}")

    # ── Official links ────────────────────────────────────────────────────────
    if results.get("links"):
        st.subheader("Apply Online")
        for url in results["links"]:
            st.markdown(f"- {url}")

    # ── Technical details (for judges) ───────────────────────────────────────
    with st.expander("Pipeline details  (for judges)"):
        filters = results.get("rag_filters", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Active Qdrant filters**")
            if filters:
                if s := filters.get("state"):
                    st.markdown(f"- State: `{s}` + Central/National")
                for k, label in [
                    ("is_for_sc_st",    "SC/ST only"),
                    ("is_for_students", "Students only"),
                    ("is_for_women",    "Women only"),
                    ("is_for_farmers",  "Farmers only"),
                    ("is_for_disabled", "Persons with disability only"),
                ]:
                    if filters.get(k):
                        st.markdown(f"- {label}")
            else:
                st.markdown("None (unfiltered search)")

        with col2:
            st.markdown("**Query info**")
            st.markdown(f"- Detected language: `{lang_code}` ({lang_label})")
            st.markdown(f"- English query: _{results['english_query'][:120]}_")

        if english_schemes:
            st.markdown("**English answer (pre-translation)**")
            for i, s in enumerate(english_schemes, 1):
                st.markdown(f"**Scheme {i}:** {s}")
        else:
            st.markdown(f"**English answer:** {english_answer}")

        contexts = results.get("rag_contexts", [])
        if contexts:
            st.markdown("**Retrieved chunks after Cohere rerank**")
            for i, ctx in enumerate(contexts):
                st.markdown(
                    f"{i+1}. **{ctx.get('scheme_name', '?')}** "
                    f"| {ctx.get('state_or_ut', '?')} "
                    f"| _{ctx.get('scheme_category', '?')}_"
                )


# ── Page config (MUST be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Vani Jan Sahayak — AI Voice Helpline",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🇮🇳 Vani Jan Sahayak")
    st.caption("*Voice-first AI helpline for Indian government schemes*")
    st.divider()

    st.markdown("### How it works")
    st.markdown("""
1. **Voice / Text** — User asks in any Indian language
2. **Azure STT** — Multilingual speech-to-text with auto language detection
3. **Bedrock Nova Lite** — Translates native → English
4. **Qdrant Hybrid Search** — Dense + Sparse (BM25) search over 22,287 docs with metadata pre-filtering
5. **Cohere Rerank** — Reranks top-20 candidates → top-5
6. **Bedrock Nova Lite** — Generates answer with top-2 schemes
7. **Bedrock Nova Lite** — Translates English → user's language
8. **Azure Neural TTS** — Converts answer to voice in user's language
9. **WhatsApp / Browser** — Delivers to user
""")

    st.divider()
    st.markdown("### Tech Stack")
    for service, detail in [
        ("STT",         "Azure Cognitive Services (10+ Indian languages)"),
        ("TTS",         "Azure Neural TTS (Neural voices per language)"),
        ("Translation", "Amazon Bedrock Nova Lite"),
        ("Vector DB",   "Qdrant Cloud — Hybrid (Dense + BM25)"),
        ("Embeddings",  "BAAI/bge-large-en-v1.5 + Qdrant/BM25"),
        ("Reranker",    "Cohere rerank-v4-fast (Azure AI Foundry)"),
        ("LLM",         "Amazon Bedrock Nova Lite"),
        ("Compute",     "AWS Lambda (container) + API Gateway"),
        ("Storage",     "Amazon S3 (cache + sessions + audio)"),
        ("Delivery",    "Twilio WhatsApp Business API"),
    ]:
        st.markdown(f"**{service}:** {detail}")

    st.divider()
    st.markdown("### Dataset")
    m1, m2, m3 = st.columns(3)
    m1.metric("Docs", "22,287")
    m2.metric("Schemes", "2,500+")
    m3.metric("Languages", "10+")

    st.divider()
    if not AZURE_SPEECH_KEY:
        st.error("AZURE_SPEECH_KEY not set — STT and TTS will not work.")
    else:
        st.success("Azure Speech: connected")

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>🇮🇳 Vani Jan Sahayak</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:1.15rem;color:#555;margin-top:4px'>"
    "Voice-first AI helpline for Indian government schemes &nbsp;|&nbsp; "
    "Ask in <strong>any Indian language</strong>, get answers in your language"
    "</p>",
    unsafe_allow_html=True,
)

# Key metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Scheme Documents",  "22,287+",    help="Chunked, embedded, and indexed in Qdrant Cloud")
m2.metric("Languages Supported", "10+",      help="Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Assamese, English")
m3.metric("Schemes per Query",  "Top 2",     help="Most relevant schemes matched to user's state and profile")
m4.metric("Avg Response Time",  "~7 s",      help="End-to-end: voice in → audio answer out on WhatsApp")

st.divider()

# Pipeline flow visualization
st.markdown("#### End-to-end Pipeline")
p = st.columns([2, 0.3, 2, 0.3, 2.8, 0.3, 2, 0.3, 2])
pipeline_steps = [
    ("🎙️", "Voice / Text", "User query\nin any language"),
    None,
    ("📝", "Azure STT", "Multilingual\nauto lang-detect"),
    None,
    ("🔍", "Qdrant + Cohere + Bedrock", "Hybrid search → Rerank → Answer (top 2 schemes)"),
    None,
    ("🔄", "Bedrock Translate", "English →\nnative language"),
    None,
    ("🔊", "Azure Neural TTS", "Voice answer\ndelivered"),
]
for col, step in zip(p, pipeline_steps):
    if step is None:
        col.markdown(
            "<div style='text-align:center;font-size:1.6rem;padding-top:22px'>→</div>",
            unsafe_allow_html=True,
        )
    else:
        icon, title, subtitle = step
        col.markdown(
            f"<div style='background:#f0f2f6;border-radius:8px;padding:10px;text-align:center'>"
            f"<div style='font-size:1.6rem'>{icon}</div>"
            f"<div style='font-weight:600;font-size:0.85rem'>{title}</div>"
            f"<div style='color:#888;font-size:0.75rem'>{subtitle}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_voice, tab_text = st.tabs(["🎙️  Voice", "⌨️  Text"])

# -- Voice tab ----------------------------------------------------------------
with tab_voice:
    st.markdown(
        "Record or upload a voice note in **any Indian language** — "
        "the system auto-detects your language."
    )

    left, right = st.columns([1, 2])

    with left:
        preferred_lang = st.selectbox(
            "Language hint",
            options=list(AZURE_VOICE_MAP.keys()),
            format_func=lambda x: _LANG_DISPLAY.get(x, x),
            index=list(AZURE_VOICE_MAP.keys()).index("te-IN"),
            help="Used as the primary STT hint. Auto-detection still runs on top.",
        )

        audio_recorded = None
        try:
            audio_recorded = st.audio_input("Record your question")
        except AttributeError:
            st.info("Audio recording requires Streamlit ≥ 1.31. Use upload below.")

        audio_uploaded = st.file_uploader(
            "Or upload an audio file",
            type=["wav", "mp3", "ogg", "aac", "m4a", "opus", "webm"],
        )

    audio_source = audio_recorded or audio_uploaded

    if audio_source:
        if st.button("Process Voice", type="primary", use_container_width=True):
            audio_bytes = audio_source.read()
            content_type = "audio/wav"
            if audio_uploaded:
                fname = getattr(audio_uploaded, "name", "")
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "wav"
                content_type = {
                    "wav": "audio/wav", "ogg": "audio/ogg; codecs=opus",
                    "opus": "audio/ogg; codecs=opus", "mp3": "audio/mpeg",
                    "aac": "audio/aac", "m4a": "audio/mp4", "webm": "audio/webm",
                }.get(ext, "audio/wav")

            with st.spinner("Transcribing audio..."):
                transcript, lang_code = azure_stt(audio_bytes, content_type, preferred_lang)

            if not transcript:
                st.error("Could not transcribe the audio. Please try again.")
            else:
                st.success(f"**Transcript:** {transcript}")
                st.info(f"**Detected language:** {_LANG_DISPLAY.get(lang_code, lang_code)}")
                _display_results(run_pipeline(transcript, lang_code))

# -- Text tab -----------------------------------------------------------------
with tab_text:
    st.markdown("Type your question in **any Indian language or English**.")

    example_queries = {
        "SC student Karnataka B.Tech (English)":
            "What scholarships are available for SC students doing BTech in Karnataka?",
        "Same query in Telugu":
            "SC vidhyarthiniki Karnataka lo BTech ki ela apply cheyali?",
        "Same query in Hindi":
            "Karnataka mein SC chhatra ko BTech ke liye kaunsi yojana milegi?",
        "Farmer Maharashtra (English)":
            "What farming subsidies are available for farmers in Maharashtra?",
    }

    left, right = st.columns([1, 2])
    with left:
        choice = st.selectbox(
            "Try an example query",
            options=["(type your own)"] + list(example_queries.keys()),
        )

    with right:
        prefill = example_queries.get(choice, "")
        user_text = st.text_area(
            "Your question",
            value=prefill,
            placeholder=(
                "e.g. What scholarships are available for SC students doing BTech in Karnataka?\n"
                "or: SC vidhyarthiniki Karnataka lo ela apply cheyali?\n"
                "ya: SC chhatra Karnataka mein BTech ke liye kaunsi yojana hai?"
            ),
            height=110,
        )

    if st.button("Search Schemes", type="primary", use_container_width=False):
        query = user_text.strip()
        if not query:
            st.warning("Please enter a question first.")
        else:
            lang_code = _detect_lang_from_script(query, fallback="en-IN")
            st.info(f"**Detected language:** {_LANG_DISPLAY.get(lang_code, lang_code)}")
            _display_results(run_pipeline(query, lang_code))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Supported languages: Hindi · Telugu · Tamil · Kannada · Malayalam · "
    "Marathi · Bengali · Gujarati · Punjabi · Odia · Assamese · English  |  "
    "Production delivery via Twilio WhatsApp"
)

