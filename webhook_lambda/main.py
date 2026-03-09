"""main.py — AWS Lambda entry point for the Gov Schemes Voice Bot.

Architecture overview
─────────────────────
WhatsApp user
    │  voice note
    ▼
Twilio ──webhook──► API Gateway ──► vani-jan-webhook (this Lambda)
                                         │
                                         ├─ ACK Twilio immediately (<100 ms)
                                         │  (avoids Twilio's 15-second webhook timeout)
                                         │
                                         └─ fire async Lambda invocation ──► _handle_voice_async()
                                                                                  │
                                              ┌───────────────────────────────────┤
                                              │  Parallel A                       │
                                              │  session read ◄──S3               │
                                              │  audio download ◄──Twilio CDN     │
                                              └───────────────────────────────────┤
                                                                                  │
                                              [STT]  Azure Cognitive Services     │
                                              [translate →EN]  Bedrock Nova Lite  │
                                              [RAG]   gov-schemes-voice-rag Lambda│
                                                       └─ Qdrant hybrid search    │
                                                       └─ Cohere rerank           │
                                                       └─ Bedrock answer gen      │
                                              [translate →native]  Bedrock        │
                                                                                  │
                                              ┌───────────────────────────────────┤
                                              │  Parallel B                       │
                                              │  session save ──► S3              │
                                              │  (runs while back-translation     │
                                              │   is in progress)                 │
                                              └───────────────────────────────────┤
                                                                                  │
                                              send TEXT answer  ──► Twilio ──► WhatsApp
                                              [TTS]  Azure Neural TTS ──► S3     │
                                              send AUDIO answer ──► Twilio ──► WhatsApp
"""

import json
import os
import time
import threading

from config import lambda_client, s3, S3_BUCKET_IN
from stt import azure_stt, download_twilio_audio, detect_lang_from_script
from tts import synthesize_speech
from translation import translate_text
from rag import get_rag_answer, has_scheme_intent, extract_urls, strip_urls
from session import get_session_context, save_session_context
from twilio_utils import parse_body, twiml_reply_text, send_whatsapp
from greetings import _LANG_MENU, _LANG_SELECTION, _LANG_CONFIRM


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """Route incoming events to the correct handler.

    Two invocation modes:
      Sync  — Twilio webhook (HTTP POST from API Gateway).
              Must respond within Twilio's 15-second timeout.
              Immediately fires an async Lambda for heavy processing.
      Async — Self-invoked with `_async_process: True`.
              Runs the full voice pipeline and pushes results to WhatsApp.
    """
    # Async branch: invoked by itself after the webhook was acknowledged
    if event.get("_async_process"):
        _handle_voice_async(event)
        return {"statusCode": 200, "body": "ok"}

    params      = parse_body(event)
    message_sid = params.get("MessageSid", f"msg-{int(time.time())}")
    raw_phone   = params.get("From", "unknown")
    phone_id    = raw_phone.replace("+", "").replace(":", " ")
    media_url   = params.get("MediaUrl0")
    user_text   = params.get("Body", "").strip()

    # ── Voice note received ──────────────────────────────────────────────
    if media_url:
        print(f"Processing Voice Note from: {media_url}")
        fn_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "vani-jan-webhook")
        lambda_client.invoke(
            FunctionName=fn_name,
            InvocationType="Event",          # fire-and-forget
            Payload=json.dumps({
                "_async_process": True,
                "media_url":   media_url,
                "message_sid": message_sid,
                "from_number": params.get("From", ""),
                "to_number":   params.get("To", ""),
                "phone_id":    phone_id,
            }),
        )
        # ACK Twilio immediately — this text reaches the user in ~3-5 s
        return twiml_reply_text(
            "Got your voice note! Searching for schemes, reply in a moment..."
        )

    # ── Sandbox join / greeting ──────────────────────────────────────────
    if user_text.lower().startswith("join ") or user_text.strip().lower() in (
        "hi", "hello", "start", "help", "language", "change language"
    ):
        return twiml_reply_text(_LANG_MENU)

    # ── Language selection (user replies "1"–"12" or a language name) ───
    _sel_key = user_text.strip().lower()
    if _sel_key in _LANG_SELECTION:
        chosen_lang = _LANG_SELECTION[_sel_key]
        s3.put_object(
            Bucket=S3_BUCKET_IN,
            Key=f"wa-sessions/{phone_id}.txt",
            Body=f"LANG:{chosen_lang}\n".encode("utf-8"),
        )
        print(f"[lang-select] {phone_id} chose {chosen_lang}")
        return twiml_reply_text(_LANG_CONFIRM[chosen_lang])

    # ── Text query fallback ──────────────────────────────────────────────
    if not user_text:
        return twiml_reply_text(_LANG_MENU)

    lang_code = detect_lang_from_script(user_text, fallback="en-IN")
    print(f"[text] lang={lang_code} query='{user_text[:80]}'")

    english_query = (
        translate_text(user_text, source_lang_code=lang_code, target_lang_code="en-IN")
        if lang_code != "en-IN"
        else user_text
    )

    english_ans = get_rag_answer(english_query)

    native_ans = (
        translate_text(english_ans, source_lang_code="en-IN", target_lang_code=lang_code)
        if lang_code != "en-IN"
        else english_ans
    )

    links = extract_urls(english_ans)
    reply = strip_urls(native_ans)
    if links:
        reply += "\n\n🔗 Official link(s):\n" + "\n".join(links)

    save_session_context(phone_id, english_query, english_ans, lang_code)
    return twiml_reply_text(reply)


# ---------------------------------------------------------------------------
# Async voice pipeline
# ---------------------------------------------------------------------------

def _handle_voice_async(event: dict) -> None:
    """Full voice processing pipeline — runs in a separate async Lambda invocation.

    Threading is used for two I/O-parallel operations:
      Parallel A — session S3 read runs while Twilio audio is downloading  (~1-2 s saved)
      Parallel B — session S3 write runs while back-translation is running  (~1 s saved)
    """
    media_url   = event["media_url"]
    message_sid = event["message_sid"]
    from_number = event["from_number"]
    to_number   = event["to_number"]
    phone_id    = event["phone_id"]

    # ── Parallel A: load session + download audio ────────────────────────
    session_result = ["", "te-IN"]   # [context_text, preferred_lang]

    def _load_session():
        ctx, lc = get_session_context(phone_id)
        session_result[0], session_result[1] = ctx, lc

    session_thread = threading.Thread(target=_load_session, daemon=True)
    session_thread.start()

    audio_bytes, content_type = download_twilio_audio(media_url)
    try:
        # Archive raw audio for debugging / replay
        s3.put_object(
            Bucket=S3_BUCKET_IN,
            Key=f"wa-in/{message_sid}.ogg",
            Body=audio_bytes,
        )
    except Exception:
        pass

    session_thread.join(timeout=5.0)
    preferred_lang = session_result[1]   # language from user's last turn

    # ── Step 1: Speech-to-Text ───────────────────────────────────────────
    native_query, lang_code = azure_stt(
        audio_bytes, content_type, preferred_lang=preferred_lang
    )
    print(f"[STT] lang={lang_code} text='{native_query[:80]}'")

    if not native_query:
        send_whatsapp(
            from_number, to_number,
            text="Sorry, I couldn't understand your voice note. Please try again.",
        )
        return

    # ── Step 2: Translate native → English ───────────────────────────────
    english_query = translate_text(
        native_query, source_lang_code=lang_code, target_lang_code="en-IN"
    )
    print(f"[translate->EN] '{english_query[:80]}'")

    # ── Step 2.5: Intent guard ───────────────────────────────────────────
    # Skip RAG if the query has no scheme-related keywords — avoids
    # returning a random scheme for greetings or off-topic questions.
    if not has_scheme_intent(english_query):
        print(f"[intent] No scheme intent in: '{english_query[:80]}'")
        clarify_en = (
            "I didn't catch a specific question about a government scheme. "
            "Could you ask something like: 'What scholarships are available "
            "for SC students?' or 'How do I apply for PM Kisan Yojana?'"
        )
        send_whatsapp(
            from_number, to_number,
            text=translate_text(
                clarify_en, source_lang_code="en-IN", target_lang_code=lang_code
            ),
        )
        return

    # ── Step 3: RAG — 3-layer cache (memory → S3 → RAG Lambda) ──────────
    english_answer = get_rag_answer(english_query)
    print(f"[RAG] answer='{english_answer[:80]}'")

    # ── Parallel B: save session + back-translate ────────────────────────
    save_thread = threading.Thread(
        target=save_session_context,
        args=(phone_id, english_query, english_answer, lang_code),
        daemon=True,
    )
    save_thread.start()

    # ── Step 4: Translate English → user's language ──────────────────────
    native_answer = translate_text(
        english_answer, source_lang_code="en-IN", target_lang_code=lang_code
    )
    print(f"[translate->{lang_code}] '{native_answer[:80]}'")

    links    = extract_urls(english_answer)   # extract before stripping
    tts_text = strip_urls(native_answer)      # URLs read aloud sound terrible

    save_thread.join(timeout=5.0)   # ensure session is persisted before Lambda exits

    # ── Step 5: Send TEXT answer immediately ─────────────────────────────
    # The user receives readable text while TTS is still generating (~1.5 s).
    # This reduces perceived latency from ~15 s to ~10 s "first response".
    send_whatsapp(from_number, to_number, text=tts_text)
    print("[text-first] text answer sent, starting TTS")

    # ── Step 6: TTS → S3 → send audio ────────────────────────────────────
    try:
        audio_url = synthesize_speech(tts_text, message_sid, lang_code=lang_code)
        send_whatsapp(from_number, to_number, media_url=audio_url)
    except Exception as e:
        print(f"[TTS] failed ({e}), text was already sent above")

    # ── Step 7: Send clickable scheme links ───────────────────────────────
    if links:
        link_msg = "🔗 Official link(s):\n" + "\n".join(links)
        print(f"[links] sending {len(links)} link(s)")
        send_whatsapp(from_number, to_number, text=link_msg)
