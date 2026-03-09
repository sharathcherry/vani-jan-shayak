"""session.py — Per-user conversation session management.

Each WhatsApp user gets a lightweight session file stored in S3.
The session records:
  - The language detected in their last voice note (used as the Azure STT hint
    on the next turn, so the correct script appears in the transcript)
  - A short Q&A summary for multi-turn context (not currently injected into RAG)

Sessions expire automatically after 3 hours of inactivity.
"""

import time

from config import s3, S3_BUCKET_IN, SESSION_TTL_SECONDS


def get_session_context(phone_number: str) -> tuple[str, str]:
    """Load the user's session from S3.

    Returns (context_text, preferred_lang_code).
    Returns ("", "te-IN") for new or expired sessions (te-IN is the default
    language for this deployment).

    Session file format:
        LANG:<bcp47-code>
        Q: <last english query, truncated to 200 chars>
        A: <last english answer, truncated to 300 chars>
    """
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_IN, Key=f"wa-sessions/{phone_number}.txt")
        age  = time.time() - resp["LastModified"].timestamp()

        if age > SESSION_TTL_SECONDS:
            print(f"[session] Expired ({age/3600:.1f}h old) — starting fresh.")
            s3.delete_object(Bucket=S3_BUCKET_IN, Key=f"wa-sessions/{phone_number}.txt")
            return "", "te-IN"

        raw = resp["Body"].read().decode("utf-8")
        if raw.startswith("LANG:"):
            first_line, _, rest = raw.partition("\n")
            lang_code = first_line[5:].strip() or "te-IN"
            return rest, lang_code
        return raw, "te-IN"

    except Exception:
        return "", "te-IN"


def save_session_context(
    phone_number: str,
    english_query: str,
    answer: str,
    lang_code: str = "en-IN",
) -> None:
    """Persist Q&A summary and detected language for the next turn.

    Called in a background thread (parallel to back-translation) so it
    does not add to the critical-path latency.
    """
    context = f"LANG:{lang_code}\nQ: {english_query[:200]}\nA: {answer[:300]}"
    s3.put_object(
        Bucket=S3_BUCKET_IN,
        Key=f"wa-sessions/{phone_number}.txt",
        Body=context.encode("utf-8"),
    )
