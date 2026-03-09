"""translation.py — Language translation via Amazon Bedrock Nova Lite.

Translates between any two supported Indian languages and English.
Used twice per voice query: native → English (before RAG) and
English → native (before TTS).

Caching: translated pairs are persisted to S3 (wa-translate-cache/).
On a cache hit the Bedrock call (~1-2 s) is skipped entirely.
"""

import hashlib
import time

from config import bedrock, LANG_NAMES, s3, S3_BUCKET_IN, CACHE_TTL_SECONDS

_TRANSLATE_CACHE_PREFIX = "wa-translate-cache"


def _translate_cache_get(cache_key: str) -> str | None:
    """Return cached translation from S3, or None on miss/expiry/error."""
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_IN, Key=f"{_TRANSLATE_CACHE_PREFIX}/{cache_key}.txt")
        age = time.time() - resp["LastModified"].timestamp()
        if age > CACHE_TTL_SECONDS:
            return None
        return resp["Body"].read().decode("utf-8")
    except Exception:
        return None


def _translate_cache_set(cache_key: str, text: str) -> None:
    """Write translation to S3 cache (best-effort; errors are non-fatal)."""
    try:
        s3.put_object(
            Bucket=S3_BUCKET_IN,
            Key=f"{_TRANSLATE_CACHE_PREFIX}/{cache_key}.txt",
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
    except Exception as e:
        print(f"[translate] S3 cache write failed: {e}")


def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> str:
    """Translate text between languages using Amazon Bedrock Nova Lite.

    Returns the original text unchanged if:
    - source and target languages are the same, or
    - the input text is empty, or
    - the Bedrock call fails (graceful degradation).

    Checks S3 cache before calling Bedrock and writes through on a miss.
    """
    src = LANG_NAMES.get(source_lang_code, "English")
    tgt = LANG_NAMES.get(target_lang_code, "English")

    if src == tgt or not text.strip():
        return text

    cache_key = hashlib.sha256(f"{text}|{source_lang_code}|{target_lang_code}".encode()).hexdigest()
    cached = _translate_cache_get(cache_key)
    if cached is not None:
        print(f"[translate] S3 cache hit: {src} -> {tgt}")
        return cached

    prompt = (
        f"Translate the following {src} text to {tgt}. "
        f"Output ONLY the translation, nothing else.\n\n{text}"
    )
    try:
        resp = bedrock.converse(
            modelId="eu.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 512},
        )
        translated = resp["output"]["message"]["content"][0]["text"].strip()
        print(f"[translate] {src} -> {tgt}: '{translated[:80]}'")
        if translated:
            _translate_cache_set(cache_key, translated)
            return translated
        return text
    except Exception as e:
        print(f"[translate] failed ({e}), returning original")
        return text
