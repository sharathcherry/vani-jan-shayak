"""twilio_utils.py — Twilio WhatsApp API helpers.

Provides three primitives the voice pipeline needs:

  parse_body()      — Decode the URL-encoded webhook payload from Twilio.
  twiml_reply_*()   — Build synchronous TwiML HTTP responses (sent while
                      the Lambda is still handling the webhook request).
  send_whatsapp()   — Call the Twilio REST API to push an outbound message
                      (used in the async Lambda invocation after processing).
"""

import base64
import urllib.parse

import httpx

from config import TWILIO_SID, TWILIO_TOKEN


def parse_body(event: dict) -> dict:
    """Decode the application/x-www-form-urlencoded Twilio webhook body.

    Handles both plain-text and base64-encoded API Gateway payloads.
    """
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    try:
        return dict(urllib.parse.parse_qsl(body))
    except Exception:
        return {}


def twiml_reply_text(message: str) -> dict:
    """Return a synchronous TwiML text reply to a Twilio webhook call.

    Twilio delivers this message to the user immediately when it receives
    the HTTP response, before any async processing has started.
    """
    safe = message.replace("&", "&amp;").replace("<", "").replace(">", "")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{safe}</Message>"
        "</Response>"
    )
    return {"statusCode": 200, "headers": {"Content-Type": "text/xml"}, "body": xml}


def twiml_reply_media(audio_url: str) -> dict:
    """Return a synchronous TwiML media (audio) reply to a Twilio webhook call."""
    escaped = audio_url.replace("&", "&amp;")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Message>"
        f"<Media>{escaped}</Media>"
        "</Message>"
        "</Response>"
    )
    return {"statusCode": 200, "headers": {"Content-Type": "text/xml"}, "body": xml}


def send_whatsapp(
    to_number: str,
    from_number: str,
    text: str | None = None,
    media_url: str | None = None,
) -> None:
    """Send a WhatsApp message via the Twilio REST API.

    Used in the async processing Lambda (after the webhook has already been
    acknowledged) to push the actual answer back to the user.

    Either `text` or `media_url` (or both) must be provided.
    """
    auth_b64 = base64.b64encode(f"{TWILIO_SID}:{TWILIO_TOKEN}".encode()).decode()
    data: dict = {"From": from_number, "To": to_number}
    if media_url:
        data["MediaUrl"] = media_url
    if text:
        data["Body"] = text

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            headers={"Authorization": f"Basic {auth_b64}"},
            data=data,
        )

    if resp.status_code >= 400:
        print(f"[Twilio API] send failed {resp.status_code}: {resp.text}")
    else:
        print(f"[Twilio API] sent to {to_number}")
