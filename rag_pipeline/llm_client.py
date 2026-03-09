"""Pluggable LLM client — Groq (default), OpenAI, or AWS SageMaker.

Set LLM_PROVIDER in .env to switch backends.  No code changes needed.

Groq recommendation:
  - llama-3.3-70b-versatile: best quality, very fast (free tier: 14,400 req/day)
  - mixtral-8x7b-32768: fast, multilingual
  - gemma2-9b-it: lightweight

AWS SageMaker:
  Deploy any HuggingFace model (Mistral 7B, Llama 3.1 8B, etc.) and set:
  SAGEMAKER_LLM_ENDPOINT=your-endpoint-name
  LLM_PROVIDER=sagemaker
"""
from __future__ import annotations

from rag.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SAGEMAKER_LLM_ENDPOINT,
    AWS_REGION,
)

# ── Answer generation templates ───────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a helpful and accurate government scheme adviser for India.
Your job is to help Indian citizens find the right government welfare schemes and understand how to apply.
Answer ONLY based on the schemes provided in the context.
Be specific: include scheme names, eligibility criteria, benefit amounts, and application steps.
If no scheme in the context matches the question, clearly say so."""

_ANSWER_TEMPLATE = """Based on the following government schemes, answer the question accurately.

QUESTION: {question}

RELEVANT GOVERNMENT SCHEMES:
{schemes_context}

Provide a clear, helpful answer that includes:
- Which scheme(s) are most relevant
- Key eligibility criteria
- Benefit amount/type (with ₹ amounts if available)
- How to apply
If multiple schemes match, list them clearly."""

_QUERY_SYSTEM_PROMPT = (
    "You are a search query optimizer for an Indian government scheme database. "
    "Output ONLY what is asked, no explanations."
)


class LLMClient:
    """Thin LLM wrapper.  Single instance shared per process."""

    def __init__(self) -> None:
        self._provider = LLM_PROVIDER
        self._client = self._init_client()

    # ── Public ────────────────────────────────────────────────────────────────

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Generic completion call."""
        if self._provider == "groq":
            return self._groq_complete(system, user, max_tokens)
        if self._provider == "openai":
            return self._openai_complete(system, user, max_tokens)
        if self._provider == "sagemaker":
            return self._sagemaker_complete(system, user, max_tokens)
        raise ValueError(f"Unknown LLM provider: {self._provider!r}")

    def generate_answer(self, question: str, parent_docs: list[dict]) -> str:
        """Generate a final answer given the question and retrieved scheme docs."""
        schemes_context = _format_schemes_context(parent_docs)
        user_msg = _ANSWER_TEMPLATE.format(
            question=question,
            schemes_context=schemes_context,
        )
        return self.complete(_SYSTEM_PROMPT, user_msg, max_tokens=1024)

    def rewrite_and_expand_queries(
        self, question: str, n_variants: int, prompt: str
    ) -> str:
        """Call LLM to rewrite + generate variant queries. Returns raw text."""
        return self.complete(_QUERY_SYSTEM_PROMPT, prompt, max_tokens=256)

    # ── Provider implementations ──────────────────────────────────────────────

    def _groq_complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    def _openai_complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    def _sagemaker_complete(self, system: str, user: str, max_tokens: int) -> str:
        """SageMaker endpoint using Llama/Mistral instruction format."""
        import boto3
        import json

        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
        runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        resp = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_LLM_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens, "temperature": 0.1},
            }),
        )
        result = json.loads(resp["Body"].read())
        if isinstance(result, list):
            return result[0].get("generated_text", "").split("[/INST]")[-1].strip()
        return result.get("generated_text", "")

    def _init_client(self):
        if self._provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in .env")
            from groq import Groq
            return Groq(api_key=GROQ_API_KEY)
        if self._provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in .env")
            from openai import OpenAI
            return OpenAI(api_key=OPENAI_API_KEY)
        if self._provider == "sagemaker":
            return None  # boto3 client created on each call
        raise ValueError(f"Unknown LLM provider: {self._provider!r}")


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_schemes_context(parent_docs: list[dict]) -> str:
    """Format retrieved parent documents into a structured LLM context block."""
    blocks: list[str] = []
    for i, doc in enumerate(parent_docs, 1):
        name = doc.get("scheme_name", "Unknown Scheme")
        state = doc.get("state_or_ut", "")
        category = doc.get("scheme_category", "")
        header = f"{'─'*60}\nSCHEME {i}: {name}"
        if state:
            header += f" | {state}"
        if category:
            header += f" | {category}"

        parts = [header]
        if summary := doc.get("summary"):
            parts.append(f"Overview: {summary}")
        if eligibility := doc.get("eligibility"):
            parts.append(f"Eligibility: {eligibility}")
        if benefits := doc.get("benefits"):
            parts.append(f"Benefits: {benefits}")
        if application := doc.get("application"):
            parts.append(f"How to Apply: {application}")

        blocks.append("\n".join(parts))

    return "\n\n".join(blocks) if blocks else "No matching schemes found."
