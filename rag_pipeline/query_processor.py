"""Query processing: expansion, rewriting, multi-query, and metadata extraction.

All LLM calls are batched into a single API call to minimise latency.
Rule-based expansion is always applied (zero latency).
"""
from __future__ import annotations

import re

# ── Domain synonym expansion dictionary ──────────────────────────────────────
# Adds related terms before retrieval to improve BM25 recall for Indian
# government scheme terminology (English + transliterated Hindi).

EXPANSION_DICT: dict[str, str] = {
    # Occupation / sector
    "farmer":       "kisan agriculture crop subsidy farming",
    "kisan":        "farmer agriculture crop subsidy farming",
    "weaver":       "handloom bunkar textile artisan weaving",
    "bunkar":       "weaver handloom textile artisan",
    "fisherman":    "fisheries fishing fisher marine coastal",
    "artisan":      "craftsman handicraft skill traditional",
    # Beneficiary groups
    "widow":        "widowed destitute women pension bereaved",
    "disabled":     "disability handicapped divyang differently abled specially abled",
    "divyang":      "disabled handicapped disability differently abled",
    "student":      "education scholarship school college university",
    "youth":        "young unemployed skill development",
    "women":        "woman female girl mahila",
    "mahila":       "women woman female girl",
    "senior citizen": "elderly old age pension",
    "veteran":      "ex-servicemen armed forces defence",
    # Benefit types
    "scholarship":  "stipend financial assistance education grant",
    "pension":      "monthly allowance retirement old age financial support",
    "subsidy":      "financial assistance grant incentive reduction",
    "loan":         "credit finance borrowing interest",
    # Social categories
    "sc":           "scheduled caste dalit backward",
    "st":           "scheduled tribe tribal adivasi",
    "obc":          "other backward class backward community",
    "bpl":          "below poverty line poor low income",
    # Sectors
    "health":       "medical healthcare hospital insurance treatment",
    "housing":      "house shelter accommodation awas pradhan mantri",
    "skill":        "training vocational apprenticeship",
    # Common abbreviations
    "pm":           "pradhan mantri prime minister central government",
    "pmay":         "pradhan mantri awas yojana housing scheme",
    "pmkvy":        "pradhan mantri kaushal vikas yojana skill training",
    "pm kisan":     "pradhan mantri kisan samman nidhi farmer income support",
    "msme":         "micro small medium enterprise startup business",
    "startup":      "new business enterprise entrepreneurship msme",
}

# ── Indian states / UTs canonical mapping ─────────────────────────────────────
# Maps common user inputs (lowercase) → official name stored in the index.
STATE_MAP: dict[str, str] = {
    "andhra pradesh": "Andhra Pradesh",
    "ap": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "hp": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "tn": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
    "wb": "West Bengal",
    "andaman": "Andaman and Nicobar Islands",
    "chandigarh": "Chandigarh",
    "dadra": "Dadra and Nagar Haveli and Daman and Diu",
    "daman": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi": "Delhi",
    "jammu": "Jammu & Kashmir",
    "kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
}

# ── Beneficiary keyword detector ──────────────────────────────────────────────
_BENEFICIARY_PATTERNS: list[tuple[str, list[str]]] = [
    ("for_farmers",  ["farmer", "kisan", "agricultur", "farming", "crop"]),
    ("for_women",    ["women", "woman", "female", "girl", "mahila", "widow"]),
    ("for_disabled", ["disab", "handicap", "divyang", "differently abled", "specially abled"]),
    ("for_sc_st",    ["scheduled caste", "scheduled tribe", "dalit", "tribal", "adivasi", r"\bsc\b", r"\bst\b"]),
    ("for_students", ["student", "scholarship", "school", "college", "universit", "educat"]),
]


# ── Public API ────────────────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """Add domain synonyms to the query before retrieval (rule-based, ~0ms)."""
    q_lower = query.lower()
    extras: list[str] = []
    for keyword, expansion in EXPANSION_DICT.items():
        if keyword in q_lower:
            extras.append(expansion)
    if extras:
        return query + " " + " ".join(extras)
    return query


def extract_metadata_filters(query: str) -> dict:
    """
    Parse structured metadata filters from free-text query.
    Returns a dict compatible with retriever._passes_filter().
    Empty dict = no filter (search everything).
    """
    q_lower = query.lower()
    filters: dict = {}

    # State detection (longest match first to avoid "up" matching "support")
    for state_key in sorted(STATE_MAP, key=len, reverse=True):
        if re.search(r"\b" + re.escape(state_key) + r"\b", q_lower):
            filters["state"] = STATE_MAP[state_key]
            break

    # Beneficiary flags
    for flag, patterns in _BENEFICIARY_PATTERNS:
        for pat in patterns:
            if re.search(pat, q_lower):
                filters[flag] = True
                break

    return filters


def build_query_prompt(question: str, n_variants: int) -> str:
    """
    Prompt that generates a rewritten query AND n_variants alternatives
    in a single LLM call.  Returns a parseable plain-text block.
    """
    return f"""You are a search query optimizer for an Indian government scheme database.

Given this user question, output:
1. A clearer rewritten version using official terminology
2. {n_variants} alternative queries that find related schemes

Question: {question}

Output format (exact, no extra text):
REWRITTEN: <rewritten query>
VARIANT1: <variant 1>
{"VARIANT2: <variant 2>" if n_variants >= 2 else ""}

Rules:
- Use terms like "scheme", "yojana", "subsidy", "pension", "scholarship" where relevant
- Include beneficiary groups: farmer, woman, student, disabled, SC/ST, youth
- Keep each query under 20 words
- No explanations, just the queries"""


def parse_query_variants(llm_output: str, n_variants: int) -> tuple[str, list[str]]:
    """Parse the LLM output from build_query_prompt into (rewritten, [variants])."""
    rewritten = ""
    variants: list[str] = []

    for line in llm_output.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("REWRITTEN:"):
            rewritten = line.split(":", 1)[1].strip()
        elif line.upper().startswith("VARIANT"):
            val = line.split(":", 1)[1].strip() if ":" in line else ""
            if val:
                variants.append(val)

    # Fallback: if parsing fails, use the original
    if not rewritten:
        rewritten = llm_output.strip().splitlines()[0] if llm_output.strip() else ""

    return rewritten, variants[:n_variants]
