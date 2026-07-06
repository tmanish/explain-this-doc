"""Redaction mode.

Detects and masks sensitive identifiers before text is stored, exported,
or sent to a remote model. Regex-based and conservative: prefers masking a
little too much over leaking anything.
"""

from __future__ import annotations

import re

from app.schemas import RedactionResult

# Order matters: more specific patterns run before generic number patterns.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    )),
    ("policy_or_claim", re.compile(
        r"\b(?:policy|claim|member|patient|account|acct|employee|routing)"
        r"(?:\s*(?:no\.?|number|#|id))?\s*[:#]?\s*[A-Z0-9][A-Z0-9-]{5,}\b",
        re.IGNORECASE,
    )),
    ("street_address", re.compile(
        r"\b\d{1,6}\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}\s+"
        r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|"
        r"Drive|Dr\.?|Court|Ct\.?|Way|Place|Pl\.?|Circle|Cir\.?)\b"
    )),
    ("long_id", re.compile(r"\b\d{9,}\b")),
]

MASKS = {
    "ssn": "[SSN REDACTED]",
    "credit_card": "[CARD NUMBER REDACTED]",
    "email": "[EMAIL REDACTED]",
    "phone": "[PHONE REDACTED]",
    "policy_or_claim": "[ID REDACTED]",
    "street_address": "[ADDRESS REDACTED]",
    "long_id": "[NUMBER REDACTED]",
}


def _luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    result = text

    for name, pattern in PATTERNS:
        def _sub(match: re.Match, _name=name) -> str:
            token = match.group(0)
            if _name == "credit_card":
                digits = re.sub(r"\D", "", token)
                if not (13 <= len(digits) <= 16 and _luhn_valid(digits)):
                    return token  # not a card; leave for later patterns
            counts[_name] = counts.get(_name, 0) + 1
            return MASKS[_name]

        result = pattern.sub(_sub, result)

    return RedactionResult(redacted_text=result, counts=counts)
