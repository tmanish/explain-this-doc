"""Heuristic analysis engine.

Runs entirely offline with rules and regex. This is the local-first privacy
mode and the demo mode: no text ever leaves the machine. It is honest about
what it is; confidence scores are capped lower than the LLM engine and
findings are grounded in verbatim snippets found in the text.
"""

from __future__ import annotations

import re

from app.parsing.parser import ParsedDocument
from app.schemas import (
    SOURCE_UNAVAILABLE,
    ActionItem,
    AnalysisResult,
    Citation,
    Classification,
    DateMention,
    DocumentType,
    DOCUMENT_TYPE_LABELS,
    ExtractedField,
    MoneyMention,
    Question,
    RiskItem,
    Severity,
    WhatThisMeans,
)

# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

TYPE_KEYWORDS: dict[DocumentType, list[str]] = {
    DocumentType.LEASE: [
        "lease", "landlord", "tenant", "security deposit", "premises",
        "monthly rent", "rental",
    ],
    DocumentType.MEDICAL_BILL: [
        "patient", "date of service", "insurance paid", "patient responsibility",
        "provider", "cpt", "copay", "explanation of benefits", "billing",
    ],
    DocumentType.INSURANCE_POLICY: [
        "policyholder", "premium", "deductible", "coverage", "exclusions",
        "claim", "insured", "policy period",
    ],
    DocumentType.CREDIT_CARD_NOTICE: [
        "apr", "annual percentage rate", "cardholder", "credit card",
        "minimum payment", "billing cycle", "opt out", "interest rate",
    ],
    DocumentType.LOAN_DOCUMENT: [
        "borrower", "lender", "principal", "loan", "amortization",
        "promissory", "repayment",
    ],
    DocumentType.EMPLOYMENT_AGREEMENT: [
        "employee", "employer", "employment", "compensation", "non-compete",
        "confidentiality", "termination of employment", "probation",
        "salary", "at-will",
    ],
    DocumentType.GOVERNMENT_NOTICE: [
        "department of", "internal revenue", "irs", "notice", "agency",
        "federal", "respond by", "case number",
    ],
    DocumentType.SCHOOL_FORM: [
        "school", "student", "parent", "guardian", "permission",
        "field trip", "enrollment",
    ],
    DocumentType.WARRANTY: [
        "warranty", "warrantor", "defect", "repair or replace",
        "limited warranty",
    ],
    DocumentType.SUBSCRIPTION_TERMS: [
        "subscription", "billing period", "auto-renew", "free trial",
        "cancel anytime", "monthly plan",
    ],
}


def classify(text: str) -> Classification:
    lower = text.lower()
    scores: dict[DocumentType, int] = {}
    for dtype, keywords in TYPE_KEYWORDS.items():
        scores[dtype] = sum(1 for k in keywords if k in lower)
    best = max(scores, key=scores.get)
    hits = scores[best]
    if hits == 0:
        return Classification(
            document_type=DocumentType.UNKNOWN,
            confidence=0.2,
            statement=(
                "I'm not fully sure what type of document this is, "
                "but I can still explain the contents."
            ),
        )
    confidence = min(0.85, 0.35 + 0.1 * hits)
    label = DOCUMENT_TYPE_LABELS[best]
    if confidence < 0.5:
        statement = (
            "I'm not fully sure what type of document this is, "
            "but I can still explain the contents."
        )
    else:
        statement = f"This appears to be {label}."
    return Classification(document_type=best, confidence=confidence, statement=statement)


# --------------------------------------------------------------------------
# Money and dates
# --------------------------------------------------------------------------

MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
)
RECURRING_WORDS = ("per month", "monthly", "each month", "per year",
                   "annually", "per week", "/mo", "/yr")
DEADLINE_WORDS = ("due", "deadline", "no later than", "must", "by",
                  "expires", "respond")


def _context(text: str, start: int, end: int, span: int = 90) -> str:
    return text[max(0, start - span): min(len(text), end + span)]


def _snippet(text: str, start: int, end: int, span: int = 45) -> str:
    return " ".join(text[max(0, start - span): min(len(text), end + span)].split())


def _label_for(text: str, start: int, fallback: str) -> str:
    """Label an amount/date by the clause that introduces it: the text from
    the previous sentence or line boundary up to the match."""
    window = text[max(0, start - 140): start]
    for boundary in (". ", ".\n", "\n", "; "):
        idx = window.rfind(boundary)
        if idx != -1:
            window = window[idx + len(boundary):]
            break
    label = " ".join(window.split()).strip(" .:,-")
    if len(label) < 4:
        return fallback
    return (label[:1].upper() + label[1:])[:80]


def find_money(doc: ParsedDocument) -> list[MoneyMention]:
    text = doc.full_text
    out, seen = [], set()
    for m in MONEY_RE.finditer(text):
        ctx = _context(text, m.start(), m.end())
        key = (m.group(0), ctx[:40])
        if key in seen:
            continue
        seen.add(key)
        snippet = _snippet(text, m.start(), m.end())
        out.append(
            MoneyMention(
                label=_label_for(text, m.start(), "Amount mentioned"),
                amount=m.group(0).replace(" ", ""),
                recurring=any(w in ctx.lower() for w in RECURRING_WORDS),
                citation=Citation(quote=snippet, page=doc.locate(snippet)),
                source_ref=f"page {doc.locate(snippet) or 1}",
            )
        )
    return out[:20]


def find_dates(doc: ParsedDocument) -> list[DateMention]:
    text = doc.full_text
    out, seen = [], set()
    for m in DATE_RE.finditer(text):
        if m.group(0) in seen:
            continue
        seen.add(m.group(0))
        ctx = _context(text, m.start(), m.end())
        snippet = _snippet(text, m.start(), m.end())
        out.append(
            DateMention(
                label=_label_for(text, m.start(), "Date mentioned"),
                date_text=m.group(0),
                is_deadline=any(w in ctx.lower() for w in DEADLINE_WORDS),
                citation=Citation(quote=snippet, page=doc.locate(snippet)),
                source_ref=f"page {doc.locate(snippet) or 1}",
            )
        )
    return out[:20]


# --------------------------------------------------------------------------
# Risk detection
# --------------------------------------------------------------------------

RISK_RULES: list[dict] = [
    {
        "title": "Automatic Renewal",
        "pattern": re.compile(r"auto[- ]?renew|automatically renew|renews automatically", re.I),
        "explanation": "This document appears to renew automatically unless you cancel before a specific date.",
        "why": "You could be charged again or locked in for another term if you miss the cancellation window.",
        "question": "What is the exact deadline to cancel before renewal?",
        "severity": Severity.HIGH,
    },
    {
        "title": "Late Fee",
        "pattern": re.compile(r"late (?:fee|charge|payment (?:fee|charge))", re.I),
        "explanation": "A fee appears to apply if a payment is made after the due date.",
        "why": "Missing a due date could cost you extra money on top of what you already owe.",
        "question": "How much is the late fee, and is there a grace period?",
        "severity": Severity.MEDIUM,
    },
    {
        "title": "Early Termination Fee",
        "pattern": re.compile(r"early termination|terminat\w+ (?:the )?(?:lease|agreement|contract) (?:early|before)|break(?:ing)? (?:the )?lease", re.I),
        "explanation": "Ending this agreement before its end date appears to carry a cost or penalty.",
        "why": "Leaving early could be expensive; the amount and conditions matter.",
        "question": "What exactly would I owe if I ended this agreement early?",
        "severity": Severity.HIGH,
    },
    {
        "title": "Arbitration Clause",
        "pattern": re.compile(r"arbitrat", re.I),
        "explanation": "Disputes appear to go to arbitration instead of court.",
        "why": "This can limit how you resolve disagreements, including giving up a jury trial or class action.",
        "question": "Can I opt out of the arbitration clause, and how?",
        "severity": Severity.HIGH,
    },
    {
        "title": "Waiver of Rights",
        "pattern": re.compile(r"waiv\w+ (?:of )?(?:any |all )?(?:rights?|claims?)|hereby waives?", re.I),
        "explanation": "This document appears to ask you to give up certain rights or claims.",
        "why": "Waivers can be broad; it's worth knowing exactly what you would be giving up.",
        "question": "Which specific rights am I waiving by agreeing to this?",
        "severity": Severity.HIGH,
    },
    {
        "title": "Non-Refundable Payment",
        "pattern": re.compile(r"non[- ]?refundable", re.I),
        "explanation": "At least one payment appears to be non-refundable.",
        "why": "You may not get this money back even if circumstances change.",
        "question": "Under what conditions, if any, is this payment refundable?",
        "severity": Severity.MEDIUM,
    },
    {
        "title": "Non-Compete or Non-Solicit",
        "pattern": re.compile(r"non[- ]?compete|non[- ]?solicit|shall not.{0,60}compet", re.I),
        "explanation": "This appears to restrict where you can work or who you can contact after this agreement ends.",
        "why": "These clauses can limit your future job options; scope and duration matter a lot.",
        "question": "How long does this restriction last, and what geographic area or companies does it cover?",
        "severity": Severity.HIGH,
    },
    {
        "title": "Broad Liability Language",
        "pattern": re.compile(r"indemnif|hold harmless|not (?:be )?(?:liable|responsible) for", re.I),
        "explanation": "This document appears to shift responsibility for certain losses onto you or away from the other party.",
        "why": "You could be responsible for costs you didn't expect.",
        "question": "Can you give examples of situations where I would be responsible under this clause?",
        "severity": Severity.MEDIUM,
    },
    {
        "title": "Unclear Cancellation Rules",
        "pattern": re.compile(r"cancel", re.I),
        "explanation": "Cancellation is mentioned; the steps and deadlines are worth reading closely.",
        "why": "Vague cancellation terms are a common source of surprise charges.",
        "question": "What are the exact steps and deadline to cancel, and will I get written confirmation?",
        "severity": Severity.LOW,
    },
    {
        "title": "Rate or Fee Change",
        "pattern": re.compile(r"(?:apr|rate|fee)s? (?:will|shall|may) (?:change|increase)|new (?:apr|rate|fee)", re.I),
        "explanation": "A rate or fee appears to be changing.",
        "why": "Changes to rates or fees affect what you pay going forward.",
        "question": "Is this change permanent or promotional, and can I opt out?",
        "severity": Severity.MEDIUM,
    },
]


def find_risks(doc: ParsedDocument) -> list[RiskItem]:
    text = doc.full_text
    out = []
    for rule in RISK_RULES:
        m = rule["pattern"].search(text)
        if not m:
            continue
        snippet = _snippet(text, m.start(), m.end())
        page = doc.locate(snippet)
        out.append(
            RiskItem(
                title=rule["title"],
                explanation=rule["explanation"],
                why_it_matters=rule["why"],
                question_to_ask=rule["question"],
                severity=rule["severity"],
                confidence=0.6,
                citation=Citation(quote=snippet, page=page),
                source_ref=f"page {page}" if page else SOURCE_UNAVAILABLE,
            )
        )
    order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
    return sorted(out, key=lambda r: order[r.severity])


# --------------------------------------------------------------------------
# Type-specific fields (best-effort keyword proximity)
# --------------------------------------------------------------------------

FIELD_HINTS: dict[DocumentType, list[tuple[str, re.Pattern]]] = {
    DocumentType.LEASE: [
        ("Monthly rent", re.compile(r"(?:monthly rent|rent)(?:\s+\w+){0,8}?\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
        ("Security deposit", re.compile(r"security deposit(?:\s+\w+){0,8}?\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
        ("Late fee", re.compile(r"late (?:fee|charge)(?:\s+\w+){0,8}?\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
    ],
    DocumentType.MEDICAL_BILL: [
        ("Total billed", re.compile(r"total (?:billed|charges?)\s*:?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
        ("Insurance paid", re.compile(r"insurance (?:paid|payment|adjustment)s?\s*:?\s*-?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
        ("Patient responsibility", re.compile(r"(?:patient responsibility|amount (?:you owe|due))\s*:?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
    ],
    DocumentType.INSURANCE_POLICY: [
        ("Premium", re.compile(r"premium\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
        ("Deductible", re.compile(r"deductible\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
    ],
    DocumentType.CREDIT_CARD_NOTICE: [
        ("New APR", re.compile(r"(?:new|revised)?\s*apr\s*(?:of|will be|is|:)?\s*([\d.]+%)", re.I)),
        ("Annual fee", re.compile(r"annual fee\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
    ],
    DocumentType.EMPLOYMENT_AGREEMENT: [
        ("Compensation", re.compile(r"(?:salary|compensation)\s*(?:of|is|:)?\s*(\$[\d,]+(?:\.\d{2})?)", re.I)),
        ("Probation period", re.compile(r"probation(?:ary)? period\s*(?:of|is|:)?\s*([\w\s()]{3,30}?(?:days|months))", re.I)),
    ],
}


def find_fields(doc: ParsedDocument, dtype: DocumentType) -> list[ExtractedField]:
    text = doc.full_text
    out = []
    for name, pattern in FIELD_HINTS.get(dtype, []):
        m = pattern.search(text)
        if not m:
            continue
        snippet = _snippet(text, m.start(), m.end(), span=30)
        page = doc.locate(snippet)
        out.append(
            ExtractedField(
                name=name,
                value=m.group(1).strip(),
                confidence=0.65,
                citation=Citation(quote=snippet, page=page),
                source_ref=f"page {page}" if page else SOURCE_UNAVAILABLE,
            )
        )
    return out


# --------------------------------------------------------------------------
# Summary, actions, questions
# --------------------------------------------------------------------------

SUMMARY_TEMPLATES: dict[DocumentType, str] = {
    DocumentType.LEASE: (
        "This document is a rental lease. It covers your rent, deposit, lease "
        "term, fees, and responsibilities, and what happens if the lease ends early."
    ),
    DocumentType.MEDICAL_BILL: (
        "This document is a medical bill or healthcare statement. It shows what "
        "was billed, what insurance covered, and what you may owe."
    ),
    DocumentType.INSURANCE_POLICY: (
        "This document is an insurance policy. It describes what is covered, "
        "what is excluded, what you pay, and how claims work."
    ),
    DocumentType.CREDIT_CARD_NOTICE: (
        "This document is a notice from a bank or card issuer. It describes "
        "changes to your account terms, such as rates, fees, or due dates."
    ),
    DocumentType.EMPLOYMENT_AGREEMENT: (
        "This document is an employment agreement. It covers your role, pay, "
        "and the rules that apply during and after your employment."
    ),
}


def analyze(doc: ParsedDocument) -> AnalysisResult:
    classification = classify(doc.full_text)
    dtype = classification.document_type

    money = find_money(doc)
    dates = find_dates(doc)
    risks = find_risks(doc)
    fields = find_fields(doc, dtype)

    summary = SUMMARY_TEMPLATES.get(
        dtype,
        "This document contains terms, amounts, and dates summarized below. "
        "The sections that mention money, deadlines, or obligations deserve "
        "the most attention.",
    )

    deadlines = [d for d in dates if d.is_deadline]
    wtm_parts = [f"This appears to be {DOCUMENT_TYPE_LABELS[dtype]}."]
    if money:
        wtm_parts.append(f"It mentions {len(money)} amount(s) of money.")
    if deadlines:
        wtm_parts.append(
            f"It appears to include {len(deadlines)} date(s) that may be deadlines."
        )
    if risks:
        wtm_parts.append(
            f"{len(risks)} clause(s) are worth reading carefully before you sign or respond."
        )
    wtm_parts.append("A professional can confirm anything that affects an important decision.")

    actions: list[ActionItem] = []
    for d in deadlines[:5]:
        actions.append(
            ActionItem(
                action=f"Check the date '{d.date_text}' ({d.label.lower()}) and note it in your calendar.",
                priority=Severity.HIGH,
                due_date=d.date_text,
                reason="This date appears to be a deadline in the document.",
                source_ref=d.source_ref,
            )
        )
    for r in risks:
        if r.severity == Severity.HIGH:
            actions.append(
                ActionItem(
                    action=f"Read the '{r.title}' clause carefully and ask about it before agreeing.",
                    priority=Severity.HIGH,
                    reason=r.why_it_matters,
                    source_ref=r.source_ref,
                )
            )
    actions.append(
        ActionItem(
            action="Save a copy of this document for your records.",
            priority=Severity.LOW,
            reason="You may need to refer back to the exact wording later.",
        )
    )

    questions = [Question(text=r.question_to_ask, context=r.title) for r in risks]
    if not questions:
        questions = [
            Question(
                text="Are there any fees, deadlines, or obligations not listed in this document?",
                context="General",
            )
        ]

    overall = round(
        min(0.75, 0.3 + 0.05 * (len(fields) + len(money) + len(dates))), 2
    )

    return AnalysisResult(
        document_name=doc.filename,
        classification=classification,
        summary=summary,
        what_this_means=WhatThisMeans(text=" ".join(wtm_parts)),
        key_details=fields,
        money=money,
        dates=dates,
        risks=risks,
        actions=actions,
        questions=questions[:8],
        overall_confidence=overall,
        engine="heuristic (local, offline)",
    )
