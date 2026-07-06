"""Core data models for Explain This Document Like I'm Human.

Every AI-generated item carries text, confidence, and a source reference.
When a citation cannot be grounded in the document, source_ref is set to
"Source reference unavailable" rather than fabricated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

SOURCE_UNAVAILABLE = "Source reference unavailable"


class DocumentType(str, Enum):
    LEASE = "lease"
    INSURANCE_POLICY = "insurance_policy"
    MEDICAL_BILL = "medical_bill"
    CREDIT_CARD_NOTICE = "credit_card_notice"
    LOAN_DOCUMENT = "loan_document"
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    GOVERNMENT_NOTICE = "government_notice"
    SCHOOL_FORM = "school_form"
    WARRANTY = "warranty"
    SUBSCRIPTION_TERMS = "subscription_terms"
    UNKNOWN = "unknown"


DOCUMENT_TYPE_LABELS: dict[DocumentType, str] = {
    DocumentType.LEASE: "a lease agreement",
    DocumentType.INSURANCE_POLICY: "an insurance policy",
    DocumentType.MEDICAL_BILL: "a medical bill or healthcare statement",
    DocumentType.CREDIT_CARD_NOTICE: "a bank or credit card notice",
    DocumentType.LOAN_DOCUMENT: "a loan document",
    DocumentType.EMPLOYMENT_AGREEMENT: "an employment agreement",
    DocumentType.GOVERNMENT_NOTICE: "a government notice",
    DocumentType.SCHOOL_FORM: "a school form",
    DocumentType.WARRANTY: "a warranty document",
    DocumentType.SUBSCRIPTION_TERMS: "subscription terms",
    DocumentType.UNKNOWN: "an unidentified document",
}


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Citation(BaseModel):
    """Grounds a claim in the source document."""

    quote: str = Field(description="Short verbatim snippet from the document")
    page: Optional[int] = Field(default=None, description="1-based page number")
    section: Optional[str] = Field(default=None, description="Section title if known")
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    def label(self) -> str:
        parts = []
        if self.page:
            parts.append(f"page {self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts) if parts else "document text"


class ExtractedField(BaseModel):
    """A single structured detail pulled from the document."""

    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    citation: Optional[Citation] = None
    source_ref: str = SOURCE_UNAVAILABLE


class RiskItem(BaseModel):
    title: str
    explanation: str = Field(description="Plain-English explanation")
    why_it_matters: str
    question_to_ask: str
    severity: Severity = Severity.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0)
    citation: Optional[Citation] = None
    source_ref: str = SOURCE_UNAVAILABLE


class ActionItem(BaseModel):
    action: str
    priority: Severity = Severity.MEDIUM
    due_date: Optional[str] = None
    reason: str
    source_ref: str = SOURCE_UNAVAILABLE


class Question(BaseModel):
    text: str
    context: Optional[str] = Field(
        default=None, description="Why this question matters for this document"
    )


class MoneyMention(BaseModel):
    label: str
    amount: str
    recurring: bool = False
    citation: Optional[Citation] = None
    source_ref: str = SOURCE_UNAVAILABLE


class DateMention(BaseModel):
    label: str
    date_text: str
    is_deadline: bool = False
    citation: Optional[Citation] = None
    source_ref: str = SOURCE_UNAVAILABLE


class Classification(BaseModel):
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    statement: str = Field(
        description='Human phrasing, e.g. "This appears to be a lease agreement."'
    )


class WhatThisMeans(BaseModel):
    text: str
    tone_check: str = Field(
        default="Written in plain English without professional advice.",
        description="Internal note confirming the safety register",
    )


class AnalysisResult(BaseModel):
    """Complete analysis of one document."""

    document_name: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    classification: Classification
    summary: str
    what_this_means: WhatThisMeans
    key_details: list[ExtractedField] = []
    money: list[MoneyMention] = []
    dates: list[DateMention] = []
    risks: list[RiskItem] = []
    actions: list[ActionItem] = []
    questions: list[Question] = []
    overall_confidence: float = Field(ge=0.0, le=1.0)
    engine: str = Field(description="Which analysis engine produced this result")
    disclaimer: str = (
        "This explanation is for understanding only. It is not legal, medical, "
        "financial, or professional advice, and it is not a replacement for a "
        "qualified professional. For important decisions, consult a lawyer, "
        "doctor, accountant, financial advisor, insurance agent, or other "
        "relevant expert."
    )


class ExplainRequest(BaseModel):
    """Request to explain a selected passage."""

    passage: str
    mode: str = Field(
        default="explain",
        description="explain | is_this_normal | what_to_ask | risks | rewrite",
    )
    document_type: DocumentType = DocumentType.UNKNOWN


class ExplainResponse(BaseModel):
    passage: str
    mode: str
    explanation: str
    engine: str


class ComparisonChange(BaseModel):
    category: str = Field(
        description="money | dates | new_obligation | removed_protection | added_restriction | new_risk | other"
    )
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    severity: Severity = Severity.MEDIUM


class ComparisonResult(BaseModel):
    old_name: str
    new_name: str
    overview: str
    changes: list[ComparisonChange] = []
    engine: str
    disclaimer: str = (
        "This comparison highlights differences for understanding only and is "
        "not professional advice."
    )


class RedactionResult(BaseModel):
    redacted_text: str
    counts: dict[str, int] = Field(
        description="How many items of each category were masked"
    )
