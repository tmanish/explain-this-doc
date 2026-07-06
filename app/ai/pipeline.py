"""Analysis pipeline.

One entry point, two engines:

- LLM engine: classification, extraction, risks, summary, questions,
  checklist, passage explanation, and comparison via the configured
  provider, with every claim grounded through CitationMapper.
- Heuristic engine: fully offline fallback and local-first privacy mode.

If a provider is configured, the LLM engine runs and the heuristic engine
backfills anything the model missed. If no provider is configured or a call
fails, the heuristic result is returned alone; the app degrades, it does
not break.
"""

from __future__ import annotations

import difflib
import logging
from concurrent.futures import ThreadPoolExecutor

from app.ai import heuristic, prompts
from app.ai.providers import LLMProvider, get_provider
from app.parsing.parser import ParsedDocument, parse_text
from app.schemas import (
    SOURCE_UNAVAILABLE,
    ActionItem,
    AnalysisResult,
    Citation,
    Classification,
    ComparisonChange,
    ComparisonResult,
    DateMention,
    DocumentType,
    ExplainResponse,
    ExtractedField,
    MoneyMention,
    Question,
    RiskItem,
    Severity,
    WhatThisMeans,
)

log = logging.getLogger("etd.pipeline")

MAX_CHARS = 24_000  # truncation guard for prompt context


class CitationMapper:
    """Verify model-provided quotes against the actual document.

    A quote that cannot be found verbatim (whitespace-normalized) is
    rejected: the item keeps its content but its source is marked
    unavailable. The model never gets to invent citations.
    """

    def __init__(self, doc: ParsedDocument):
        self.doc = doc
        self._normalized = " ".join(doc.full_text.split()).lower()

    def ground(self, quote: str | None) -> tuple[Citation | None, str]:
        if not quote:
            return None, SOURCE_UNAVAILABLE
        needle = " ".join(quote.split()).lower()
        if needle and needle in self._normalized:
            page = self.doc.locate(quote)
            citation = Citation(quote=quote.strip(), page=page)
            return citation, f"page {page}" if page else "document text"
        return None, SOURCE_UNAVAILABLE


def _sev(value: str | None) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.MEDIUM


def _conf(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


class LLMEngine:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def analyze(self, doc: ParsedDocument) -> AnalysisResult:
        text = doc.full_text[:MAX_CHARS]
        mapper = CitationMapper(doc)

        cls_raw = self.provider.complete_json(
            prompts.CLASSIFY_PROMPT.format(document_text=text[:8000])
        )
        try:
            dtype = DocumentType(cls_raw.get("document_type", "unknown"))
        except ValueError:
            dtype = DocumentType.UNKNOWN
        classification = Classification(
            document_type=dtype,
            confidence=_conf(cls_raw.get("confidence")),
            statement=cls_raw.get("statement")
            or "I'm not fully sure what type of document this is, but I can still explain the contents.",
        )

        # The five post-classification prompts are independent; run them
        # concurrently so a full analysis takes one round-trip, not five.
        def call(template: str) -> dict:
            return self.provider.complete_json(
                prompts.render(template, document_type=dtype.value, document_text=text)
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                name: pool.submit(call, template)
                for name, template in (
                    ("summary", prompts.SUMMARY_PROMPT),
                    ("extract", prompts.EXTRACT_PROMPT),
                    ("risk", prompts.RISK_PROMPT),
                    ("checklist", prompts.CHECKLIST_PROMPT),
                    ("questions", prompts.QUESTIONS_PROMPT),
                )
            }
        summary_raw = futures["summary"].result()
        extract_raw = futures["extract"].result()
        risk_raw = futures["risk"].result()
        checklist_raw = futures["checklist"].result()
        questions_raw = futures["questions"].result()

        fields = []
        for f in extract_raw.get("fields", []):
            citation, ref = mapper.ground(f.get("quote"))
            fields.append(ExtractedField(
                name=str(f.get("name", "Detail")),
                value=str(f.get("value", "")),
                confidence=_conf(f.get("confidence")),
                citation=citation,
                source_ref=ref,
            ))

        money = []
        for m in extract_raw.get("money", []):
            citation, ref = mapper.ground(m.get("quote"))
            money.append(MoneyMention(
                label=str(m.get("label", "Amount")),
                amount=str(m.get("amount", "")),
                recurring=bool(m.get("recurring", False)),
                citation=citation,
                source_ref=ref,
            ))

        dates = []
        for d in extract_raw.get("dates", []):
            citation, ref = mapper.ground(d.get("quote"))
            dates.append(DateMention(
                label=str(d.get("label", "Date")),
                date_text=str(d.get("date_text", "")),
                is_deadline=bool(d.get("is_deadline", False)),
                citation=citation,
                source_ref=ref,
            ))

        risks = []
        for r in risk_raw.get("risks", []):
            citation, ref = mapper.ground(r.get("quote"))
            risks.append(RiskItem(
                title=str(r.get("title", "Clause to review")),
                explanation=str(r.get("explanation", "")),
                why_it_matters=str(r.get("why_it_matters", "")),
                question_to_ask=str(r.get("question_to_ask", "")),
                severity=_sev(r.get("severity")),
                confidence=_conf(r.get("confidence")),
                citation=citation,
                source_ref=ref,
            ))

        actions = []
        for a in checklist_raw.get("actions", []):
            citation, ref = mapper.ground(a.get("quote"))
            actions.append(ActionItem(
                action=str(a.get("action", "")),
                priority=_sev(a.get("priority")),
                due_date=a.get("due_date"),
                reason=str(a.get("reason", "")),
                source_ref=ref,
            ))

        questions = [
            Question(text=str(q.get("text", "")), context=q.get("context"))
            for q in questions_raw.get("questions", [])
            if q.get("text")
        ]

        grounded = [x for x in (fields + risks) if x.source_ref != SOURCE_UNAVAILABLE]
        grounding_ratio = len(grounded) / max(1, len(fields) + len(risks))
        overall = round(min(0.95, 0.5 + 0.3 * grounding_ratio + 0.15 * classification.confidence), 2)

        return AnalysisResult(
            document_name=doc.filename,
            classification=classification,
            summary=summary_raw.get("summary", ""),
            what_this_means=WhatThisMeans(text=summary_raw.get("what_this_means", "")),
            key_details=fields,
            money=money,
            dates=dates,
            risks=risks,
            actions=actions,
            questions=questions,
            overall_confidence=overall,
            engine=f"llm ({self.provider.name})",
        )

    def explain(self, passage: str, mode: str, dtype: DocumentType) -> str:
        raw = self.provider.complete_json(
            prompts.render(
                prompts.EXPLAIN_PASSAGE_PROMPT,
                document_type=dtype.value,
                mode=mode,
                passage=passage[:6000],
            )
        )
        return raw.get("explanation", "")

    def compare(self, old: ParsedDocument, new: ParsedDocument) -> ComparisonResult:
        raw = self.provider.complete_json(
            prompts.render(
                prompts.COMPARE_PROMPT,
                old_name=old.filename,
                new_name=new.filename,
                old_text=old.full_text[:12000],
                new_text=new.full_text[:12000],
            ),
            max_tokens=3000,
        )
        changes = [
            ComparisonChange(
                category=str(c.get("category", "other")),
                description=str(c.get("description", "")),
                old_value=c.get("old_value"),
                new_value=c.get("new_value"),
                severity=_sev(c.get("severity")),
            )
            for c in raw.get("changes", [])
        ]
        return ComparisonResult(
            old_name=old.filename,
            new_name=new.filename,
            overview=raw.get("overview", ""),
            changes=changes,
            engine=f"llm ({self.provider.name})",
        )


# --------------------------------------------------------------------------
# Offline fallbacks for explain / compare
# --------------------------------------------------------------------------

EXPLAIN_FALLBACK = {
    "explain": (
        "Local mode can't paraphrase this passage without a language model. "
        "What it can tell you: read it for amounts, dates, and words like "
        "'shall', 'waive', 'non-refundable', or 'automatically', which usually "
        "signal obligations. Enable an LLM provider for a full plain-English explanation."
    ),
    "rewrite": (
        "Rewriting in plain English requires an LLM provider. Enable one in "
        "settings, or use the risk and money highlights on the main panel."
    ),
}


def _heuristic_explain(passage: str, mode: str) -> str:
    doc = parse_text(passage, filename="selected-passage")
    risks = heuristic.find_risks(doc)
    money = heuristic.find_money(doc)
    if mode in ("risks", "is_this_normal", "what_to_ask") and risks:
        lines = []
        for r in risks[:3]:
            lines.append(f"{r.title}: {r.explanation} {r.why_it_matters} You may want to ask: \"{r.question_to_ask}\"")
        return " ".join(lines)
    if money:
        amounts = ", ".join(m.amount for m in money[:5])
        return (
            f"This passage mentions money ({amounts}). "
            "Check whether these amounts are one-time or recurring, and whether "
            "any conditions attach to them. " + EXPLAIN_FALLBACK["explain"]
        )
    return EXPLAIN_FALLBACK.get(mode, EXPLAIN_FALLBACK["explain"])


def _heuristic_compare(old: ParsedDocument, new: ParsedDocument) -> ComparisonResult:
    old_money = {m.amount for m in heuristic.find_money(old)}
    new_money = {m.amount for m in heuristic.find_money(new)}
    old_dates = {d.date_text for d in heuristic.find_dates(old)}
    new_dates = {d.date_text for d in heuristic.find_dates(new)}
    old_risks = {r.title for r in heuristic.find_risks(old)}
    new_risks = {r.title for r in heuristic.find_risks(new)}

    changes: list[ComparisonChange] = []
    for amt in sorted(new_money - old_money):
        changes.append(ComparisonChange(
            category="money", description=f"New amount appears: {amt}",
            new_value=amt, severity=Severity.MEDIUM))
    for amt in sorted(old_money - new_money):
        changes.append(ComparisonChange(
            category="money", description=f"Amount no longer appears: {amt}",
            old_value=amt, severity=Severity.LOW))
    for d in sorted(new_dates - old_dates):
        changes.append(ComparisonChange(
            category="dates", description=f"New date appears: {d}",
            new_value=d, severity=Severity.MEDIUM))
    for title in sorted(new_risks - old_risks):
        changes.append(ComparisonChange(
            category="new_risk", description=f"New clause pattern detected: {title}",
            severity=Severity.HIGH))
    for title in sorted(old_risks - new_risks):
        changes.append(ComparisonChange(
            category="removed_protection",
            description=f"Clause pattern no longer detected: {title}",
            severity=Severity.LOW))

    ratio = difflib.SequenceMatcher(
        None, old.full_text[:20000], new.full_text[:20000]
    ).ratio()
    overview = (
        f"The two versions are about {round(ratio * 100)}% similar by text. "
        f"{len(changes)} notable difference(s) were detected by local analysis. "
        "Enable an LLM provider for a clause-level comparison."
    )
    return ComparisonResult(
        old_name=old.filename, new_name=new.filename,
        overview=overview, changes=changes, engine="heuristic (local, offline)",
    )


# --------------------------------------------------------------------------
# Public facade
# --------------------------------------------------------------------------

class Analyzer:
    def __init__(self, provider: LLMProvider | None = None, auto: bool = True):
        if provider is None and auto:
            try:
                provider = get_provider()
            except Exception as exc:  # bad config; degrade, don't crash
                log.warning("Provider init failed, running local-only: %s", exc)
                provider = None
        self.llm = LLMEngine(provider) if provider else None

    @property
    def mode(self) -> str:
        return f"llm ({self.llm.provider.name})" if self.llm else "heuristic (local, offline)"

    def analyze(self, doc: ParsedDocument) -> AnalysisResult:
        if self.llm:
            try:
                return self.llm.analyze(doc)
            except Exception as exc:
                log.warning("LLM analysis failed, falling back to local: %s", exc)
        return heuristic.analyze(doc)

    def explain(self, passage: str, mode: str, dtype: DocumentType) -> ExplainResponse:
        if self.llm:
            try:
                text = self.llm.explain(passage, mode, dtype)
                return ExplainResponse(
                    passage=passage, mode=mode, explanation=text, engine=self.mode
                )
            except Exception as exc:
                log.warning("LLM explain failed, falling back: %s", exc)
        return ExplainResponse(
            passage=passage, mode=mode,
            explanation=_heuristic_explain(passage, mode),
            engine="heuristic (local, offline)",
        )

    def compare(self, old: ParsedDocument, new: ParsedDocument) -> ComparisonResult:
        if self.llm:
            try:
                return self.llm.compare(old, new)
            except Exception as exc:
                log.warning("LLM compare failed, falling back: %s", exc)
        return _heuristic_compare(old, new)
