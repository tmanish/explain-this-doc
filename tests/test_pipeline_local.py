from pathlib import Path

from app.ai import heuristic
from app.ai.pipeline import Analyzer, CitationMapper, _heuristic_compare
from app.export.reports import to_markdown
from app.parsing.parser import parse_text
from app.schemas import SOURCE_UNAVAILABLE, DocumentType, Severity

DEMO = Path(__file__).resolve().parent.parent / "demo"


def _load(name):
    return parse_text((DEMO / name).read_text(), filename=name)


def test_lease_classification_and_risks():
    doc = _load("lease.txt")
    result = heuristic.analyze(doc)
    assert result.classification.document_type == DocumentType.LEASE
    titles = {r.title for r in result.risks}
    assert "Automatic Renewal" in titles
    assert "Arbitration Clause" in titles
    assert "Early Termination Fee" in titles
    assert "Non-Refundable Payment" in titles
    amounts = {m.amount for m in result.money}
    assert "$1,850.00" in amounts
    assert result.questions and result.actions


def test_medical_bill_fields():
    doc = _load("medical_bill.txt")
    result = heuristic.analyze(doc)
    assert result.classification.document_type == DocumentType.MEDICAL_BILL
    names = {f.name: f.value for f in result.key_details}
    assert names.get("Total billed") == "$628.00"
    assert names.get("Patient responsibility") == "$140.00"
    assert any(d.is_deadline for d in result.dates)


def test_credit_card_notice():
    doc = _load("credit_card_notice.txt")
    result = heuristic.analyze(doc)
    assert result.classification.document_type == DocumentType.CREDIT_CARD_NOTICE
    assert any(r.title == "Rate or Fee Change" for r in result.risks)


def test_employment_agreement_noncompete():
    doc = _load("employment_agreement.txt")
    result = heuristic.analyze(doc)
    assert result.classification.document_type == DocumentType.EMPLOYMENT_AGREEMENT
    assert any(r.title == "Non-Compete or Non-Solicit" for r in result.risks)


def test_unknown_document_is_honest():
    doc = parse_text("A short grocery list: apples, milk, coffee.")
    result = heuristic.analyze(doc)
    assert result.classification.document_type == DocumentType.UNKNOWN
    assert "not fully sure" in result.classification.statement


def test_every_risk_has_citation_or_flag():
    doc = _load("insurance_policy.txt")
    result = heuristic.analyze(doc)
    for r in result.risks:
        assert r.citation is not None or r.source_ref == SOURCE_UNAVAILABLE


def test_citation_mapper_rejects_fabricated_quote():
    doc = parse_text("The rent is $1,850 per month.")
    mapper = CitationMapper(doc)
    citation, ref = mapper.ground("the rent is $1,850")
    assert citation is not None
    citation, ref = mapper.ground("a totally invented quote")
    assert citation is None and ref == SOURCE_UNAVAILABLE


def test_markdown_report_structure():
    result = heuristic.analyze(_load("lease.txt"))
    md = to_markdown(result)
    for heading in (
        "## Document type", "## What this document says",
        "## What this means for you", "## Money mentioned",
        "## Dates and deadlines", "## Things to check carefully",
        "## Questions you should ask", "## Action checklist", "## Disclaimer",
    ):
        assert heading in md


def test_compare_detects_money_and_risk_changes():
    old = parse_text("Rent is $1,700 per month.", filename="old-lease.txt")
    new = parse_text(
        "Rent is $1,850 per month. The lease automatically renews each year.",
        filename="new-lease.txt",
    )
    result = _heuristic_compare(old, new)
    cats = {c.category for c in result.changes}
    assert "money" in cats and "new_risk" in cats


def test_analyzer_local_mode_without_provider(monkeypatch):
    monkeypatch.setenv("ETD_PROVIDER", "none")
    analyzer = Analyzer()
    assert analyzer.mode == "heuristic (local, offline)"
    out = analyzer.explain("Payments are non-refundable.", "risks", DocumentType.UNKNOWN)
    assert "Non-Refundable" in out.explanation
