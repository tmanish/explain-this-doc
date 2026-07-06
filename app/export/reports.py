"""Report export: Markdown and JSON.

The Markdown report follows the required structure: name, type, summary,
key details, money, dates, risks, questions, checklist, disclaimer, sources.
"""

from __future__ import annotations

from app.schemas import SOURCE_UNAVAILABLE, AnalysisResult, Severity

PRIORITY_MARK = {Severity.HIGH: "HIGH", Severity.MEDIUM: "MEDIUM", Severity.LOW: "LOW"}


def to_markdown(result: AnalysisResult) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# {result.document_name}")
    add("")
    add(f"*Analyzed {result.analyzed_at.strftime('%B %d, %Y')} "
        f"with the {result.engine} engine. "
        f"Overall confidence: {int(result.overall_confidence * 100)}%.*")
    add("")
    add("## Document type")
    add("")
    add(result.classification.statement)
    add("")
    add("## What this document says")
    add("")
    add(result.summary)
    add("")
    add("## What this means for you")
    add("")
    add(result.what_this_means.text)

    if result.key_details:
        add("")
        add("## What matters most")
        add("")
        for f in result.key_details:
            add(f"- **{f.name}:** {f.value} "
                f"(confidence {int(f.confidence * 100)}%, {f.source_ref})")

    if result.money:
        add("")
        add("## Money mentioned")
        add("")
        for m in result.money:
            rec = ", recurring" if m.recurring else ""
            add(f"- **{m.amount}**{rec}: {m.label} ({m.source_ref})")

    if result.dates:
        add("")
        add("## Dates and deadlines")
        add("")
        for d in result.dates:
            mark = " **(possible deadline)**" if d.is_deadline else ""
            add(f"- **{d.date_text}**{mark}: {d.label} ({d.source_ref})")

    if result.risks:
        add("")
        add("## Things to check carefully")
        add("")
        for r in result.risks:
            add(f"### {r.title} ({PRIORITY_MARK[r.severity]})")
            add("")
            add(r.explanation)
            add("")
            add(f"**Why it matters:** {r.why_it_matters}")
            add("")
            add(f"**Question to ask:** \"{r.question_to_ask}\"")
            add("")
            add(f"*Source: {r.source_ref}. Confidence {int(r.confidence * 100)}%.*")
            add("")

    if result.questions:
        add("## Questions you should ask before you sign or respond")
        add("")
        for q in result.questions:
            ctx = f" ({q.context})" if q.context else ""
            add(f"- {q.text}{ctx}")

    if result.actions:
        add("")
        add("## Action checklist")
        add("")
        for a in result.actions:
            due = f" Due: {a.due_date}." if a.due_date else ""
            add(f"- [ ] **[{PRIORITY_MARK[a.priority]}]** {a.action}{due} {a.reason}")

    add("")
    add("## Disclaimer")
    add("")
    add(result.disclaimer)

    cited = [
        r for r in result.risks if r.citation
    ] + [f for f in result.key_details if f.citation]
    if cited:
        add("")
        add("## Source references")
        add("")
        for item in cited:
            title = getattr(item, "title", None) or getattr(item, "name", "Item")
            add(f"- **{title}:** \"{item.citation.quote}\" ({item.citation.label()})")
    add("")
    return "\n".join(lines)


def to_json(result: AnalysisResult) -> str:
    return result.model_dump_json(indent=2)
