# Product Notes

## 1. Product concept

Explain This Document Like I'm Human is an AI-powered document explainer for regular people. It takes the paperwork that runs a person's life (leases, medical bills, insurance policies, bank notices, employment agreements, government letters) and returns a calm, plain-English breakdown: what the document says, the money and deadlines inside it, the clauses worth reading twice, a prioritized action checklist, and the specific questions worth asking before signing or responding. Every claim is grounded in a verbatim quote from the document or explicitly marked as ungrounded; the tool never gives professional advice, only understanding.

## 2. User problems it solves

- People sign documents they do not understand because the alternative (paying a professional to read a two-page notice) is disproportionate.
- Critical facts (deadlines, fees, auto-renewals, waived rights) are buried in language designed for other professionals.
- People don't know what questions to ask, so they ask none.
- Comparing an old and new version of terms by eye is error-prone; the changes that matter are the ones designed not to stand out.
- Sensitive documents make people reluctant to use cloud tools at all; local processing removes that barrier.

## 3. Feature set

**MVP (implemented in this repo)**
- Upload PDF / image / text, plus paste
- OCR fallback for scans (optional dependency)
- Document type detection with honest uncertainty
- Plain-English summary + "what this means for you"
- Type-specific key detail extraction
- Money and deadline extraction with recurring/deadline flags
- Rule- and LLM-based risk detection with severity, why-it-matters, and a question per risk
- Prioritized action checklist
- Document-specific questions section
- Select-to-explain with five modes (explain, rewrite, is this normal, risks, what to ask)
- Two-document comparison
- Citation grounding with fabrication rejection (CitationMapper)
- Redaction mode (SSN, cards with Luhn check, phone, email, address, IDs)
- Local-first heuristic engine; opt-in storage with delete; Markdown/JSON export
- Five synthetic demo documents

**Advanced (next)**
- PDF report export; clickable citation anchors that scroll the source pane
- Side-by-side visual diff for comparison mode
- Persistent library (Postgres) behind the existing opt-in store
- Batch analysis; email-in a document

**Premium / future**
- Jurisdiction-aware context packs (informational, still not advice)
- Multi-language output
- Browser extension for terms-of-service pages
- Shared family/household document vault
- "Ask a professional" handoff that packages the analysis and questions

## 4. UX flow

1. **Intake.** One screen, one job: get the document in. Drop zone, paste, or a sample chip. A redaction toggle and a one-line privacy statement sit beside the input, not behind a settings page.
2. **Reading state.** A single quiet status line. No fake progress bars.
3. **Results.** Two panes. Left: the document text, selectable. Right: the analysis as a vertical stack of cards in a fixed order that mirrors how a person triages paperwork: what is this, what does it say, what does it mean for me, what does it cost, when is it due, what should worry me, what should I ask, what should I do.
4. **Interrogation.** Selecting text in the left pane raises a five-action popover. Answers appear in a corner panel without losing scroll position. Clicking a citation chip on any risk re-runs explain on that exact quote.
5. **Exit.** Export Markdown or JSON, or start another document. The disclaimer closes every report.

## 5. UI design direction

- Dark mode first: deep neutral ink (#101318) with soft panel steps, never pure black.
- One calm accent, a desaturated sea-glass green (#7fbfb0), used for trust moments: the type statement, section labels, primary action.
- Risk severity as a quiet left border (soft clay for high, amber for medium, gray for low), never full red cards. The product's whole personality is "calm person who read it for you"; alarm coloring would undercut it.
- Type system: DM Sans for interface language, DM Mono for everything extracted from the document (amounts, dates, quotes, confidence). The mono face is the signature: anything in mono came from the paper, anything in sans came from the explainer. That one rule makes grounding visible.
- Microcopy in plain verbs: "What this document says", "Things to check carefully", "Money mentioned", "Explain this section".
- No charts, no gauges, no chat bubble as the primary surface.

## 6. Technical architecture

See the diagram in the README. Design decisions worth recording:

- **Two engines, one Pydantic contract.** The heuristic engine is not a mock; it is the privacy tier and the reliability floor. LLM failure degrades to it silently.
- **CitationMapper is the trust boundary.** Model-supplied quotes are verified verbatim (whitespace-normalized) against the parsed text. A quote that cannot be found keeps its content but loses its citation and is labeled "Source reference unavailable." The model cannot mint evidence.
- **Prompts return one JSON object each**, parsed defensively (code-fence stripping, brace extraction). Prompt templates live in one module (`app/ai/prompts.py`) and share a safety register block and a citation rules block, so the non-advice voice is enforced in one place.
- **Storage is opt-in per request** and in-memory in the MVP; swapping in SQLAlchemy/Postgres only touches the `_store` seam in `main.py`.

## 7. Implementation plan (as executed)

1. Pydantic schemas for the full domain (AnalysisResult and children)
2. Parser: PDF via PyMuPDF, OCR fallback, page-aware text with `locate()`
3. Redaction engine + tests (including Luhn false-positive control)
4. Prompt library with shared safety/citation blocks
5. Provider adapters (Anthropic / OpenAI / Ollama / none)
6. Heuristic engine: classifier, money/date/risk/field extraction, templated summaries
7. LLM engine + CitationMapper + Analyzer facade with graceful degradation
8. Report export (Markdown, JSON)
9. FastAPI endpoints incl. demo mode, opt-in store, compare, explain, redact
10. Single-file frontend
11. Test suite green (31 tests), example report generated from the sample lease

## 8. Data schema

Authoritative in `app/schemas.py`. Core: `AnalysisResult` composed of `Classification`, `ExtractedField[]`, `MoneyMention[]`, `DateMention[]`, `RiskItem[]`, `ActionItem[]`, `Question[]`, `WhatThisMeans`, plus `overall_confidence`, `engine`, `disclaimer`. Grounding via `Citation {quote, page, section, span}`. Interaction models: `ExplainRequest/Response`, `ComparisonResult/Change`, `RedactionResult`.

## 9. Prompt system

All eight prompts (classification, summary, extraction, risks, checklist, questions, passage explanation, comparison) are in `app/ai/prompts.py`, each with the safety register and citation rules injected.
