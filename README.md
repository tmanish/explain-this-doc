# Explain This Document Like I'm Human

**A personal translator for bureaucracy. Give it the paperwork; it tells you what matters, what it means, and what to look at carefully.**

Leases, insurance policies, medical bills, bank notices, employment agreements, government letters. Documents written by professionals, for professionals, handed to people who just need to know: what am I agreeing to, what do I owe, when is it due, and what should I ask before I sign?

This app answers those questions in plain English, with every claim grounded in a quote from the document itself.

## What it does

Upload a PDF, image, scan, or pasted text and get:

1. **Document type detection.** "This appears to be a lease agreement." When it isn't sure, it says so.
2. **A plain-English summary** and a "what this means for you" section.
3. **Key details** extracted per document type: rent and deposit for a lease, patient responsibility for a medical bill, APR changes for a card notice, non-compete scope for an employment agreement.
4. **Money mentioned**, with recurring amounts flagged.
5. **Dates and deadlines**, with likely deadlines flagged.
6. **Things to check carefully**: automatic renewal, arbitration clauses, waivers of rights, early termination fees, non-refundable payments, broad liability language, and more. Each one comes with why it matters and a specific question to ask.
7. **An action checklist** with priorities and due dates.
8. **Questions you should ask before you sign or respond**, specific to the document.
9. **Select-to-explain**: highlight any confusing passage and ask "explain this," "is this normal?", "what are the risks?", or "rewrite in simple English."
10. **Document comparison**: old lease vs new lease, old card terms vs new terms.
11. **Confidence scores and citations** on every extracted item. When a claim cannot be grounded in the document text, it is marked "Source reference unavailable" rather than invented.

## What it is not

This is not legal, medical, financial, or professional advice, and it is not a replacement for a qualified professional. It explains, organizes, and helps you ask better questions. It will never tell you to sign or not sign, never diagnose, and never claim to know your local law. Every report carries this disclaimer.

## Why it matters

The documents that shape people's lives are the ones they understand least. The information asymmetry is the product: late fees, auto-renewals, and waived rights survive because nobody reads page four. This tool reads page four.

## Architecture

```
Upload (PDF / image / text)
        |
   DocumentParser  ............ PyMuPDF + optional tesseract OCR
        |
   Redaction (optional) ....... masks SSNs, cards, phones, emails, IDs, addresses
        |
   Analyzer facade
     |-- LLM engine ........... classify -> summarize -> extract -> risks
     |     (Anthropic / OpenAI     -> checklist -> questions, all JSON-schema prompts
     |      / Ollama adapter)
     |-- CitationMapper ....... verifies every model quote against the real text;
     |                          unverifiable quotes are stripped, never trusted
     |-- Heuristic engine ..... fully offline: keyword classifier, regex money/date
                                extraction, rule-based risk detection
        |
   AnalysisResult (Pydantic) -> Web UI / Markdown report / JSON export
```

Two engines, one contract. If no LLM provider is configured (the default), everything runs locally through the heuristic engine and no text ever leaves the machine. If a provider is configured and a call fails, the app degrades to local analysis instead of breaking.

## Tech stack

- Python 3.10+ (3.12 recommended), FastAPI, Pydantic v2
- PyMuPDF for PDF extraction, pytesseract (optional) for scans and images
- Provider adapters: Anthropic, OpenAI, OpenRouter, Ollama, or none (local mode)
- Single-file vanilla JS frontend, dark mode, no external dependencies
- pytest, 34 tests

## Screenshots

*(placeholder: intake screen, lease analysis, select-to-explain popover)*

## Setup

Requires Python 3.10 or newer (3.12 recommended).

```bash
git clone https://github.com/tmanish/explain-this-doc
cd explain-this-doc
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000. Local mode works immediately; try the five built-in sample documents.

**Enable an LLM provider (optional):**

Pick an engine on the landing page — Local, Anthropic, OpenAI, OpenRouter, or Ollama — and
paste an API key there (held in server memory only, never stored or logged).
Or configure it via environment variables before starting:

```bash
export ETD_PROVIDER=anthropic   # or openai, openrouter, ollama
export ANTHROPIC_API_KEY=sk-...
# optional: export ETD_MODEL=claude-sonnet-5
```

**OCR for scanned documents (optional):**

```bash
sudo apt install tesseract-ocr        # or: brew install tesseract
pip install pytesseract Pillow
```

**Docker:**

```bash
docker build -t explain-this-doc .
docker run -p 8000:8000 explain-this-doc
```

## Usage

Web UI: drop a file, paste text, or pick a sample. Toggle "Mask sensitive details" to redact SSNs, account numbers, phones, and addresses before analysis. Select any passage in the document pane to explain it. Export Markdown or JSON from the results pane.

API:

```bash
# Analyze pasted text
curl -X POST localhost:8000/api/analyze/text \
  -H 'Content-Type: application/json' \
  -d '{"text": "...", "redact_first": true}'

# Analyze a file
curl -X POST localhost:8000/api/analyze/upload -F file=@lease.pdf

# Explain a passage
curl -X POST localhost:8000/api/explain \
  -H 'Content-Type: application/json' \
  -d '{"passage": "binding arbitration...", "mode": "risks"}'

# Compare two versions
curl -X POST localhost:8000/api/compare -F old_file=@old.pdf -F new_file=@new.pdf

# Built-in synthetic samples
curl -X POST localhost:8000/api/demo/lease
```

Interactive API docs at `/docs`. Example output for the sample lease is in [`examples/lease.report.md`](examples/lease.report.md).

## Privacy

- Documents are processed in memory and are **not stored** unless you opt in per request (`store=true`). Stored documents can be deleted with one call.
- `ETD_PROVIDER=none` (the default) keeps every byte on your machine.
- Redaction mode masks SSNs, credit card numbers (Luhn-validated), phone numbers, emails, street addresses, and policy/claim/account/patient IDs before analysis or export.
- The server is single-user and has no authentication: anyone who can reach the port can use it, including reading documents stored with `store=true`. Run it on localhost (the default) or behind your own auth; don't expose it directly to a network.

## Safety disclaimer

This software explains documents for understanding only. It is not legal, medical, financial, or professional advice. For decisions that matter, consult a lawyer, doctor, accountant, financial advisor, insurance agent, or other qualified professional. The five sample documents in `demo/` are synthetic and describe no real people, companies, or accounts.

## Roadmap

- PDF report export with highlighted source regions
- Side-by-side diff view for comparison mode
- Per-page citation anchors in the document viewer (click a risk, scroll to the clause)
- Postgres persistence behind the existing opt-in store
- Multi-language plain-English output
- Browser extension: explain terms-of-service pages in place

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers parsing, redaction (including Luhn validation and false-positive control), classification across all five sample documents, risk detection, citation grounding (fabricated quotes are rejected), report structure, comparison, and every API endpoint.
