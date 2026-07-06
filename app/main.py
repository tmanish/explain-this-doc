"""Explain This Document Like I'm Human - FastAPI backend.

Privacy stance: documents are processed in memory and are NOT stored
unless the client opts in (store=true). Stored documents live in an
in-process store and can be deleted at any time via DELETE /documents/{id}.
Set ETD_PROVIDER=none (default) for fully local processing.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ai.pipeline import Analyzer
from app.export.reports import to_markdown
from app.parsing.parser import ParsedDocument, parse_bytes, parse_text
from app.privacy.redaction import redact
from app.schemas import (
    AnalysisResult,
    ComparisonResult,
    DocumentType,
    ExplainRequest,
    ExplainResponse,
    RedactionResult,
)

app = FastAPI(
    title="Explain This Document Like I'm Human",
    description="Turns confusing paperwork into a clear, human-readable breakdown. "
    "Not legal, medical, financial, or professional advice.",
    version="0.1.0",
)

analyzer = Analyzer()

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Opt-in in-memory store: {id: (ParsedDocument, AnalysisResult)}
_store: dict[str, tuple[ParsedDocument, AnalysisResult]] = {}


class TextIn(BaseModel):
    text: str
    filename: str = "pasted-text"
    redact_first: bool = False
    store: bool = False


class AnalyzeOut(BaseModel):
    result: AnalysisResult
    document_text: str = ""
    document_id: str | None = None
    warnings: list[str] = []
    stored: bool = False


def _run(doc: ParsedDocument, redact_first: bool, store: bool) -> AnalyzeOut:
    if not doc.full_text.strip():
        raise HTTPException(422, "No readable text was found in this document.")
    if redact_first:
        redacted = redact(doc.full_text)
        doc = parse_text(redacted.redacted_text, filename=doc.filename, source=doc.source)
    result = analyzer.analyze(doc)
    doc_id = None
    if store:
        doc_id = uuid.uuid4().hex[:12]
        _store[doc_id] = (doc, result)
    return AnalyzeOut(result=result, document_text=doc.full_text, document_id=doc_id, warnings=doc.warnings, stored=store)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine": analyzer.mode}


@app.post("/api/analyze/upload", response_model=AnalyzeOut)
async def analyze_upload(
    file: UploadFile = File(...),
    redact_first: bool = Form(False),
    store: bool = Form(False),
) -> AnalyzeOut:
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 20 MB.")
    doc = parse_bytes(data, file.filename or "upload")
    return _run(doc, redact_first, store)


@app.post("/api/analyze/text", response_model=AnalyzeOut)
def analyze_text(body: TextIn) -> AnalyzeOut:
    doc = parse_text(body.text, filename=body.filename)
    return _run(doc, body.redact_first, body.store)


@app.post("/api/explain", response_model=ExplainResponse)
def explain_passage(body: ExplainRequest) -> ExplainResponse:
    if not body.passage.strip():
        raise HTTPException(422, "Select some text to explain.")
    return analyzer.explain(body.passage, body.mode, body.document_type)


@app.post("/api/compare", response_model=ComparisonResult)
async def compare(
    old_file: UploadFile = File(...), new_file: UploadFile = File(...)
) -> ComparisonResult:
    old_doc = parse_bytes(await old_file.read(), old_file.filename or "old")
    new_doc = parse_bytes(await new_file.read(), new_file.filename or "new")
    if not old_doc.full_text.strip() or not new_doc.full_text.strip():
        raise HTTPException(422, "One of the documents has no readable text.")
    return analyzer.compare(old_doc, new_doc)


@app.post("/api/redact", response_model=RedactionResult)
def redact_text(body: TextIn) -> RedactionResult:
    return redact(body.text)


@app.get("/api/demo")
def list_demos() -> list[dict]:
    return [
        {"id": p.stem, "name": p.stem.replace("_", " ").title()}
        for p in sorted(DEMO_DIR.glob("*.txt"))
    ]


@app.post("/api/demo/{demo_id}", response_model=AnalyzeOut)
def analyze_demo(demo_id: str) -> AnalyzeOut:
    if not re.fullmatch(r"[a-z0-9_]+", demo_id):
        raise HTTPException(404, "Demo document not found.")
    path = (DEMO_DIR / f"{demo_id}.txt").resolve()
    if not path.is_file() or path.parent != DEMO_DIR.resolve():
        raise HTTPException(404, "Demo document not found.")
    doc = parse_text(path.read_text(), filename=f"{demo_id}.txt", source="text")
    return _run(doc, redact_first=False, store=False)


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return [
        {
            "id": doc_id,
            "name": doc.filename,
            "type": result.classification.document_type.value,
        }
        for doc_id, (doc, result) in _store.items()
    ]


@app.get("/api/documents/{doc_id}", response_model=AnalyzeOut)
def get_document(doc_id: str) -> AnalyzeOut:
    if doc_id not in _store:
        raise HTTPException(404, "Document not found or already deleted.")
    doc, result = _store[doc_id]
    return AnalyzeOut(result=result, document_text=doc.full_text, document_id=doc_id, warnings=doc.warnings, stored=True)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if _store.pop(doc_id, None) is None:
        raise HTTPException(404, "Document not found or already deleted.")
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/report.md", response_class=PlainTextResponse)
def document_report(doc_id: str) -> str:
    if doc_id not in _store:
        raise HTTPException(404, "Document not found or already deleted.")
    _, result = _store[doc_id]
    return to_markdown(result)


class ReportIn(BaseModel):
    result: AnalysisResult


@app.post("/api/report/markdown", response_class=PlainTextResponse)
def report_markdown(body: ReportIn) -> str:
    """Render a Markdown report from an analysis result the client holds,
    so exports work even when nothing was stored server-side."""
    return to_markdown(body.result)


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = STATIC_DIR / "favicon.ico"
    if path.is_file():
        return FileResponse(path)
    raise HTTPException(404)
