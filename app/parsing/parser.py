"""Document text extraction.

Handles PDF (PyMuPDF), images (pytesseract OCR when installed), and plain
text. Returns page-aware text so downstream citations can point to pages.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field


@dataclass
class ParsedPage:
    number: int
    text: str


@dataclass
class ParsedDocument:
    filename: str
    pages: list[ParsedPage] = field(default_factory=list)
    source: str = "text"  # text | pdf | pdf_ocr | image_ocr | paste
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def locate(self, snippet: str) -> int | None:
        """Return the 1-based page number containing a snippet, if any."""
        needle = " ".join(snippet.split()).lower()
        for page in self.pages:
            if needle in " ".join(page.text.split()).lower():
                return page.number
        return None


class ParseError(ValueError):
    """The uploaded bytes could not be decoded as the claimed format."""


PDF_MAGIC = b"%PDF"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif"}
MIN_CHARS_BEFORE_OCR = 40  # per page; below this we assume a scan


def parse_bytes(data: bytes, filename: str) -> ParsedDocument:
    """Route raw upload bytes to the right extractor."""
    lower = filename.lower()
    if data[:4] == PDF_MAGIC or lower.endswith(".pdf"):
        return _parse_pdf(data, filename)
    if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return _parse_image(data, filename)
    return parse_text(data.decode("utf-8", errors="replace"), filename, source="text")


def parse_text(text: str, filename: str = "pasted-text", source: str = "paste") -> ParsedDocument:
    """Treat pasted or plain text as a single-page document."""
    return ParsedDocument(
        filename=filename,
        pages=[ParsedPage(number=1, text=text.strip())],
        source=source,
    )


def _parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    import fitz  # PyMuPDF

    doc = ParsedDocument(filename=filename, source="pdf")
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ParseError(
            "This file could not be read as a PDF. It may be corrupt or "
            "mislabeled; try re-exporting it or pasting the text instead."
        ) from exc
    with pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < MIN_CHARS_BEFORE_OCR:
                ocr_text = _ocr_pixmap(page)
                if ocr_text:
                    text = ocr_text
                    doc.source = "pdf_ocr"
                else:
                    doc.warnings.append(
                        f"Page {i} looks scanned and OCR is not available. "
                        "Install tesseract for scanned documents."
                    )
            doc.pages.append(ParsedPage(number=i, text=text))
    return doc


def _ocr_pixmap(page) -> str | None:
    """OCR a PyMuPDF page. Returns None if pytesseract is unavailable."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip() or None
    except Exception:
        return None


def _parse_image(data: bytes, filename: str) -> ParsedDocument:
    doc = ParsedDocument(filename=filename, source="image_ocr")
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        doc.pages.append(ParsedPage(number=1, text=""))
        doc.warnings.append(
            "Image uploads need OCR. Install tesseract and pytesseract, "
            "or paste the text instead."
        )
        return doc
    try:
        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img).strip()
    except Exception as exc:
        raise ParseError(
            "This file could not be read as an image. It may be corrupt "
            "or in an unsupported format; try a PNG or JPEG, or paste the text."
        ) from exc
    doc.pages.append(ParsedPage(number=1, text=text))
    if not text:
        doc.warnings.append("OCR found no readable text in this image.")
    return doc
