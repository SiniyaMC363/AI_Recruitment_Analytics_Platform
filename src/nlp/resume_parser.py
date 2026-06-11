"""Resume parsing from plain text and PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pdfplumber


@dataclass
class PDFExtractionResult:
    """Result of extracting text from a PDF resume."""

    text: str
    success: bool
    error: str | None = None
    page_count: int = 0


class ResumeParserError(Exception):
    """Raised when a resume cannot be parsed."""


def extract_text_from_pdf(source: Path | str | bytes | BinaryIO) -> PDFExtractionResult:
    """Extract text from a PDF resume using pdfplumber.

    Handles invalid, corrupted, and empty PDF files gracefully by returning
    a result object instead of raising exceptions to the UI layer.

    Args:
        source: File path, raw bytes, or file-like object containing PDF data.

    Returns:
        PDFExtractionResult with extracted text or an error message.
    """
    try:
        pdf_stream = _open_pdf_stream(source)
    except (TypeError, ValueError) as exc:
        return PDFExtractionResult(text="", success=False, error=str(exc))

    if pdf_stream is None:
        return PDFExtractionResult(
            text="",
            success=False,
            error="No PDF content was provided.",
        )

    try:
        with pdfplumber.open(pdf_stream) as pdf:
            if not pdf.pages:
                return PDFExtractionResult(
                    text="",
                    success=False,
                    error="The PDF file contains no pages.",
                    page_count=0,
                )

            page_texts: list[str] = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_texts.append(page_text.strip())

            combined_text = "\n".join(text for text in page_texts if text).strip()

            if not combined_text:
                return PDFExtractionResult(
                    text="",
                    success=False,
                    error="The PDF file is empty or contains no readable text.",
                    page_count=len(pdf.pages),
                )

            return PDFExtractionResult(
                text=combined_text,
                success=True,
                page_count=len(pdf.pages),
            )

    except Exception as exc:
        # pdfplumber raises various errors for corrupted or invalid PDFs.
        return PDFExtractionResult(
            text="",
            success=False,
            error=f"Unable to read PDF file: {exc}",
        )


def parse_resume(source: str | Path | bytes | BinaryIO) -> dict[str, Any]:
    """Parse a resume from text or a PDF file.

    Args:
        source: Plain resume text, path to a PDF, or PDF bytes/stream.

    Returns:
        Dictionary with raw text, extraction metadata, and placeholder sections.
    """
    if isinstance(source, Path):
        if source.suffix.lower() == ".pdf":
            result = extract_text_from_pdf(source)
            return _build_parse_result(result.text, result)
        text = source.read_text(encoding="utf-8", errors="ignore")
        return _build_parse_result(text)

    if isinstance(source, (bytes, BytesIO)) or hasattr(source, "read"):
        result = extract_text_from_pdf(source)
        return _build_parse_result(result.text, result)

    if isinstance(source, str) and source.lower().endswith(".pdf"):
        result = extract_text_from_pdf(Path(source))
        return _build_parse_result(result.text, result)

    return _build_parse_result(str(source))


def _open_pdf_stream(
    source: Path | str | bytes | BinaryIO,
) -> BytesIO | None:
    """Convert supported PDF inputs into a BytesIO stream for pdfplumber."""
    if isinstance(source, Path):
        if not source.exists():
            raise ValueError(f"PDF file not found: {source}")
        return BytesIO(source.read_bytes())

    if isinstance(source, str):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"PDF file not found: {source}")
        return BytesIO(path.read_bytes())

    if isinstance(source, bytes):
        if not source:
            return None
        return BytesIO(source)

    if hasattr(source, "read"):
        content = source.read()
        if not content:
            return None
        if isinstance(content, str):
            content = content.encode("utf-8")
        return BytesIO(content)

    raise TypeError("Unsupported PDF source type.")


def _build_parse_result(
    text: str,
    pdf_result: PDFExtractionResult | None = None,
) -> dict[str, Any]:
    """Build a consistent parse result dictionary."""
    return {
        "raw_text": text,
        "summary": None,
        "experience": [],
        "education": [],
        "skills": [],
        "pdf_extraction": {
            "success": pdf_result.success if pdf_result else True,
            "error": pdf_result.error if pdf_result else None,
            "page_count": pdf_result.page_count if pdf_result else 0,
        },
    }
