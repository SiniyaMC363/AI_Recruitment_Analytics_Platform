"""Resume parsing from plain text and document files.

Phase 2: Add PDF/DOCX ingestion and structured field extraction.
"""

from pathlib import Path
from typing import Any


def parse_resume(source: str | Path) -> dict[str, Any]:
    """Parse a resume into structured fields.

    Args:
        source: Resume text or path to a resume file.

    Returns:
        Dictionary with parsed resume sections.

    Raises:
        NotImplementedError: Full parsing is planned for Phase 2.
    """
    if isinstance(source, Path) and source.suffix.lower() in {".pdf", ".docx"}:
        raise NotImplementedError(
            "Document resume parsing will be implemented in Phase 2."
        )

    text = str(source)
    return {
        "raw_text": text,
        "summary": None,
        "experience": [],
        "education": [],
        "skills": [],
    }
