"""Skill extraction from resumes and job descriptions.

Phase 2: Implement dictionary matching, NER, and embedding-based extraction.
"""

from typing import Iterable


def extract_skills(text: str, known_skills: Iterable[str] | None = None) -> list[str]:
    """Extract skills mentioned in the given text.

    Args:
        text: Resume or job description text.
        known_skills: Optional vocabulary of skills to match against.

    Returns:
        List of matched skill names.
    """
    if not text:
        return []

    if known_skills is None:
        return []

    normalized_text = text.lower()
    return sorted(
        skill for skill in known_skills if skill.lower() in normalized_text
    )
