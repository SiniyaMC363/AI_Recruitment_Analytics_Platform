"""Text preprocessing utilities for recruitment documents.

Phase 2: Implement normalization, tokenization, and stopword handling.
"""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and lowercase raw text.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens.

    Phase 2 will replace this with spaCy-based tokenization.

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []
