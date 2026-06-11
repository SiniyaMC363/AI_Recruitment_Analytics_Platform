"""Text preprocessing utilities for recruitment documents."""

from __future__ import annotations

import re
from functools import lru_cache

import spacy

from config.settings import settings


def normalize_text(text: str) -> str:
    """Convert text to lowercase and remove special characters.

    Keeps letters, numbers, and spaces. Collapses repeated whitespace.

    Args:
        text: Raw input text.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    text = text.lower()
    # Keep alphanumeric characters and spaces only.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Backward-compatible alias for basic text normalization.

    Phase 1 code may still import clean_text; it now delegates to normalize_text.
    """
    return normalize_text(text)


@lru_cache(maxsize=1)
def _load_spacy_model() -> spacy.language.Language:
    """Load the configured spaCy model once and reuse it."""
    try:
        return spacy.load(settings.spacy_model)
    except OSError:
        # Fallback for environments where the model is not yet downloaded.
        return spacy.blank("en")


def tokenize(text: str) -> list[str]:
    """Tokenize text using spaCy.

    Args:
        text: Input text.

    Returns:
        List of token strings (no punctuation-only tokens).
    """
    if not text:
        return []

    nlp = _load_spacy_model()
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space and token.text.strip()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stopwords from a token list.

    Args:
        tokens: List of word tokens.

    Returns:
        Tokens with stopwords removed.
    """
    if not tokens:
        return []

    nlp = _load_spacy_model()
    stopwords = nlp.Defaults.stop_words
    return [token for token in tokens if token.lower() not in stopwords]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens using spaCy.

    Args:
        tokens: List of word tokens.

    Returns:
        Lemmatized token strings.
    """
    if not tokens:
        return []

    nlp = _load_spacy_model()
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.lemma_.strip()]


def preprocess_text(text: str) -> str:
    """Run the full NLP preprocessing pipeline.

    Steps:
        1. Normalize text (lowercase, remove special characters)
        2. Tokenize with spaCy
        3. Remove English stopwords
        4. Lemmatize tokens
        5. Return cleaned text suitable for analysis

    Args:
        text: Raw resume or job description text.

    Returns:
        Cleaned, space-joined text.
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""

    tokens = tokenize(normalized)
    tokens = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(tokens)
    return " ".join(lemmas).strip()
