"""Tests for NLP utilities."""

from src.nlp.preprocessing import clean_text, tokenize
from src.nlp.skill_extractor import extract_skills


def test_clean_text_normalizes_whitespace():
    assert clean_text("  Hello   World  ") == "hello world"


def test_tokenize_splits_on_whitespace():
    assert tokenize("Python  ML") == ["python", "ml"]


def test_extract_skills_matches_known_vocabulary():
    text = "Experienced in Python, SQL, and machine learning."
    skills = extract_skills(text, known_skills=["Python", "Java", "SQL"])
    assert skills == ["Python", "SQL"]
