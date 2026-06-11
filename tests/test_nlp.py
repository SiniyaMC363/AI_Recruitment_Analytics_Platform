"""Tests for NLP utilities."""

import pytest

from src.nlp.preprocessing import (
    normalize_text,
    preprocess_text,
    remove_stopwords,
    tokenize,
)
from src.nlp.resume_parser import extract_text_from_pdf
from src.nlp.skill_extractor import (
    extract_skills,
    match_skills,
    parse_skills_input,
)
from src.nlp.skills_database import PREDEFINED_SKILLS


# --- Preprocessing tests ---


def test_normalize_text_lowercases_and_removes_special_characters():
    text = "Hello!!! I know Python3 & SQL @ Company."
    result = normalize_text(text)
    assert result == "hello i know python3 sql company"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  Data   Science   ") == "data science"


def test_normalize_text_handles_empty_input():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_tokenize_returns_word_tokens():
    tokens = tokenize("Python developer with SQL experience")
    assert "Python" in tokens or "python" in [t.lower() for t in tokens]
    assert len(tokens) >= 3


def test_remove_stopwords_filters_common_words():
    tokens = ["the", "developer", "and", "python"]
    filtered = remove_stopwords(tokens)
    assert "developer" in filtered
    assert "python" in filtered
    assert "the" not in filtered
    assert "and" not in filtered


def test_preprocess_text_returns_cleaned_string():
    text = "The developer is skilled in Python and SQL!!!"
    result = preprocess_text(text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "python" in result
    assert "sql" in result


# --- Skill extraction tests ---


def test_extract_skills_uses_predefined_database_by_default():
    text = "Experienced in Python, SQL, and Machine Learning."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Machine Learning" in skills


def test_extract_skills_supports_custom_vocabulary():
    text = "Experienced in Python, SQL, and machine learning."
    skills = extract_skills(text, known_skills=["Python", "Java", "SQL"])
    assert skills == ["Python", "SQL"]


def test_extract_skills_handles_multi_word_phrases():
    text = "Worked on deep learning and cloud computing projects."
    skills = extract_skills(text)
    assert "Deep Learning" in skills
    assert "Cloud Computing" in skills


def test_extract_skills_returns_empty_for_blank_text():
    assert extract_skills("") == []
    assert extract_skills("   ") == []


def test_parse_skills_input_splits_comma_and_newline_values():
    raw = "Python, SQL\nMachine Learning; Git"
    parsed = parse_skills_input(raw)
    assert parsed == ["Python", "SQL", "Machine Learning", "Git"]


# --- Skill matching tests ---


def test_match_skills_calculates_matched_and_missing():
    resume = "Python developer with SQL, Git, and AWS experience."
    required = ["Python", "SQL", "Docker", "AWS"]
    result = match_skills(resume, required)

    assert result.skill_count == 4
    assert result.matched_skills == ["AWS", "Python", "SQL"]
    assert result.missing_skills == ["Docker"]
    assert result.match_percentage == 75.0


def test_match_skills_returns_zero_percent_when_no_requirements():
    resume = "Python and Java developer."
    result = match_skills(resume, [])
    assert result.match_percentage == 0.0
    assert result.matched_skills == []
    assert result.missing_skills == []


def test_match_skills_handles_no_overlap():
    resume = "Graphic designer skilled in Photoshop."
    required = ["Python", "SQL"]
    result = match_skills(resume, required)

    assert result.matched_skills == []
    assert result.missing_skills == ["Python", "SQL"]
    assert result.match_percentage == 0.0


def test_predefined_skills_database_contains_required_entries():
    expected = {
        "Python",
        "Java",
        "SQL",
        "Machine Learning",
        "Deep Learning",
        "Data Analysis",
        "Power BI",
        "Excel",
        "Tableau",
        "Git",
        "Docker",
        "AWS",
        "Azure",
        "Cloud Computing",
    }
    assert set(PREDEFINED_SKILLS) == expected


# --- PDF parsing tests ---


def test_extract_text_from_pdf_handles_empty_bytes():
    result = extract_text_from_pdf(b"")
    assert result.success is False
    assert result.text == ""
    assert result.error is not None


def test_extract_text_from_pdf_handles_invalid_pdf():
    result = extract_text_from_pdf(b"this is not a valid pdf file")
    assert result.success is False
    assert result.text == ""
    assert "Unable to read PDF file" in (result.error or "")
