"""Tests for resume PDF parsing."""

from src.nlp.resume_parser import parse_resume


def test_parse_resume_from_plain_text():
    text = "Alice Johnson\nPython developer with SQL skills."
    result = parse_resume(text)

    assert result["raw_text"] == text
    assert result["pdf_extraction"]["success"] is True
    assert result["pdf_extraction"]["error"] is None


def test_parse_resume_from_text_dict_structure():
    result = parse_resume("Sample resume content")

    assert "raw_text" in result
    assert "skills" in result
    assert "experience" in result
    assert "education" in result
    assert "pdf_extraction" in result
