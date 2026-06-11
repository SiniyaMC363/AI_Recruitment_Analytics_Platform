"""Skill extraction and job-resume skill matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.nlp.skills_database import PREDEFINED_SKILLS, SKILL_LOOKUP


@dataclass
class SkillMatchResult:
    """Result of comparing resume skills against job requirements."""

    extracted_skills: list[str]
    required_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    skill_count: int
    match_percentage: float


def _normalize_skill_name(skill: str) -> str:
    """Normalize a skill name to its canonical form when possible."""
    cleaned = skill.strip()
    return SKILL_LOOKUP.get(cleaned.lower(), cleaned)


def _skill_in_text(skill: str, text: str) -> bool:
    """Check whether a skill appears in text using keyword matching.

    Multi-word skills use phrase matching. Single-word skills use word
    boundaries to reduce false positives (e.g. Java vs JavaScript).
    """
    skill_lower = skill.lower()
    text_lower = text.lower()

    if " " in skill_lower:
        return skill_lower in text_lower

    pattern = rf"\b{re.escape(skill_lower)}\b"
    return re.search(pattern, text_lower) is not None


def extract_skills(
    text: str,
    known_skills: Iterable[str] | None = None,
) -> list[str]:
    """Extract skills from resume or job description text.

    Uses keyword matching against the predefined skill database by default.

    Args:
        text: Resume or job description text.
        known_skills: Optional custom skill vocabulary. Defaults to
            PREDEFINED_SKILLS when not provided.

    Returns:
        Sorted list of matched canonical skill names.
    """
    if not text:
        return []

    skills = list(known_skills) if known_skills is not None else PREDEFINED_SKILLS
    matched = [_normalize_skill_name(skill) for skill in skills if _skill_in_text(skill, text)]

    # Preserve canonical order while removing duplicates.
    seen: set[str] = set()
    unique_skills: list[str] = []
    for skill in matched:
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            unique_skills.append(skill)

    return sorted(unique_skills, key=str.lower)


def parse_skills_input(skills_text: str) -> list[str]:
    """Parse comma- or newline-separated skill input from the UI.

    Args:
        skills_text: Raw user input listing required skills.

    Returns:
        Normalized list of skill names.
    """
    if not skills_text:
        return []

    raw_skills = re.split(r"[,;\n]+", skills_text)
    skills = [_normalize_skill_name(skill) for skill in raw_skills if skill.strip()]
    return skills


def match_skills(
    resume_text: str,
    required_skills: Iterable[str],
    *,
    known_skills: Iterable[str] | None = None,
) -> SkillMatchResult:
    """Compare extracted resume skills with required job skills.

    Args:
        resume_text: Raw or cleaned resume text.
        required_skills: Skills required by the job description.
        known_skills: Optional custom skill vocabulary for extraction.

    Returns:
        SkillMatchResult with matched, missing, and percentage metrics.
    """
    extracted = extract_skills(resume_text, known_skills=known_skills)
    required = [_normalize_skill_name(skill) for skill in required_skills if str(skill).strip()]

    # Remove duplicate required skills while preserving order.
    seen_required: set[str] = set()
    unique_required: list[str] = []
    for skill in required:
        key = skill.lower()
        if key not in seen_required:
            seen_required.add(key)
            unique_required.append(skill)

    extracted_lookup = {skill.lower(): skill for skill in extracted}
    matched: list[str] = []
    missing: list[str] = []

    for skill in unique_required:
        if skill.lower() in extracted_lookup:
            matched.append(extracted_lookup[skill.lower()])
        else:
            missing.append(skill)

    match_percentage = (
        round((len(matched) / len(unique_required)) * 100, 1)
        if unique_required
        else 0.0
    )

    return SkillMatchResult(
        extracted_skills=extracted,
        required_skills=unique_required,
        matched_skills=sorted(matched, key=str.lower),
        missing_skills=sorted(missing, key=str.lower),
        skill_count=len(extracted),
        match_percentage=match_percentage,
    )
