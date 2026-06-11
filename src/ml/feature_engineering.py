"""Feature engineering for candidate suitability ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# Columns used by the Random Forest model (order matters for inference).
FEATURE_COLUMNS: list[str] = [
    "skill_match_score",
    "matched_skills_count",
    "skills_count",
    "experience_years",
    "education_score",
]

TARGET_COLUMN = "candidate_rating"

# Suitability labels used for classification.
SUITABILITY_LABELS: list[str] = [
    "Highly Suitable",
    "Suitable",
    "Less Suitable",
]

# Map education levels to numeric scores between 0 and 1.
EDUCATION_SCORES: dict[str, float] = {
    "phd": 1.0,
    "doctorate": 1.0,
    "master": 0.85,
    "masters": 0.85,
    "bachelor": 0.65,
    "bachelors": 0.65,
    "associate": 0.45,
    "diploma": 0.4,
    "high school": 0.25,
    "other": 0.5,
}


@dataclass
class CandidateFeatures:
    """Structured candidate feature set for ML inference."""

    skill_match_score: float
    matched_skills_count: int
    skills_count: int
    experience_years: float
    education_score: float

    def to_dict(self) -> dict[str, float | int]:
        """Convert features to a dictionary."""
        return {
            "skill_match_score": self.skill_match_score,
            "matched_skills_count": self.matched_skills_count,
            "skills_count": self.skills_count,
            "experience_years": self.experience_years,
            "education_score": self.education_score,
        }


def education_level_to_score(education_level: str | float | int) -> float:
    """Convert an education level label into a numeric score.

    Args:
        education_level: Education label (e.g. 'Bachelor') or numeric score.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if isinstance(education_level, (int, float)):
        return float(max(0.0, min(1.0, education_level)))

    if not education_level:
        return EDUCATION_SCORES["other"]

    normalized = str(education_level).strip().lower()
    return EDUCATION_SCORES.get(normalized, EDUCATION_SCORES["other"])


def build_candidate_features(
    skill_match_percentage: float,
    matched_skills_count: int,
    skills_count: int,
    experience_years: float,
    education_level: str | float = "Bachelor",
) -> CandidateFeatures:
    """Build validated candidate features from raw recruiter inputs.

    Args:
        skill_match_percentage: Skill overlap percentage (0-100).
        matched_skills_count: Number of required skills found in resume.
        skills_count: Total skills detected for the candidate.
        experience_years: Years of professional experience.
        education_level: Education label or numeric score.

    Returns:
        CandidateFeatures ready for model inference.
    """
    skill_match = _clamp(skill_match_percentage, 0.0, 100.0)
    matched = max(0, int(matched_skills_count))
    total_skills = max(0, int(skills_count))

    # Ensure matched skills never exceed total skills.
    if matched > total_skills:
        total_skills = matched

    experience = max(0.0, float(experience_years))
    education_score = education_level_to_score(education_level)

    return CandidateFeatures(
        skill_match_score=round(skill_match, 1),
        matched_skills_count=matched,
        skills_count=total_skills,
        experience_years=round(experience, 1),
        education_score=round(education_score, 2),
    )


def build_features_from_skill_match(
    match_percentage: float,
    matched_skills: list[str],
    extracted_skills: list[str],
    experience_years: float = 2.0,
    education_level: str = "Bachelor",
) -> CandidateFeatures:
    """Build features using outputs from the Phase 2 NLP skill matcher.

    Args:
        match_percentage: Skill match percentage from NLP module.
        matched_skills: Skills matched against job requirements.
        extracted_skills: All skills extracted from the resume.
        experience_years: Estimated years of experience.
        education_level: Candidate education level.

    Returns:
        CandidateFeatures for ML ranking.
    """
    return build_candidate_features(
        skill_match_percentage=match_percentage,
        matched_skills_count=len(matched_skills),
        skills_count=len(extracted_skills),
        experience_years=experience_years,
        education_level=education_level,
    )


def features_to_dataframe(
    features: CandidateFeatures | dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert feature dictionaries into a model-ready DataFrame.

    Args:
        features: Single feature set or list of feature dictionaries.

    Returns:
        DataFrame containing only the model feature columns.
    """
    if isinstance(features, CandidateFeatures):
        records = [features.to_dict()]
    elif isinstance(features, dict):
        records = [features]
    else:
        records = list(features)

    df = pd.DataFrame(records)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return df[FEATURE_COLUMNS].copy()


def build_feature_matrix(
    candidates: pd.DataFrame,
    job_postings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper for Phase 1 imports.

    Expects candidate rows to already contain engineered feature columns.
    """
    _ = job_postings
    return features_to_dataframe(candidates.to_dict(orient="records"))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Restrict a numeric value to a given range."""
    return max(minimum, min(maximum, float(value)))
