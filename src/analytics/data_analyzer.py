"""Data analysis and analytics dataset preparation for recruitment insights."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import settings
from src.ml.feature_engineering import SUITABILITY_LABELS, build_features_from_skill_match
from src.ml.predictor import model_is_available, predict_candidate
from src.nlp.skill_extractor import extract_skills, match_skills

# Processed analytics dataset filename and column schema.
ANALYTICS_DATASET_FILENAME = "candidate_analytics.csv"

ANALYTICS_COLUMNS: list[str] = [
    "candidate_id",
    "candidate_name",
    "skill_match_score",
    "experience_years",
    "total_skills",
    "education_level",
    "predicted_suitability_category",
]

# Simple keyword map for education detection from resume text.
EDUCATION_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("PhD", ("phd", "doctorate", "doctoral")),
    ("Master", ("master", "masters", "mba", "m.s.", "msc")),
    ("Bachelor", ("bachelor", "bachelors", "b.s.", "bsc", "undergraduate")),
    ("Associate", ("associate",)),
    ("High School", ("high school", "diploma")),
]

DEFAULT_REQUIRED_SKILLS: list[str] = [
    "Python",
    "SQL",
    "Machine Learning",
    "Git",
    "AWS",
]


def get_analytics_dataset_path(output_dir: Path | None = None) -> Path:
    """Return the default path for the processed analytics dataset."""
    directory = output_dir or settings.data_processed_dir
    return directory / ANALYTICS_DATASET_FILENAME


def extract_experience_years(resume_text: str) -> float:
    """Estimate years of experience from resume text using simple patterns.

    Args:
        resume_text: Raw resume content.

    Returns:
        Estimated years of experience, or 0.0 when not found.
    """
    if not resume_text:
        return 0.0

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+in\s+",
    ]

    for pattern in patterns:
        match = re.search(pattern, resume_text, flags=re.IGNORECASE)
        if match:
            return round(float(match.group(1)), 1)

    return 0.0


def extract_education_level(resume_text: str) -> str:
    """Detect the highest education level mentioned in resume text.

    Args:
        resume_text: Raw resume content.

    Returns:
        Education level label (defaults to 'Other' when unknown).
    """
    if not resume_text:
        return "Other"

    text_lower = resume_text.lower()
    for level, keywords in EDUCATION_KEYWORDS:
        if any(keyword in text_lower for keyword in keywords):
            return level

    return "Other"


def _rule_based_suitability(features) -> str:
    """Fallback suitability label when the ML model is unavailable."""
    score = (
        features.skill_match_score * 0.45
        + features.matched_skills_count * 6.0
        + min(features.skills_count, 10) * 1.5
        + features.experience_years * 2.0
        + features.education_score * 20.0
    )
    if score >= 105:
        return "Highly Suitable"
    if score >= 70:
        return "Suitable"
    return "Less Suitable"


def _predict_suitability(
    candidate_name: str,
    features,
    model_path: Path | None = None,
) -> str:
    """Predict suitability using ML when available, otherwise use rules."""
    if model_is_available(model_path):
        try:
            prediction = predict_candidate(
                candidate_name=candidate_name,
                features=features,
                model_path=model_path,
            )
            return prediction.predicted_category
        except FileNotFoundError:
            pass

    return _rule_based_suitability(features)


def build_analytics_dataset(
    candidates: pd.DataFrame,
    job_postings: pd.DataFrame | None = None,
    required_skills: Iterable[str] | None = None,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Build a processed analytics dataset from candidates and ML predictions.

    For each candidate, NLP skill matching and resume heuristics produce
    features that are passed to the trained ranker (or a rule-based fallback).

    Args:
        candidates: Raw candidate records with resume text.
        job_postings: Optional job postings used to infer required skills.
        required_skills: Optional explicit skill list for matching.
        model_path: Optional custom ML model path.

    Returns:
        DataFrame with standardized analytics columns.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=ANALYTICS_COLUMNS)

    skills = _resolve_required_skills(job_postings, required_skills)
    records: list[dict[str, object]] = []

    for _, row in candidates.iterrows():
        resume_text = str(row.get("resume_text", "") or "")
        candidate_id = row.get("candidate_id", "")
        candidate_name = str(row.get("name", "Unknown Candidate"))

        match_result = match_skills(resume_text, skills)
        experience_years = extract_experience_years(resume_text)
        education_level = extract_education_level(resume_text)

        features = build_features_from_skill_match(
            match_percentage=match_result.match_percentage,
            matched_skills=match_result.matched_skills,
            extracted_skills=match_result.extracted_skills,
            experience_years=experience_years,
            education_level=education_level,
        )

        predicted_category = _predict_suitability(
            candidate_name=candidate_name,
            features=features,
            model_path=model_path,
        )

        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "skill_match_score": features.skill_match_score,
                "experience_years": features.experience_years,
                "total_skills": features.skills_count,
                "education_level": education_level,
                "predicted_suitability_category": predicted_category,
            }
        )

    return pd.DataFrame(records, columns=ANALYTICS_COLUMNS)


def save_analytics_dataset(
    analytics_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Persist the analytics dataset to CSV."""
    path = output_path or get_analytics_dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    analytics_df.to_csv(path, index=False)
    return path


def load_analytics_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load a previously saved analytics dataset from disk."""
    path = dataset_path or get_analytics_dataset_path()
    if not path.exists():
        return pd.DataFrame(columns=ANALYTICS_COLUMNS)

    df = pd.read_csv(path)
    for column in ANALYTICS_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df[ANALYTICS_COLUMNS].copy()


def filter_analytics_data(
    analytics_df: pd.DataFrame,
    categories: list[str] | None = None,
    min_experience: float = 0.0,
    min_skill_match: float = 0.0,
) -> pd.DataFrame:
    """Filter analytics data by category, experience, and skill match score."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(columns=ANALYTICS_COLUMNS)

    filtered = analytics_df.copy()

    if categories:
        filtered = filtered[
            filtered["predicted_suitability_category"].isin(categories)
        ]

    filtered["experience_years"] = pd.to_numeric(
        filtered["experience_years"], errors="coerce"
    ).fillna(0.0)
    filtered["skill_match_score"] = pd.to_numeric(
        filtered["skill_match_score"], errors="coerce"
    ).fillna(0.0)

    filtered = filtered[filtered["experience_years"] >= float(min_experience)]
    filtered = filtered[filtered["skill_match_score"] >= float(min_skill_match)]

    return filtered.reset_index(drop=True)


def get_category_distribution(analytics_df: pd.DataFrame) -> pd.DataFrame:
    """Return candidate counts grouped by suitability category."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame({"category": SUITABILITY_LABELS, "count": [0, 0, 0]})

    counts = (
        analytics_df["predicted_suitability_category"]
        .value_counts()
        .reindex(SUITABILITY_LABELS, fill_value=0)
        .reset_index()
    )
    counts.columns = ["category", "count"]
    return counts


def get_score_distribution(analytics_df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Return a histogram-style distribution of skill match scores."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(columns=["score_range", "count"])

    scores = pd.to_numeric(analytics_df["skill_match_score"], errors="coerce").dropna()
    if scores.empty:
        return pd.DataFrame(columns=["score_range", "count"])

    bucketed = pd.cut(scores, bins=bins, include_lowest=True)
    distribution = bucketed.value_counts(sort=False).reset_index()
    distribution.columns = ["score_range", "count"]
    distribution["score_range"] = distribution["score_range"].astype(str)
    return distribution


def get_experience_distribution(
    analytics_df: pd.DataFrame,
    bins: int = 8,
) -> pd.DataFrame:
    """Return a histogram-style distribution of experience years."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(columns=["experience_range", "count"])

    experience = pd.to_numeric(analytics_df["experience_years"], errors="coerce").dropna()
    if experience.empty:
        return pd.DataFrame(columns=["experience_range", "count"])

    bucketed = pd.cut(experience, bins=bins, include_lowest=True)
    distribution = bucketed.value_counts(sort=False).reset_index()
    distribution.columns = ["experience_range", "count"]
    distribution["experience_range"] = distribution["experience_range"].astype(str)
    return distribution


def get_education_comparison(analytics_df: pd.DataFrame) -> pd.DataFrame:
    """Compare average skill match scores across education levels."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(columns=["education_level", "candidate_count", "avg_match_score"])

    working = analytics_df.copy()
    working["skill_match_score"] = pd.to_numeric(
        working["skill_match_score"], errors="coerce"
    )

    grouped = (
        working.groupby("education_level", dropna=False)
        .agg(
            candidate_count=("candidate_id", "count"),
            avg_match_score=("skill_match_score", "mean"),
        )
        .reset_index()
    )
    grouped["avg_match_score"] = grouped["avg_match_score"].round(1)
    return grouped.sort_values("avg_match_score", ascending=False)


def get_top_candidates(
    analytics_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-ranked candidates sorted by skill match score."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(columns=ANALYTICS_COLUMNS)

    working = analytics_df.copy()
    working["skill_match_score"] = pd.to_numeric(
        working["skill_match_score"], errors="coerce"
    ).fillna(0.0)

    return (
        working.sort_values(
            by=["skill_match_score", "experience_years"],
            ascending=[False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def _resolve_required_skills(
    job_postings: pd.DataFrame | None,
    required_skills: Iterable[str] | None,
) -> list[str]:
    """Determine required skills from explicit input or job descriptions."""
    if required_skills:
        return list(required_skills)

    if job_postings is not None and not job_postings.empty:
        # Use the first job posting description as the default benchmark role.
        description = str(job_postings.iloc[0].get("description", "") or "")
        detected = extract_skills(description)
        if detected:
            return detected

    return DEFAULT_REQUIRED_SKILLS.copy()
