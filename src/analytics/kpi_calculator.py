"""KPI calculation utilities for recruitment analytics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ml.feature_engineering import SUITABILITY_LABELS

# Expected column names in the analytics dataset.
COL_SKILL_MATCH = "skill_match_score"
COL_EXPERIENCE = "experience_years"
COL_TOTAL_SKILLS = "total_skills"
COL_CATEGORY = "predicted_suitability_category"


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, coercing invalid values to NaN."""
    if column not in df.columns or df.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def calculate_total_candidates(analytics_df: pd.DataFrame) -> int:
    """Count the number of candidates in the analytics dataset."""
    if analytics_df is None or analytics_df.empty:
        return 0
    return int(len(analytics_df))


def calculate_category_counts(analytics_df: pd.DataFrame) -> dict[str, int]:
    """Count candidates in each suitability category."""
    counts = {label: 0 for label in SUITABILITY_LABELS}
    if analytics_df is None or analytics_df.empty or COL_CATEGORY not in analytics_df.columns:
        return counts

    value_counts = analytics_df[COL_CATEGORY].value_counts()
    for label in SUITABILITY_LABELS:
        counts[label] = int(value_counts.get(label, 0))
    return counts


def calculate_category_percentages(analytics_df: pd.DataFrame) -> dict[str, float]:
    """Calculate the percentage of candidates in each suitability category."""
    total = calculate_total_candidates(analytics_df)
    counts = calculate_category_counts(analytics_df)
    if total == 0:
        return {label: 0.0 for label in SUITABILITY_LABELS}

    return {label: round((count / total) * 100, 1) for label, count in counts.items()}


def calculate_average_skill_match(analytics_df: pd.DataFrame) -> float:
    """Calculate the average skill match score across candidates."""
    scores = _safe_numeric_series(analytics_df, COL_SKILL_MATCH)
    if scores.empty or scores.dropna().empty:
        return 0.0
    return round(float(scores.mean()), 1)


def calculate_average_experience(analytics_df: pd.DataFrame) -> float:
    """Calculate the average years of experience across candidates."""
    experience = _safe_numeric_series(analytics_df, COL_EXPERIENCE)
    if experience.empty or experience.dropna().empty:
        return 0.0
    return round(float(experience.mean()), 1)


def calculate_average_skills(analytics_df: pd.DataFrame) -> float:
    """Calculate the average number of skills detected per candidate."""
    skills = _safe_numeric_series(analytics_df, COL_TOTAL_SKILLS)
    if skills.empty or skills.dropna().empty:
        return 0.0
    return round(float(skills.mean()), 1)


def calculate_all_kpis(analytics_df: pd.DataFrame) -> dict[str, Any]:
    """Compute all recruitment analytics KPIs in one call.

    Returns:
        Dictionary with totals, category breakdowns, and averages.
    """
    category_counts = calculate_category_counts(analytics_df)
    category_percentages = calculate_category_percentages(analytics_df)

    return {
        "total_candidates": calculate_total_candidates(analytics_df),
        "highly_suitable_count": category_counts.get("Highly Suitable", 0),
        "suitable_count": category_counts.get("Suitable", 0),
        "less_suitable_count": category_counts.get("Less Suitable", 0),
        "highly_suitable_percentage": category_percentages.get("Highly Suitable", 0.0),
        "suitable_percentage": category_percentages.get("Suitable", 0.0),
        "less_suitable_percentage": category_percentages.get("Less Suitable", 0.0),
        "average_skill_match_score": calculate_average_skill_match(analytics_df),
        "average_experience_years": calculate_average_experience(analytics_df),
        "average_skills_count": calculate_average_skills(analytics_df),
    }
