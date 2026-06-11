"""Tests for KPI calculation utilities."""

import pandas as pd
import pytest

from src.analytics.kpi_calculator import (
    calculate_all_kpis,
    calculate_average_experience,
    calculate_average_skill_match,
    calculate_average_skills,
    calculate_category_counts,
    calculate_category_percentages,
    calculate_total_candidates,
)


@pytest.fixture
def sample_analytics_df() -> pd.DataFrame:
    """Sample analytics dataset for KPI tests."""
    return pd.DataFrame(
        {
            "candidate_id": ["1", "2", "3", "4"],
            "candidate_name": ["Alice", "Bob", "Carol", "Dan"],
            "skill_match_score": [80.0, 60.0, 40.0, 90.0],
            "experience_years": [5.0, 3.0, 1.0, 7.0],
            "total_skills": [8, 6, 4, 10],
            "education_level": ["Master", "Bachelor", "Bachelor", "PhD"],
            "predicted_suitability_category": [
                "Highly Suitable",
                "Suitable",
                "Less Suitable",
                "Highly Suitable",
            ],
        }
    )


def test_calculate_total_candidates(sample_analytics_df):
    assert calculate_total_candidates(sample_analytics_df) == 4
    assert calculate_total_candidates(pd.DataFrame()) == 0
    assert calculate_total_candidates(None) == 0


def test_calculate_category_counts(sample_analytics_df):
    counts = calculate_category_counts(sample_analytics_df)

    assert counts["Highly Suitable"] == 2
    assert counts["Suitable"] == 1
    assert counts["Less Suitable"] == 1


def test_calculate_category_percentages(sample_analytics_df):
    percentages = calculate_category_percentages(sample_analytics_df)

    assert percentages["Highly Suitable"] == 50.0
    assert percentages["Suitable"] == 25.0
    assert percentages["Less Suitable"] == 25.0


def test_calculate_average_metrics(sample_analytics_df):
    assert calculate_average_skill_match(sample_analytics_df) == 67.5
    assert calculate_average_experience(sample_analytics_df) == 4.0
    assert calculate_average_skills(sample_analytics_df) == 7.0


def test_calculate_average_metrics_handle_missing_data():
    df = pd.DataFrame(
        {
            "skill_match_score": [None, "invalid", 50.0],
            "experience_years": [None, 2.0, 4.0],
            "total_skills": [None, 3, 5],
        }
    )

    assert calculate_average_skill_match(df) == 50.0
    assert calculate_average_experience(df) == 3.0
    assert calculate_average_skills(df) == 4.0


def test_calculate_all_kpis(sample_analytics_df):
    kpis = calculate_all_kpis(sample_analytics_df)

    assert kpis["total_candidates"] == 4
    assert kpis["highly_suitable_count"] == 2
    assert kpis["average_skill_match_score"] == 67.5
    assert kpis["average_experience_years"] == 4.0
    assert kpis["average_skills_count"] == 7.0
