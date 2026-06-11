"""Tests for analytics data analysis utilities."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.analytics.data_analyzer import (
    ANALYTICS_COLUMNS,
    build_analytics_dataset,
    extract_education_level,
    extract_experience_years,
    filter_analytics_data,
    get_category_distribution,
    get_education_comparison,
    get_experience_distribution,
    get_score_distribution,
    get_top_candidates,
    load_analytics_dataset,
    save_analytics_dataset,
)
from src.ml.feature_engineering import FEATURE_COLUMNS
from src.ml.train_model import generate_synthetic_dataset, train_candidate_ranker


@pytest.fixture
def sample_candidates() -> pd.DataFrame:
    """Minimal candidate records for analytics tests."""
    return pd.DataFrame(
        {
            "candidate_id": ["1", "2"],
            "name": ["Alice Johnson", "Bob Smith"],
            "resume_text": [
                "Senior data scientist with 5 years of experience in Python, SQL, and machine learning.",
                "Software engineer with 3 years of experience in Java and AWS.",
            ],
        }
    )


@pytest.fixture
def sample_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": ["1"],
            "title": ["Data Scientist"],
            "description": [
                "Seeking Python, SQL, and machine learning skills for recruitment analytics."
            ],
        }
    )


@pytest.fixture
def sample_analytics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["1", "2", "3"],
            "candidate_name": ["Alice", "Bob", "Carol"],
            "skill_match_score": [80.0, 55.0, 30.0],
            "experience_years": [5.0, 3.0, 1.0],
            "total_skills": [6, 4, 2],
            "education_level": ["Master", "Bachelor", "Other"],
            "predicted_suitability_category": [
                "Highly Suitable",
                "Suitable",
                "Less Suitable",
            ],
        }
    )


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    """Train a small model for analytics prediction tests."""
    df = generate_synthetic_dataset(n_samples=300, random_state=42)
    result = train_candidate_ranker(df, random_state=42)

    model_path = tmp_path / "candidate_ranker.pkl"
    artifact = {
        "model": result["model"],
        "feature_columns": FEATURE_COLUMNS,
        "classes": ["Highly Suitable", "Suitable", "Less Suitable"],
        "accuracy": result["accuracy"],
    }
    joblib.dump(artifact, model_path)
    return model_path


def test_extract_experience_years():
    assert extract_experience_years("5 years of experience in Python") == 5.0
    assert extract_experience_years("No experience details") == 0.0
    assert extract_experience_years("") == 0.0


def test_extract_education_level():
    assert extract_education_level("PhD in Computer Science") == "PhD"
    assert extract_education_level("Bachelor of Science in IT") == "Bachelor"
    assert extract_education_level("Experienced developer") == "Other"


def test_build_analytics_dataset(sample_candidates, sample_jobs, trained_model_path):
    analytics_df = build_analytics_dataset(
        sample_candidates,
        job_postings=sample_jobs,
        model_path=trained_model_path,
    )

    assert list(analytics_df.columns) == ANALYTICS_COLUMNS
    assert len(analytics_df) == 2
    assert analytics_df.loc[0, "candidate_name"] == "Alice Johnson"
    assert analytics_df["predicted_suitability_category"].isin(
        {"Highly Suitable", "Suitable", "Less Suitable"}
    ).all()


def test_build_analytics_dataset_empty_input():
    result = build_analytics_dataset(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == ANALYTICS_COLUMNS


def test_filter_analytics_data(sample_analytics_df):
    filtered = filter_analytics_data(
        sample_analytics_df,
        categories=["Highly Suitable"],
        min_experience=4.0,
        min_skill_match=70.0,
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["candidate_name"] == "Alice"


def test_distribution_helpers(sample_analytics_df):
    category_df = get_category_distribution(sample_analytics_df)
    assert category_df["count"].sum() == 3

    score_df = get_score_distribution(sample_analytics_df, bins=3)
    assert score_df["count"].sum() == 3

    experience_df = get_experience_distribution(sample_analytics_df, bins=3)
    assert experience_df["count"].sum() == 3

    education_df = get_education_comparison(sample_analytics_df)
    assert "avg_match_score" in education_df.columns
    assert len(education_df) >= 1


def test_get_top_candidates(sample_analytics_df):
    top = get_top_candidates(sample_analytics_df, top_n=2)

    assert len(top) == 2
    assert top.iloc[0]["candidate_name"] == "Alice"
    assert top.iloc[0]["skill_match_score"] == 80.0


def test_save_and_load_analytics_dataset(sample_analytics_df, tmp_path):
    output_path = tmp_path / "candidate_analytics.csv"
    save_analytics_dataset(sample_analytics_df, output_path=output_path)

    loaded = load_analytics_dataset(dataset_path=output_path)
    assert len(loaded) == len(sample_analytics_df)
    assert list(loaded.columns) == ANALYTICS_COLUMNS
