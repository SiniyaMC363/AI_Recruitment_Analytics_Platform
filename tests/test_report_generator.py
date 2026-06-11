"""Tests for analytics report generation and export."""

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.report_generator import (
    analytics_to_csv_string,
    export_analytics_to_csv,
    generate_analytics_report,
)


@pytest.fixture
def sample_analytics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["1", "2"],
            "candidate_name": ["Alice", "Bob"],
            "skill_match_score": [85.0, 55.0],
            "experience_years": [5.0, 2.0],
            "total_skills": [7, 4],
            "education_level": ["Master", "Bachelor"],
            "predicted_suitability_category": ["Highly Suitable", "Suitable"],
        }
    )


def test_export_analytics_to_csv(sample_analytics_df, tmp_path):
    output_path = tmp_path / "report.csv"
    written_path = export_analytics_to_csv(sample_analytics_df, output_path=output_path)

    assert written_path.exists()
    loaded = pd.read_csv(written_path)
    assert len(loaded) == 2
    assert "candidate_name" in loaded.columns


def test_analytics_to_csv_string(sample_analytics_df):
    csv_text = analytics_to_csv_string(sample_analytics_df)

    assert "candidate_name" in csv_text
    assert "Alice" in csv_text
    assert "Bob" in csv_text


def test_generate_analytics_report(sample_analytics_df):
    report = generate_analytics_report(sample_analytics_df)

    assert report["record_count"] == 2
    assert report["status"] == "ready"
    assert "generated_at_utc" in report
    assert report["kpis"]["total_candidates"] == 2
    assert report["kpis"]["highly_suitable_count"] == 1


def test_export_empty_dataframe(tmp_path):
    empty_df = pd.DataFrame()
    output_path = tmp_path / "empty_report.csv"

    export_analytics_to_csv(empty_df, output_path=output_path)
    loaded = pd.read_csv(output_path)

    assert list(loaded.columns) == [
        "candidate_id",
        "candidate_name",
        "skill_match_score",
        "experience_years",
        "total_skills",
        "education_level",
        "predicted_suitability_category",
    ]
    assert loaded.empty
