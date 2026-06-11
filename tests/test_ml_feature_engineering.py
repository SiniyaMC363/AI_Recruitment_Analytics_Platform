"""Tests for ML feature engineering."""

import pandas as pd
import pytest

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    build_candidate_features,
    build_features_from_skill_match,
    education_level_to_score,
    features_to_dataframe,
)


def test_education_level_to_score_maps_known_levels():
    assert education_level_to_score("PhD") == 1.0
    assert education_level_to_score("Bachelor") == 0.65
    assert education_level_to_score("High School") == 0.25


def test_education_level_to_score_accepts_numeric_values():
    assert education_level_to_score(0.9) == 0.9
    assert education_level_to_score(1.5) == 1.0


def test_build_candidate_features_clamps_and_validates_values():
    features = build_candidate_features(
        skill_match_percentage=150,
        matched_skills_count=8,
        skills_count=5,
        experience_years=-2,
        education_level="Master",
    )

    assert features.skill_match_score == 100.0
    assert features.matched_skills_count == 8
    assert features.skills_count == 8
    assert features.experience_years == 0.0
    assert features.education_score == 0.85


def test_build_features_from_skill_match_uses_nlp_outputs():
    features = build_features_from_skill_match(
        match_percentage=80.0,
        matched_skills=["Python", "SQL"],
        extracted_skills=["Python", "SQL", "Git"],
        experience_years=4.0,
        education_level="Bachelor",
    )

    assert features.skill_match_score == 80.0
    assert features.matched_skills_count == 2
    assert features.skills_count == 3


def test_features_to_dataframe_returns_model_columns_only():
    features = build_candidate_features(
        skill_match_percentage=70,
        matched_skills_count=3,
        skills_count=6,
        experience_years=2.5,
        education_level="Bachelor",
    )
    df = features_to_dataframe(features)

    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_features_to_dataframe_raises_for_missing_columns():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        features_to_dataframe({"skill_match_score": 50})
