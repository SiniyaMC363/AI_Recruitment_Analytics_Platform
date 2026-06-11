"""Tests for ML prediction utilities."""

from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.ml.feature_engineering import FEATURE_COLUMNS, build_candidate_features
from src.ml.predictor import (
    explain_prediction,
    load_model_artifact,
    model_is_available,
    predict_candidate,
    predict_from_inputs,
)
from src.ml.train_model import generate_synthetic_dataset, train_candidate_ranker


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    """Train a small model and save it to a temporary path."""
    df = generate_synthetic_dataset(n_samples=400, random_state=10)
    result = train_candidate_ranker(df, random_state=10)

    model_path = tmp_path / "candidate_ranker.pkl"
    artifact = {
        "model": result["model"],
        "feature_columns": FEATURE_COLUMNS,
        "classes": ["Highly Suitable", "Suitable", "Less Suitable"],
        "accuracy": result["accuracy"],
    }
    joblib.dump(artifact, model_path)
    return model_path


def test_load_model_artifact_returns_model_and_metadata(trained_model_path):
    artifact = load_model_artifact(model_path=trained_model_path)

    assert "model" in artifact
    assert "feature_columns" in artifact
    assert isinstance(artifact["model"], RandomForestClassifier)


def test_predict_candidate_returns_category_and_confidence(trained_model_path):
    features = build_candidate_features(
        skill_match_percentage=88,
        matched_skills_count=6,
        skills_count=8,
        experience_years=6,
        education_level="Master",
    )

    prediction = predict_candidate(
        candidate_name="Jane Doe",
        features=features,
        model_path=trained_model_path,
    )

    assert prediction.candidate_name == "Jane Doe"
    assert prediction.predicted_category in {
        "Highly Suitable",
        "Suitable",
        "Less Suitable",
    }
    assert 0 <= prediction.confidence <= 100
    assert prediction.explanation
    assert len(prediction.probabilities) == 3


def test_predict_from_inputs_wraps_feature_builder(trained_model_path):
    prediction = predict_from_inputs(
        candidate_name="John Doe",
        skill_match_percentage=55,
        matched_skills_count=2,
        skills_count=5,
        experience_years=1.5,
        education_level="Bachelor",
        model_path=trained_model_path,
    )

    assert prediction.candidate_name == "John Doe"
    assert prediction.predicted_category in {
        "Highly Suitable",
        "Suitable",
        "Less Suitable",
    }


def test_explain_prediction_returns_readable_text():
    features = build_candidate_features(
        skill_match_percentage=72,
        matched_skills_count=4,
        skills_count=7,
        experience_years=3,
        education_level="Bachelor",
    )
    explanation = explain_prediction(features, "Suitable", 81.0)

    assert "Suitable" in explanation
    assert "81%" in explanation


def test_model_is_available_checks_file_existence(trained_model_path):
    assert model_is_available(model_path=trained_model_path) is True
    assert model_is_available(model_path=trained_model_path.parent / "missing.pkl") is False


def test_load_model_artifact_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_model_artifact(model_path=tmp_path / "missing.pkl")
