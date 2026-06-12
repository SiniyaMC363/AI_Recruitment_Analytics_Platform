"""Prediction utilities for candidate suitability ranking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from config.settings import settings
from src.ml.feature_engineering import (
    CandidateFeatures,
    FEATURE_COLUMNS,
    build_candidate_features,
    features_to_dataframe,
)
from src.ml.train_model import MODEL_FILENAME


@dataclass
class CandidatePrediction:
    """Prediction output for a single candidate."""

    candidate_name: str
    predicted_category: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float | int]
    explanation: str


def get_default_model_path() -> Path:
    """Return the default saved model path."""
    return settings.models_dir / MODEL_FILENAME


@lru_cache(maxsize=2)
def _load_model_from_path(path_str: str) -> dict[str, Any]:
    """Load and cache a model artifact by path string."""
    return joblib.load(path_str)


def load_model_artifact(model_path: Path | None = None) -> dict[str, Any]:
    """Load a trained model artifact from disk.

    Args:
        model_path: Optional custom model path.

    Returns:
        Dictionary containing model and metadata.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = model_path or get_default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Train the model first: python -m src.ml.train_model"
        )
    return _load_model_from_path(str(path.resolve()))


def explain_prediction(
    features: CandidateFeatures | dict[str, Any],
    predicted_category: str,
    confidence: float,
) -> str:
    """Generate a simple, human-readable explanation for a prediction.

    Args:
        features: Candidate feature values.
        predicted_category: Predicted suitability label.
        confidence: Model confidence percentage.

    Returns:
        Explanation string for recruiters.
    """
    feature_dict = (
        features.to_dict() if isinstance(features, CandidateFeatures) else features
    )

    reasons: list[str] = []

    skill_match = feature_dict.get("skill_match_score", 0)
    matched = feature_dict.get("matched_skills_count", 0)
    experience = feature_dict.get("experience_years", 0)
    education = feature_dict.get("education_score", 0)

    if skill_match >= 75:
        reasons.append(f"strong skill alignment ({skill_match}% match)")
    elif skill_match >= 50:
        reasons.append(f"moderate skill alignment ({skill_match}% match)")
    else:
        reasons.append(f"limited skill alignment ({skill_match}% match)")

    if matched >= 5:
        reasons.append(f"{matched} required skills matched")
    elif matched > 0:
        reasons.append(f"only {matched} required skills matched")
    else:
        reasons.append("no required skills matched")

    if experience >= 5:
        reasons.append(f"solid experience ({experience} years)")
    elif experience >= 2:
        reasons.append(f"some relevant experience ({experience} years)")
    else:
        reasons.append(f"limited experience ({experience} years)")

    if education >= 0.85:
        reasons.append("advanced education profile")
    elif education >= 0.65:
        reasons.append("standard degree-level education")
    else:
        reasons.append("lower education score relative to role")

    joined_reasons = ", ".join(reasons)
    return (
        f"The model classified this candidate as **{predicted_category}** "
        f"with **{confidence:.0f}%** confidence based on {joined_reasons}."
    )


def predict_candidate(
    candidate_name: str,
    features: CandidateFeatures | dict[str, Any],
    model_path: Path | None = None,
) -> CandidatePrediction:
    """Predict candidate suitability from engineered features.

    Args:
        candidate_name: Candidate display name.
        features: CandidateFeatures object or feature dictionary.
        model_path: Optional custom model path.

    Returns:
        CandidatePrediction with label, confidence, and explanation.
    """
    if not candidate_name or not str(candidate_name).strip():
        candidate_name = "Unknown Candidate"

    artifact = load_model_artifact(model_path=model_path)
    model = artifact["model"]

    if isinstance(features, dict):
        feature_frame = features_to_dataframe(features)
        feature_dict = {key: features[key] for key in FEATURE_COLUMNS if key in features}
    else:
        feature_frame = features_to_dataframe(features)
        feature_dict = features.to_dict()

    predicted = model.predict(feature_frame)[0]
    probabilities = model.predict_proba(feature_frame)[0]

    probability_map = {
        str(label): round(float(prob) * 100, 1)
        for label, prob in zip(model.classes_, probabilities)
    }

    # Confidence is the probability assigned to the predicted class.
    confidence = probability_map.get(str(predicted), max(probability_map.values()))

    explanation = explain_prediction(feature_dict, str(predicted), confidence)

    return CandidatePrediction(
        candidate_name=str(candidate_name).strip(),
        predicted_category=str(predicted),
        confidence=confidence,
        probabilities=probability_map,
        features=feature_dict,
        explanation=explanation,
    )


def predict_from_inputs(
    candidate_name: str,
    skill_match_percentage: float,
    matched_skills_count: int,
    skills_count: int,
    experience_years: float,
    education_level: str | float = "Bachelor",
    model_path: Path | None = None,
) -> CandidatePrediction:
    """Predict suitability directly from recruiter-friendly inputs."""
    features = build_candidate_features(
        skill_match_percentage=skill_match_percentage,
        matched_skills_count=matched_skills_count,
        skills_count=skills_count,
        experience_years=experience_years,
        education_level=education_level,
    )
    return predict_candidate(
        candidate_name=candidate_name,
        features=features,
        model_path=model_path,
    )


def model_is_available(model_path: Path | None = None) -> bool:
    """Check whether a trained model artifact exists."""
    path = model_path or get_default_model_path()
    return path.exists()
