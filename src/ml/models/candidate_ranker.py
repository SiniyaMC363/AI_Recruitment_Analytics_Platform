"""Candidate ranking model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.predictor import (
    CandidatePrediction,
    load_model_artifact,
    predict_candidate,
    predict_from_inputs,
)
from src.ml.train_model import save_model_artifact, train_candidate_ranker


class CandidateRanker:
    """High-level API for training and inference."""

    def __init__(self, model: Any = None, artifact: dict[str, Any] | None = None) -> None:
        self.model = model
        self.artifact = artifact

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "CandidateRanker":
        """Train the ranking model on provided features and labels."""
        dataset = features.copy()
        dataset["candidate_rating"] = labels
        result = train_candidate_ranker(dataset)
        self.model = result["model"]
        self.artifact = result
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict suitability categories for a feature matrix."""
        if self.model is None:
            artifact = load_model_artifact()
            self.model = artifact["model"]
        return pd.Series(self.model.predict(features))

    def predict_candidate(
        self,
        candidate_name: str,
        features: dict[str, Any],
        model_path: Path | None = None,
    ) -> CandidatePrediction:
        """Predict a single candidate with confidence and explanation."""
        return predict_candidate(
            candidate_name=candidate_name,
            features=features,
            model_path=model_path,
        )

    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""
        if self.artifact is None:
            raise ValueError("No trained artifact available to save.")
        save_model_artifact(self.artifact, model_path=path)

    @classmethod
    def load(cls, path: Path) -> "CandidateRanker":
        """Load a saved model from disk."""
        artifact = load_model_artifact(model_path=path)
        return cls(model=artifact["model"], artifact=artifact)

    @classmethod
    def predict_from_inputs(
        cls,
        candidate_name: str,
        skill_match_percentage: float,
        matched_skills_count: int,
        skills_count: int,
        experience_years: float,
        education_level: str = "Bachelor",
        model_path: Path | None = None,
    ) -> CandidatePrediction:
        """Convenience method for recruiter-friendly inputs."""
        return predict_from_inputs(
            candidate_name=candidate_name,
            skill_match_percentage=skill_match_percentage,
            matched_skills_count=matched_skills_count,
            skills_count=skills_count,
            experience_years=experience_years,
            education_level=education_level,
            model_path=model_path,
        )
