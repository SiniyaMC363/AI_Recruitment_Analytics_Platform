"""Machine learning module for candidate ranking and matching."""

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    SUITABILITY_LABELS,
    CandidateFeatures,
    build_candidate_features,
    build_features_from_skill_match,
    features_to_dataframe,
)
from src.ml.models.candidate_ranker import CandidateRanker
from src.ml.predictor import (
    CandidatePrediction,
    explain_prediction,
    load_model_artifact,
    model_is_available,
    predict_candidate,
    predict_from_inputs,
)
from src.ml.train_model import generate_synthetic_dataset, train_and_save, train_ranker

__all__ = [
    "FEATURE_COLUMNS",
    "SUITABILITY_LABELS",
    "CandidateFeatures",
    "CandidatePrediction",
    "CandidateRanker",
    "build_candidate_features",
    "build_features_from_skill_match",
    "explain_prediction",
    "features_to_dataframe",
    "generate_synthetic_dataset",
    "load_model_artifact",
    "model_is_available",
    "predict_candidate",
    "predict_from_inputs",
    "train_and_save",
    "train_ranker",
]
