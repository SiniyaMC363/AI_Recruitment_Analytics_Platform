"""Machine learning module for candidate ranking and matching.

Phase 3 module — implement features, training, and inference.
"""

from src.ml.features import build_feature_matrix
from src.ml.training import train_ranker
from src.ml.models.candidate_ranker import CandidateRanker

__all__ = ["build_feature_matrix", "train_ranker", "CandidateRanker"]
