"""Candidate ranking model wrapper.

Phase 3: Wrap scikit-learn or custom ranking models with a consistent API.
"""

from pathlib import Path
from typing import Any

import pandas as pd


class CandidateRanker:
    """Rank candidates against a job posting."""

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "CandidateRanker":
        """Fit the ranking model."""
        raise NotImplementedError("Training will be implemented in Phase 3.")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict relevance scores for candidate–job pairs."""
        raise NotImplementedError("Inference will be implemented in Phase 3.")

    def save(self, path: Path) -> None:
        """Persist the model to disk."""
        raise NotImplementedError("Model persistence will be implemented in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> "CandidateRanker":
        """Load a saved model from disk."""
        raise NotImplementedError("Model loading will be implemented in Phase 3.")
