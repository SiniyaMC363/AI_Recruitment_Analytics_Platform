"""Model training pipelines.

Phase 3: Implement training, evaluation, and model persistence.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings


def train_ranker(
    features: pd.DataFrame,
    labels: pd.Series,
    model_path: Path | None = None,
) -> Any:
    """Train a candidate ranking model.

    Args:
        features: Training feature matrix.
        labels: Target labels (e.g., hire/reject or relevance scores).
        model_path: Optional path to save the trained model.

    Returns:
        Trained model object.

    Raises:
        NotImplementedError: Training pipeline is planned for Phase 3.
    """
    _ = (features, labels, model_path or settings.models_dir)
    raise NotImplementedError("Model training will be implemented in Phase 3.")
