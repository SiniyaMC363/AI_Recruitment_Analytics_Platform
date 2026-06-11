"""Feature engineering for candidate–job matching.

Phase 3: Add TF-IDF, skill overlap, and semantic similarity features.
"""

import pandas as pd


def build_feature_matrix(
    candidates: pd.DataFrame,
    job_postings: pd.DataFrame,
) -> pd.DataFrame:
    """Build a feature matrix for candidate–job pairs.

    Args:
        candidates: Candidate records with resume text.
        job_postings: Job posting records with descriptions.

    Returns:
        Feature matrix placeholder. Full implementation in Phase 3.

    Raises:
        NotImplementedError: Feature engineering is planned for Phase 3.
    """
    raise NotImplementedError(
        "Feature engineering will be implemented in Phase 3."
    )
