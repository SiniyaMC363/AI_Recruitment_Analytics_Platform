"""Tests for data loading utilities."""

import pandas as pd
import pytest

from src.data.loaders import _validate_candidates, _validate_job_postings


def test_validate_candidates_passes_with_required_columns():
    df = pd.DataFrame(
        {
            "candidate_id": [1],
            "name": ["Alice"],
            "resume_text": ["Python developer with ML experience."],
        }
    )
    _validate_candidates(df)


def test_validate_candidates_raises_on_missing_columns():
    df = pd.DataFrame({"candidate_id": [1], "name": ["Alice"]})
    with pytest.raises(ValueError, match="missing columns"):
        _validate_candidates(df)


def test_validate_job_postings_passes_with_required_columns():
    df = pd.DataFrame(
        {
            "job_id": [1],
            "title": ["Data Scientist"],
            "description": ["Looking for a Python and ML expert."],
        }
    )
    _validate_job_postings(df)
