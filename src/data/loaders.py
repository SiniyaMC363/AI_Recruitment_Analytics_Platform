"""Data loaders for recruitment datasets.

Phase 1: CSV-based loading with basic validation.
Future phases will add resume file ingestion and database connectors.
"""

from pathlib import Path

import pandas as pd

from config.settings import settings


def load_candidates(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load candidate records from a CSV file.

    Args:
        filepath: Path to the candidates CSV. Defaults to data/raw/candidates.csv.

    Returns:
        DataFrame with candidate records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath) if filepath else settings.data_raw_dir / "candidates.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {path}. "
            "Place a CSV in data/raw/ or pass an explicit filepath."
        )

    df = pd.read_csv(path)
    _validate_candidates(df)
    return df


def load_job_postings(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load job posting records from a CSV file.

    Args:
        filepath: Path to the job postings CSV. Defaults to data/raw/job_postings.csv.

    Returns:
        DataFrame with job posting records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath) if filepath else settings.data_raw_dir / "job_postings.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Job postings file not found: {path}. "
            "Place a CSV in data/raw/ or pass an explicit filepath."
        )

    df = pd.read_csv(path)
    _validate_job_postings(df)
    return df


def _validate_candidates(df: pd.DataFrame) -> None:
    """Validate required columns in candidate data."""
    required = {"candidate_id", "name", "resume_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidates data missing columns: {sorted(missing)}")


def _validate_job_postings(df: pd.DataFrame) -> None:
    """Validate required columns in job posting data."""
    required = {"job_id", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Job postings data missing columns: {sorted(missing)}")
