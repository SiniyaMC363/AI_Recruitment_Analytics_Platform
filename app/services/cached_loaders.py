"""Streamlit-cached wrappers for data and model loading."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.loaders import load_candidates, load_job_postings
from src.ml.predictor import load_model_artifact, model_is_available


@st.cache_data(show_spinner="Loading candidate data...")
def load_candidates_cached(filepath: str | None = None):
    """Load candidates CSV with Streamlit caching."""
    return load_candidates(filepath)


@st.cache_data(show_spinner="Loading job postings...")
def load_job_postings_cached(filepath: str | None = None):
    """Load job postings CSV with Streamlit caching."""
    return load_job_postings(filepath)


@st.cache_resource(show_spinner="Loading ML model...")
def load_model_cached(model_path: str | None = None) -> dict:
    """Load the trained model artifact with Streamlit caching."""
    path = Path(model_path) if model_path else None
    return load_model_artifact(path)
