"""Tests for ML training pipeline."""

import pandas as pd

from src.ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from src.ml.train_model import generate_synthetic_dataset, train_candidate_ranker


def test_generate_synthetic_dataset_has_expected_columns_and_size():
    df = generate_synthetic_dataset(n_samples=600, random_state=42)

    assert len(df) == 600
    expected_columns = {"candidate_id", *FEATURE_COLUMNS, TARGET_COLUMN}
    assert expected_columns.issubset(set(df.columns))


def test_generate_synthetic_dataset_labels_are_valid_categories():
    df = generate_synthetic_dataset(n_samples=500, random_state=7)
    valid_labels = {"Highly Suitable", "Suitable", "Less Suitable"}
    assert set(df[TARGET_COLUMN].unique()).issubset(valid_labels)


def test_train_candidate_ranker_returns_metrics():
    df = generate_synthetic_dataset(n_samples=500, random_state=21)
    result = train_candidate_ranker(df, test_size=0.25, random_state=21)

    assert "model" in result
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["classification_report"] is not None
    assert result["confusion_matrix"].shape == (3, 3)


def test_synthetic_data_has_realistic_ranges():
    df = generate_synthetic_dataset(n_samples=700, random_state=99)

    assert df["skill_match_score"].between(0, 100).all()
    assert df["experience_years"].between(0, 25).all()
    assert (df["matched_skills_count"] <= df["skills_count"]).all()
