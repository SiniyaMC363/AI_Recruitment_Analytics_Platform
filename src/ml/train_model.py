"""Training pipeline for the candidate ranking classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config.settings import settings
from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    SUITABILITY_LABELS,
    TARGET_COLUMN,
)

MODEL_FILENAME = "candidate_ranker.pkl"
DATASET_FILENAME = "candidate_training_data.csv"
DEFAULT_RANDOM_STATE = 42


def generate_synthetic_dataset(
    n_samples: int = 800,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a realistic synthetic recruitment training dataset.

    Labels are derived from feature combinations so the model can learn
    meaningful patterns between skills, experience, and suitability.

    Args:
        n_samples: Number of candidate records to generate (500-1000 recommended).
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with features and candidate_rating target column.
    """
    if n_samples < 100:
        raise ValueError("n_samples should be at least 100 for stable training.")

    rng = np.random.default_rng(random_state)

    # Skill match tends to cluster around mid-to-high values in real pipelines.
    skill_match_score = np.round(rng.beta(2.2, 1.8, n_samples) * 100, 1)

    skills_count = rng.integers(2, 16, n_samples)
    matched_skills_count = np.array(
        [
            rng.integers(0, min(max_skills + 1, 12))
            for max_skills in skills_count
        ]
    )

    # Keep matched skills logically bounded by total skills.
    matched_skills_count = np.minimum(matched_skills_count, skills_count)

    # Experience follows a right-skewed distribution (many junior candidates).
    experience_years = np.round(rng.gamma(shape=2.2, scale=1.8, size=n_samples), 1)
    experience_years = np.clip(experience_years, 0.0, 25.0)

    education_choices = [0.25, 0.45, 0.65, 0.85, 1.0]
    education_probs = [0.08, 0.12, 0.45, 0.25, 0.10]
    education_score = rng.choice(education_choices, size=n_samples, p=education_probs)

    suitability_score = (
        skill_match_score * 0.45
        + matched_skills_count * 6.0
        + np.minimum(skills_count, 10) * 1.5
        + experience_years * 2.0
        + education_score * 20.0
    )

    # Add noise so classes overlap slightly (more realistic).
    suitability_score += rng.normal(0, 8, n_samples)

    labels = np.empty(n_samples, dtype=object)
    labels[suitability_score >= 105] = "Highly Suitable"
    labels[(suitability_score >= 70) & (suitability_score < 105)] = "Suitable"
    labels[suitability_score < 70] = "Less Suitable"

    candidate_ids = [f"CAND-{i:04d}" for i in range(1, n_samples + 1)]

    return pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "skill_match_score": skill_match_score,
            "matched_skills_count": matched_skills_count,
            "skills_count": skills_count,
            "experience_years": experience_years,
            "education_score": np.round(education_score, 2),
            TARGET_COLUMN: labels,
        }
    )


def save_training_dataset(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Save the training dataset to data/processed/."""
    settings.ensure_directories()
    path = output_path or settings.data_processed_dir / DATASET_FILENAME
    df.to_csv(path, index=False)
    return path


def load_training_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load the processed training dataset."""
    path = dataset_path or settings.data_processed_dir / DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {path}. "
            "Run `python -m src.ml.train_model` to generate and train."
        )
    return pd.read_csv(path)


def train_candidate_ranker(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train a Random Forest classifier and evaluate performance.

    Args:
        df: Training dataframe with feature and target columns.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for train/test split and model.

    Returns:
        Dictionary with model, metrics, and evaluation artifacts.
    """
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing feature columns: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset missing target column: {TARGET_COLUMN}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred, labels=SUITABILITY_LABELS)

    return {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": SUITABILITY_LABELS,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }


def save_model_artifact(
    training_result: dict[str, Any],
    model_path: Path | None = None,
) -> Path:
    """Persist trained model and metadata with Joblib."""
    settings.ensure_directories()
    path = model_path or settings.models_dir / MODEL_FILENAME

    artifact = {
        "model": training_result["model"],
        "feature_columns": training_result["feature_columns"],
        "classes": training_result["classes"],
        "accuracy": training_result["accuracy"],
    }
    joblib.dump(artifact, path)
    return path


def print_evaluation_report(training_result: dict[str, Any]) -> None:
    """Display model evaluation metrics in a readable format."""
    print("\n=== Candidate Ranker Training Report ===")
    print(f"Training samples : {training_result['train_size']}")
    print(f"Testing samples  : {training_result['test_size']}")
    print(f"Accuracy         : {training_result['accuracy']:.3f}")

    print("\nClassification Report:")
    report_df = pd.DataFrame(training_result["classification_report"]).T
    print(report_df.round(3).to_string())

    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    matrix_df = pd.DataFrame(
        training_result["confusion_matrix"],
        index=SUITABILITY_LABELS,
        columns=SUITABILITY_LABELS,
    )
    print(matrix_df.to_string())


def train_and_save(
    n_samples: int = 800,
    dataset_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """End-to-end pipeline: generate data, train model, and save artifacts."""
    dataset = generate_synthetic_dataset(n_samples=n_samples)
    dataset_file = save_training_dataset(dataset, output_path=dataset_path)

    training_result = train_candidate_ranker(dataset)
    model_file = save_model_artifact(training_result, model_path=model_path)

    print(f"Dataset saved to: {dataset_file}")
    print(f"Model saved to : {model_file}")
    print_evaluation_report(training_result)

    training_result["dataset_path"] = dataset_file
    training_result["model_path"] = model_file
    return training_result


def train_ranker(
    features: pd.DataFrame,
    labels: pd.Series,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible training wrapper used by older imports."""
    dataset = features.copy()
    dataset[TARGET_COLUMN] = labels
    result = train_candidate_ranker(dataset)
    save_model_artifact(result, model_path=model_path)
    return result


if __name__ == "__main__":
    train_and_save(n_samples=800)
