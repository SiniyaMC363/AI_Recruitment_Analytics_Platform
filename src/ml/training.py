"""Backward-compatible training exports."""

from src.ml.train_model import (
    generate_synthetic_dataset,
    load_training_dataset,
    save_training_dataset,
    train_and_save,
    train_candidate_ranker,
    train_ranker,
)

__all__ = [
    "generate_synthetic_dataset",
    "load_training_dataset",
    "save_training_dataset",
    "train_and_save",
    "train_candidate_ranker",
    "train_ranker",
]
