"""Local training loop with FP16 support and optional Postgres tracking."""

from .train import TrainerConfig, TrainingRunTracker, train_model  # noqa: F401
