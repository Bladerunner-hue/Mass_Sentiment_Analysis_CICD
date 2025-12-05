"""Orchestration helpers for dataset prep + model training/test runs."""

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.ml.training import TrainerConfig, train_model
from app.ml.training.train import _load_dataframe
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

# Minimal mapping to derive sentiment from emotion labels
EMOTION_TO_SENTIMENT = {
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",
    "trust": "positive",
    "neutral": "neutral",
    "anticipation": "positive",
    "sadness": "negative",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
}


class TrainingOrchestrator:
    """Prepare datasets and run training in a background thread."""

    def __init__(self):
        self.dataset_service = DatasetService()
        self.jobs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ utils
    def _normalize_split(
        self,
        path: str,
        text_col: str,
        sentiment_col: Optional[str],
        emotion_col: Optional[str],
        dataset_name: str,
        split: str,
        derive_sentiment: bool,
        default_emotion: str = "neutral",
        emotion_mapping: Optional[Dict[Any, str]] = None,
        sentiment_mapping: Optional[Dict[Any, str]] = None,
        job_tag: Optional[str] = None,
    ) -> str:
        """Convert an arbitrary dataset into our canonical columns."""
        df = _load_dataframe(path)

        if text_col not in df.columns:
            raise ValueError(f"Missing text column '{text_col}' in {path}")

        df["text"] = df[text_col].astype(str)

        # Emotion handling
        if emotion_col and emotion_col in df.columns:
            if emotion_mapping:
                df["emotion"] = df[emotion_col].map(emotion_mapping).fillna(df[emotion_col])
            elif str(df[emotion_col].dtype).startswith(("int", "Int")):
                auto_map = {0: "sadness", 1: "joy", 2: "joy", 3: "anger", 4: "fear", 5: "surprise"}
                df["emotion"] = df[emotion_col].map(auto_map).fillna(df[emotion_col])
            else:
                df["emotion"] = df[emotion_col]
        else:
            df["emotion"] = default_emotion

        # Sentiment handling
        if sentiment_col and sentiment_col in df.columns:
            if sentiment_mapping:
                df["sentiment"] = df[sentiment_col].map(sentiment_mapping).fillna(
                    df[sentiment_col]
                )
            else:
                df["sentiment"] = df[sentiment_col]
        elif derive_sentiment and "emotion" in df.columns:
            df["sentiment"] = df["emotion"].astype(str).str.lower().map(EMOTION_TO_SENTIMENT)
            df["sentiment"] = df["sentiment"].fillna("neutral")
        else:
            raise ValueError(
                "No sentiment column found. Provide one or enable derive_sentiment_from_emotion."
            )

        sentiment_series = df["sentiment"].astype(str).str.lower()
        sentiment_series = sentiment_series.map(
            {"0": "negative", "1": "neutral", "2": "positive", "4": "positive"}
        ).fillna(sentiment_series)
        allowed = {"positive", "negative", "neutral"}
        df["sentiment"] = sentiment_series.apply(lambda x: x if x in allowed else "neutral")
        allowed_emotions = {
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        }
        df["emotion"] = df["emotion"].astype(str).str.lower().apply(
            lambda x: x if x in allowed_emotions else "neutral"
        )
        df = df[["text", "sentiment", "emotion"]].dropna()
        output_dir = Path(os.environ.get("DATA_PROCESSED_DIR", "data/processed/training"))
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"-{job_tag}" if job_tag else ""
        out_path = output_dir / f"{dataset_name}{tag}-{split}.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    # ------------------------------------------------------------------ jobs
    def start_training(
        self,
        *,
        train_path: str,
        text_column: str,
        sentiment_column: Optional[str],
        emotion_column: Optional[str],
        dataset_name: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        derive_sentiment_from_emotion: bool = True,
        hyperparams: Optional[Dict[str, Any]] = None,
        emotion_mapping: Optional[Dict[Any, str]] = None,
        sentiment_mapping: Optional[Dict[Any, str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize data, kick off training, and return a job ID."""
        hyperparams = hyperparams or {}
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "preparing", "error": None, "checkpoint": None}
        job_tag = job_id.split("-")[0]

        def _run():
            try:
                prepared_train = self._normalize_split(
                    train_path,
                    text_column,
                    sentiment_column,
                    emotion_column,
                    dataset_name,
                    "train",
                    derive_sentiment_from_emotion,
                    emotion_mapping=emotion_mapping,
                    sentiment_mapping=sentiment_mapping,
                    job_tag=job_tag,
                )
                prepared_val = (
                    self._normalize_split(
                        val_path,
                        text_column,
                        sentiment_column,
                        emotion_column,
                        dataset_name,
                        "val",
                        derive_sentiment_from_emotion,
                        emotion_mapping=emotion_mapping,
                        sentiment_mapping=sentiment_mapping,
                        job_tag=job_tag,
                    )
                    if val_path
                    else None
                )
                prepared_test = (
                    self._normalize_split(
                        test_path,
                        text_column,
                        sentiment_column,
                        emotion_column,
                        dataset_name,
                        "test",
                        derive_sentiment_from_emotion,
                        emotion_mapping=emotion_mapping,
                        sentiment_mapping=sentiment_mapping,
                        job_tag=job_tag,
                    )
                    if test_path
                    else None
                )

                config_kwargs = {
                    "train_path": prepared_train,
                    "val_path": prepared_val,
                    "test_path": prepared_test,
                    "dataset_name": dataset_name,
                    "run_name": hyperparams.get("run_name", f"{dataset_name}-bilstm"),
                    "notes": notes,
                }
                for field in (
                    "batch_size",
                    "num_epochs",
                    "learning_rate",
                    "weight_decay",
                    "max_length",
                    "embedding_dim",
                    "hidden_dim",
                    "num_layers",
                    "dropout",
                    "gradient_clip",
                    "fp16",
                    "train_split",
                    "output_dir",
                    "model_name",
                ):
                    if field in hyperparams:
                        config_kwargs[field] = hyperparams[field]

                self.jobs[job_id]["status"] = "running"
                config = TrainerConfig(**config_kwargs)
                checkpoint = train_model(config)
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["checkpoint"] = checkpoint
            except Exception as exc:
                logger.exception("Training job %s failed", job_id)
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(exc)

        threading.Thread(target=_run, daemon=True).start()
        return {"job_id": job_id}

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get the current status for a job."""
        return self.jobs.get(job_id, {"status": "unknown"})

    # ---------------------------------------------------------------- history
    def list_local_datasets(self):
        """Expose datasets already downloaded via DatasetService."""
        return self.dataset_service.list_local_datasets()
