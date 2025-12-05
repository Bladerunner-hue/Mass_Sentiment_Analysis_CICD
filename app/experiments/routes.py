"""UI + APIs for training experiments."""

import os
from pathlib import Path
from typing import Dict

from flask import current_app, jsonify, render_template, request
from flask_login import login_required

from app.experiments import bp
from app.services.dataset_service import DatasetService
from app.services.training_history_service import TrainingHistoryService
from app.services.training_orchestrator import TrainingOrchestrator


def get_orchestrator() -> TrainingOrchestrator:
    """Lazy init of the orchestrator singleton."""
    if not hasattr(current_app, "_training_orchestrator"):
        current_app._training_orchestrator = TrainingOrchestrator()
    return current_app._training_orchestrator


def get_history_service() -> TrainingHistoryService:
    """Lazy init of the history reader."""
    if not hasattr(current_app, "_training_history_service"):
        current_app._training_history_service = TrainingHistoryService()
    return current_app._training_history_service


def get_dataset_service() -> DatasetService:
    """Lazy init of dataset service."""
    if not hasattr(current_app, "_training_dataset_service"):
        current_app._training_dataset_service = DatasetService()
    return current_app._training_dataset_service


@bp.route("/")
@login_required
def index():
    """Notebook-style training playground."""
    return render_template("experiments/index.html")


@bp.route("/api/context")
@login_required
def context():
    """Context for UI: datasets + running jobs + quickstarts."""
    orch = get_orchestrator()
    datasets = orch.list_local_datasets()
    jobs = orch.jobs
    quickstart = {
        "name": "🤗 Emotion (HF)",
        "dataset_id": "dair-ai/emotion",
        "text_column": "text",
        "emotion_column": "label",
        "sentiment_column": None,
        "derive_sentiment_from_emotion": True,
        "notes": "Uses emotion labels to derive sentiment automatically.",
        "suggested_splits": {"train": "train", "val": "validation", "test": "test"},
    }
    return jsonify({"datasets": datasets, "jobs": jobs, "quickstart": quickstart})


@bp.route("/api/download", methods=["POST"])
@login_required
def download_dataset():
    """Download a dataset for training via DatasetService."""
    payload: Dict = request.get_json() or {}
    source = payload.get("source", "huggingface")
    dataset_id = payload.get("dataset_id")
    subset = payload.get("subset")
    split = payload.get("split", "train")

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400

    ds_service = get_dataset_service()
    if source == "huggingface":
        result = ds_service.download_hf_dataset(dataset_id=dataset_id, subset=subset, split=split)
    elif source == "kaggle":
        result = ds_service.download_kaggle_dataset(dataset_id=dataset_id)
    else:
        return jsonify({"error": f"Unsupported source: {source}"}), 400

    return jsonify(result)


@bp.route("/api/train", methods=["POST"])
@login_required
def start_training():
    """Normalize dataset columns + kick off a background training job."""
    data: Dict = request.get_json() or {}
    train_path = data.get("train_path")
    if not train_path:
        return jsonify({"error": "train_path is required"}), 400

    text_column = data.get("text_column") or "text"
    sentiment_column = data.get("sentiment_column")
    emotion_column = data.get("emotion_column")
    dataset_name = data.get("dataset_name") or Path(train_path).stem
    derive_sentiment = data.get("derive_sentiment_from_emotion", True)
    val_path = data.get("val_path")
    test_path = data.get("test_path")
    notes = data.get("notes")
    emotion_mapping = data.get("emotion_mapping") or None
    sentiment_mapping = data.get("sentiment_mapping") or None

    orch = get_orchestrator()
    try:
        job = orch.start_training(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            text_column=text_column,
            sentiment_column=sentiment_column,
            emotion_column=emotion_column,
            dataset_name=dataset_name,
            derive_sentiment_from_emotion=derive_sentiment,
            hyperparams=data.get("hyperparams") or {},
            emotion_mapping=emotion_mapping,
            sentiment_mapping=sentiment_mapping,
            notes=notes,
        )
    except Exception as exc:  # pragma: no cover - runtime convenience
        return jsonify({"error": str(exc)}), 400

    return jsonify({"job": job, "message": "Training started"}), 202


@bp.route("/api/jobs/<job_id>")
@login_required
def job_status(job_id):
    """Return current status for a training job (in-memory)."""
    orch = get_orchestrator()
    return jsonify(orch.get_job(job_id))


@bp.route("/api/history")
@login_required
def history():
    """Return completed/ongoing runs with DB-backed history."""
    history_service = get_history_service()
    runs = history_service.fetch_runs(limit=30)

    latest_metrics = []
    if runs:
        latest_metrics = history_service.fetch_epoch_metrics(runs[0]["id"])

    return jsonify({"runs": runs, "latest_metrics": latest_metrics})


@bp.route("/api/runs/<int:run_id>/metrics")
@login_required
def run_metrics(run_id: int):
    """Per-run epoch metrics chart data."""
    history_service = get_history_service()
    metrics = history_service.fetch_epoch_metrics(run_id)
    return jsonify({"metrics": metrics})
