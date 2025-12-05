"""Kick off a quick training run using environment variables and HF/Kaggle helpers."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from app.ml.training import TrainerConfig, train_model
from app.services.dataset_service import DatasetService
from app.services.training_orchestrator import TrainingOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickstart training runner.")
    parser.add_argument("--dataset", default="dair-ai/emotion", help="HF dataset id")
    parser.add_argument("--subset", default=None, help="Optional dataset subset")
    parser.add_argument("--train-split", default="train", help="Train split name")
    parser.add_argument("--val-split", default="validation", help="Validation split name")
    parser.add_argument("--test-split", default="test", help="Test split name")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--sentiment-col", default=None, help="Sentiment column (leave empty to derive)")
    parser.add_argument("--emotion-col", default="label", help="Emotion column name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-length", type=int, default=160, help="Tokenizer max length")
    parser.add_argument("--fp16", action="store_true", help="Force FP16 (overrides USE_FP16 env)")
    parser.add_argument("--notes", default=None, help="Optional run notes")
    parser.add_argument(
        "--train-path",
        default=None,
        help="Existing train file (csv/parquet). Skips download when provided.",
    )
    parser.add_argument(
        "--val-path",
        default=None,
        help="Existing val file (csv/parquet). Skips download when provided.",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Existing test file (csv/parquet). Skips download when provided.",
    )
    return parser.parse_args()


def ensure_split(dataset_service: DatasetService, dataset: str, subset: str, split: str) -> str:
    """Download a split if missing and return the local path."""
    result = dataset_service.download_hf_dataset(dataset_id=dataset, subset=subset, split=split)
    if not result.get("success"):
        raise RuntimeError(f"Failed to download {dataset}:{split} -> {result.get('error')}")
    return result["path"]


def main():
    load_dotenv()
    args = parse_args()

    ds_service = DatasetService()
    orch = TrainingOrchestrator()

    train_path = args.train_path or ensure_split(
        ds_service, args.dataset, args.subset, args.train_split
    )
    val_path = args.val_path
    test_path = args.test_path

    if not val_path and args.val_split:
        val_path = ensure_split(ds_service, args.dataset, args.subset, args.val_split)
    if not test_path and args.test_split:
        try:
            test_path = ensure_split(ds_service, args.dataset, args.subset, args.test_split)
        except Exception:
            test_path = None  # Some datasets lack test split

    # Normalize columns using the orchestrator helper
    job_tag = Path(train_path).stem
    prepared_train = orch._normalize_split(  # pylint: disable=protected-access
        train_path,
        args.text_col,
        args.sentiment_col,
        args.emotion_col,
        args.dataset.replace("/", "_"),
        "train",
        derive_sentiment=True if args.sentiment_col is None else False,
        job_tag=job_tag,
    )
    prepared_val = (
        orch._normalize_split(  # pylint: disable=protected-access
            val_path,
            args.text_col,
            args.sentiment_col,
            args.emotion_col,
            args.dataset.replace("/", "_"),
            "val",
            derive_sentiment=True if args.sentiment_col is None else False,
            job_tag=job_tag,
        )
        if val_path
        else None
    )
    prepared_test = (
        orch._normalize_split(  # pylint: disable=protected-access
            test_path,
            args.text_col,
            args.sentiment_col,
            args.emotion_col,
            args.dataset.replace("/", "_"),
            "test",
            derive_sentiment=True if args.sentiment_col is None else False,
            job_tag=job_tag,
        )
        if test_path
        else None
    )

    config = TrainerConfig(
        train_path=prepared_train,
        val_path=prepared_val,
        test_path=prepared_test,
        dataset_name=args.dataset,
        run_name=f"{args.dataset}-quickstart",
        notes=args.notes,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16=args.fp16 or os.getenv("USE_FP16", "false").lower() == "true",
    )

    checkpoint_path = train_model(config)
    print(f"Training complete. Checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
