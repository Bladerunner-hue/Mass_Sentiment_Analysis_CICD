"""Download a Kaggle dataset and export it to Parquet for Spark/PyTorch training."""

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from kaggle import api


def _download_dataset(dataset_ref: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    api.authenticate()
    api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=True)
    return target_dir


def _auto_pick_csv(directory: Path) -> Path:
    csv_files = list(directory.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[0]


def export_kaggle_dataset(
    dataset_ref: str,
    source_file: Path,
    text_field: str,
    sentiment_field: str,
    emotion_field: str,
    default_emotion: str,
    output_path: Path,
    encoding: str = "utf-8",
):
    df = pd.read_csv(source_file, encoding=encoding)

    missing = [col for col in (text_field, sentiment_field) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    if emotion_field not in df.columns:
        df[emotion_field] = default_emotion

    df = df[[text_field, sentiment_field, emotion_field]].rename(
        columns={
            text_field: "text",
            sentiment_field: "sentiment",
            emotion_field: "emotion",
        }
    )
    df = df.dropna(subset=["text", "sentiment", "emotion"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(
        f"Exported {len(df)} rows from Kaggle dataset '{dataset_ref}' "
        f"({source_file.name}) to {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Download and export a Kaggle dataset to Parquet.")
    parser.add_argument(
        "--dataset", required=True, help="Kaggle dataset ref (e.g., kazanova/sentiment140)"
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Specific CSV file inside the dataset. If omitted, the first CSV found is used.",
    )
    parser.add_argument("--text-field", default="text", help="Column containing text")
    parser.add_argument(
        "--sentiment-field", default="sentiment", help="Column containing sentiment labels"
    )
    parser.add_argument(
        "--emotion-field",
        default="emotion",
        help="Column containing emotion labels or will be created",
    )
    parser.add_argument(
        "--default-emotion", default="neutral", help="Emotion value to fill when missing"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Parquet path (default: data/raw/kaggle/<dataset>.parquet)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding (default utf-8; sentiment140 often uses latin-1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_safe = args.dataset.replace("/", "_")
    output_path = Path(args.output or f"data/raw/kaggle/{dataset_safe}.parquet")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        download_dir = _download_dataset(args.dataset, tmp_path)

        if args.file:
            source_file = download_dir / args.file
            if not source_file.exists():
                raise FileNotFoundError(f"Specified file not found: {source_file}")
        else:
            source_file = _auto_pick_csv(download_dir)

        export_kaggle_dataset(
            dataset_ref=args.dataset,
            source_file=source_file,
            text_field=args.text_field,
            sentiment_field=args.sentiment_field,
            emotion_field=args.emotion_field,
            default_emotion=args.default_emotion,
            output_path=output_path,
            encoding=args.encoding,
        )

    # Clean up any cached downloads to keep repo lean
    if download_dir.exists():
        shutil.rmtree(download_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
