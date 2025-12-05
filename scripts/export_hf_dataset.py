"""Export a Hugging Face dataset to Parquet for Spark/PyTorch training."""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def export_dataset(
    dataset_name: str,
    split: str,
    text_field: str,
    sentiment_field: str,
    emotion_field: str,
    default_emotion: str,
    output_path: Path,
):
    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()

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
    print(f"Exported {len(df)} rows to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Hugging Face dataset to Parquet.")
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g., carblacac/twitter-sentiment-analysis)"
    )
    parser.add_argument("--split", default="train", help="Dataset split to export (default: train)")
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
        help="Output Parquet path (default: data/raw/hf/<dataset>.parquet)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_safe = args.dataset.replace("/", "_")
    output_path = Path(args.output or f"data/raw/hf/{dataset_safe}.parquet")

    export_dataset(
        dataset_name=args.dataset,
        split=args.split,
        text_field=args.text_field,
        sentiment_field=args.sentiment_field,
        emotion_field=args.emotion_field,
        default_emotion=args.default_emotion,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
