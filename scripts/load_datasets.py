"""Load and merge static datasets into standardized Parquet splits.

This script expects Parquet files with columns at least: text, sentiment, emotion.
It will:
- Read all Parquet files under data/raw/hf, data/raw/kaggle, and data/raw/static_datasets
- Standardize columns, drop empties, deduplicate by text hash
- Stratified split into train/val/test (70/15/15 by default)
- Write to data/processed/<split>/part-*.parquet and metadata.json
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _collect_parquet(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        for file in path.glob("**/*.parquet"):
            frames.append(pd.read_parquet(file))
    if not frames:
        return pd.DataFrame(columns=["text", "sentiment", "emotion", "source"])
    df = pd.concat(frames, ignore_index=True)
    # Ensure required columns exist
    for col in ("text", "sentiment", "emotion"):
        if col not in df.columns:
            df[col] = None
    if "source" not in df.columns:
        df["source"] = "static"
    return df[["text", "sentiment", "emotion", "source"]]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df["emotion"] = df["emotion"].astype(str).str.lower()
    df = df.dropna(subset=["text", "sentiment"])
    df = df[df["text"].str.len() > 3]
    df["text_hash"] = df["text"].apply(_hash_text)
    df = df.drop_duplicates(subset=["text_hash"])
    return df


def _stratified_split(
    df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Simple stratified split on sentiment; fall back to random if not enough
    train_parts = []
    val_parts = []
    test_parts = []
    for sentiment, group in df.groupby("sentiment"):
        n = len(group)
        if n < 10:
            train_parts.append(group)
            continue
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        shuffled = group.sample(frac=1.0, random_state=42)
        train_parts.append(shuffled.iloc[:train_end])
        val_parts.append(shuffled.iloc[train_end:val_end])
        test_parts.append(shuffled.iloc[val_end:])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train_df, val_df, test_df


def _write_split(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    df.to_parquet(out_dir / "part-00000.parquet", index=False)


def _write_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_root: Path,
    sources_count,
):
    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "sources": sources_count,
        "sentiment_distribution": train_df["sentiment"].value_counts().to_dict(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Merge static datasets into train/val/test splits."
    )
    parser.add_argument(
        "--raw-root", default="data/raw", help="Root directory containing hf/kaggle/static_datasets"
    )
    parser.add_argument(
        "--processed-root", default="data/processed", help="Output directory for processed splits"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)

    source_paths = [
        raw_root / "hf",
        raw_root / "kaggle",
        raw_root / "static_datasets",
    ]

    df = _collect_parquet(source_paths)
    if df.empty:
        print("No Parquet datasets found under data/raw/{hf,kaggle,static_datasets}.")
        return

    sources_count = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    df = _standardize(df)

    train_df, val_df, test_df = _stratified_split(df, args.train_ratio, args.val_ratio)

    _write_split(train_df, processed_root / "train")
    _write_split(val_df, processed_root / "val")
    _write_split(test_df, processed_root / "test")
    _write_metadata(train_df, val_df, test_df, processed_root, sources_count)

    print(
        f"Processed datasets -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)} "
        f"from {len(df)} total samples"
    )


if __name__ == "__main__":
    main()
