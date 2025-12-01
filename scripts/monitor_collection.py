"""Monitor status of static and streaming datasets."""

from pathlib import Path

import pandas as pd


def count_parquet_rows(path: Path) -> int:
    total = 0
    for file in path.glob("**/*.parquet"):
        try:
            total += len(pd.read_parquet(file))
        except Exception:
            continue
    return total


def main():
    raw_root = Path("data/raw")
    processed_root = Path("data/processed")

    static_paths = [raw_root / "hf", raw_root / "kaggle", raw_root / "static_datasets"]
    streaming_path = raw_root / "twitter_stream"

    print("=== Data Collection Status ===")
    print("Static datasets:")
    for p in static_paths:
        if p.exists():
            print(f"  {p}: {count_parquet_rows(p)} rows")
        else:
            print(f"  {p}: missing")

    print("\nStreaming:")
    if streaming_path.exists():
        print(f"  {streaming_path}: {count_parquet_rows(streaming_path)} rows")
    else:
        print("  Streaming path missing")

    print("\nProcessed splits:")
    for split in ("train", "val", "test"):
        split_path = processed_root / split
        if split_path.exists():
            print(f"  {split}: {count_parquet_rows(split_path)} rows")
        else:
            print(f"  {split}: missing")


if __name__ == "__main__":
    main()
