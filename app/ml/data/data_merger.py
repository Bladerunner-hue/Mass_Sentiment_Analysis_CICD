"""Merge static and streaming datasets into unified splits."""

from pathlib import Path
from typing import Optional, Tuple
import hashlib

import pandas as pd


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class DataMerger:
    """Combine static Parquet datasets with streaming Parquet drops."""

    def __init__(self, static_root: Path, streaming_root: Optional[Path] = None):
        self.static_root = Path(static_root)
        self.streaming_root = Path(streaming_root) if streaming_root else None

    def _read_all(self, root: Path) -> pd.DataFrame:
        frames = []
        for file in root.glob("**/*.parquet"):
            frames.append(pd.read_parquet(file))
        if not frames:
            return pd.DataFrame(columns=["text", "sentiment", "emotion", "source"])
        df = pd.concat(frames, ignore_index=True)
        if "source" not in df.columns:
            df["source"] = root.name
        return df

    def load(self) -> pd.DataFrame:
        static_df = self._read_all(self.static_root)
        streaming_df = (
            self._read_all(self.streaming_root) if self.streaming_root else pd.DataFrame()
        )
        combined = pd.concat([static_df, streaming_df], ignore_index=True)
        # Minimal cleaning/dedup
        combined["text"] = combined["text"].astype(str).str.strip()
        combined = combined[combined["text"].str.len() > 3]
        combined["text_hash"] = combined["text"].apply(_hash_text)
        combined = combined.drop_duplicates(subset=["text_hash"])
        return combined

    def stratified_split(
        self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, val, test = [], [], []
        for sentiment, group in df.groupby("sentiment"):
            n = len(group)
            if n == 0:
                continue
            shuffled = group.sample(frac=1.0, random_state=42)
            train_end = int(train_ratio * n)
            val_end = int((train_ratio + val_ratio) * n)
            train.append(shuffled.iloc[:train_end])
            val.append(shuffled.iloc[train_end:val_end])
            test.append(shuffled.iloc[val_end:])
        return (
            pd.concat(train, ignore_index=True) if train else pd.DataFrame(),
            pd.concat(val, ignore_index=True) if val else pd.DataFrame(),
            pd.concat(test, ignore_index=True) if test else pd.DataFrame(),
        )

    def save_splits(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_root: Path
    ):
        out_root = Path(out_root)
        for split, df in (("train", train_df), ("val", val_df), ("test", test_df)):
            if df.empty:
                continue
            path = out_root / split
            path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path / "part-00000.parquet", index=False)
