"""Lightweight tokenizer with vocabulary building and batching."""

import pickle
import re
import string
from collections import Counter
from typing import Dict, List, Optional

import torch


class CustomTokenizer:
    """Whitespace tokenizer with basic cleaning and special tokens."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(
        self,
        max_vocab_size: int = 50000,
        min_freq: int = 2,
        lower: bool = True,
        remove_stopwords: bool = False,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.lower = lower
        self.remove_stopwords = remove_stopwords

        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.BOS_TOKEN: self.BOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
        }
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}

        self.stopwords = set()
        if remove_stopwords:
            self.stopwords.update(
                {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "from",
                    "as",
                    "is",
                    "was",
                    "are",
                    "were",
                    "be",
                    "been",
                    "being",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "should",
                    "could",
                    "may",
                    "might",
                    "must",
                    "can",
                }
            )

        self._punct_table = str.maketrans({p: " " for p in string.punctuation if p not in {"!", "?", "."}})

    def clean_text(self, text: str) -> str:
        text = text or ""
        if self.lower:
            text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.translate(self._punct_table)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        text = self.clean_text(text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        counts = Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        # Filter and sort
        filtered = [(w, c) for w, c in counts.items() if c >= self.min_freq]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[: max(0, self.max_vocab_size - len(self.word2idx))]

        for word, _ in filtered:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]

        if truncation and max_length:
            tokens = tokens[:max_length]

        token_ids = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        mask = [1] * len(token_ids)

        if padding and max_length:
            pad_len = max_length - len(token_ids)
            if pad_len > 0:
                token_ids += [self.PAD_IDX] * pad_len
                mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = max(len(self.tokenize(t)) for t in texts)
            if add_special_tokens:
                max_length += 2

        encoded = [
            self.encode(
                text,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
            )
            for text in texts
        ]
        return {
            "input_ids": torch.stack([e["input_ids"] for e in encoded]),
            "attention_mask": torch.stack([e["attention_mask"] for e in encoded]),
        }

    def save(self, path: str) -> None:
        payload = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "lower": self.lower,
            "remove_stopwords": self.remove_stopwords,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "CustomTokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        tokenizer = cls(
            max_vocab_size=data["max_vocab_size"],
            min_freq=data["min_freq"],
            lower=data["lower"],
            remove_stopwords=data["remove_stopwords"],
        )
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = data["idx2word"]
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
