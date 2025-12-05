"""Local training script for the custom BiLSTM + Attention model."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split

from app.ml.models.bilstm_attention import BiLSTMAttentionClassifier
from app.ml.preprocessing.tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


@dataclass
class TrainerConfig:
    train_path: str
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    tokenizer_path: str = "models/tokenizer.pkl"
    output_dir: str = "models/checkpoints"
    model_name: str = "bilstm_attention.pt"
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_length: int = 160
    embedding_dim: int = 300
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    gradient_clip: float = 1.0
    fp16: bool = True
    dataset_name: Optional[str] = None
    run_name: str = "custom-bilstm-attention"
    notes: Optional[str] = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_every: int = 50
    train_split: float = 0.9  # used when val_path is not provided

    def checkpoint_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.model_name)


class TrainingRunTracker:
    """Optional Postgres tracker for recording training runs."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("TRAINING_DB_URL") or os.getenv("DATABASE_URL")
        self._conn = None

    def _connect(self):
        if not self.db_url:
            return None
        if self._conn:
            return self._conn
        try:
            import psycopg2

            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = True
            self._ensure_tables()
        except Exception as exc:  # pragma: no cover - only hits when DB missing
            logger.warning("Postgres tracking disabled: %s", exc)
            self._conn = None
        return self._conn

    def _ensure_tables(self):
        if not self._conn:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                run_name TEXT,
                params JSONB,
                status TEXT,
                dataset_name TEXT,
                train_path TEXT,
                val_path TEXT,
                test_path TEXT,
                best_val_loss FLOAT,
                best_sentiment_acc FLOAT,
                best_emotion_acc FLOAT,
                test_sentiment_acc FLOAT,
                test_emotion_acc FLOAT,
                notes TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id SERIAL PRIMARY KEY,
                training_run_id INTEGER REFERENCES training_runs(id),
                epoch INTEGER,
                phase TEXT DEFAULT 'val',
                train_loss FLOAT,
                val_loss FLOAT,
                sentiment_acc FLOAT,
                emotion_acc FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Backfill for legacy deployments
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS dataset_name TEXT;")
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS train_path TEXT;")
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS val_path TEXT;")
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS test_path TEXT;")
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS test_sentiment_acc FLOAT;")
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS test_emotion_acc FLOAT;")
        cur.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS notes TEXT;")
        cur.execute("ALTER TABLE epoch_metrics ADD COLUMN IF NOT EXISTS phase TEXT DEFAULT 'val';")
        cur.close()

    def start_run(self, name: str, params: Dict) -> Optional[int]:
        conn = self._connect()
        if not conn:
            return None
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO training_runs (
                run_name,
                params,
                status,
                dataset_name,
                train_path,
                val_path,
                test_path,
                notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                name,
                params,
                "running",
                params.get("dataset_name"),
                params.get("train_path"),
                params.get("val_path"),
                params.get("test_path"),
                params.get("notes"),
            ),
        )
        run_id = cur.fetchone()[0]
        cur.close()
        return run_id

    def log_epoch(self, run_id: Optional[int], epoch: int, metrics: Dict, phase: str = "val") -> None:
        if not run_id or not self._connect():
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO epoch_metrics (
                training_run_id,
                epoch,
                phase,
                train_loss,
                val_loss,
                sentiment_acc,
                emotion_acc
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                epoch,
                phase,
                metrics.get("train_loss"),
                metrics.get("val_loss"),
                metrics.get("sentiment_acc"),
                metrics.get("emotion_acc"),
            ),
        )
        cur.close()

    def finish_run(
        self, run_id: Optional[int], status: str, best: Dict, test_metrics: Optional[Dict] = None
    ) -> None:
        if not run_id or not self._connect():
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE training_runs
            SET status=%s,
                best_val_loss=%s,
                best_sentiment_acc=%s,
                best_emotion_acc=%s,
                test_sentiment_acc=%s,
                test_emotion_acc=%s,
                finished_at=NOW()
            WHERE id=%s
            """,
            (
                status,
                best.get("val_loss"),
                best.get("sentiment_acc"),
                best.get("emotion_acc"),
                (test_metrics or {}).get("sentiment_acc") if test_metrics else None,
                (test_metrics or {}).get("emotion_acc") if test_metrics else None,
                run_id,
            ),
        )
        cur.close()


class SentimentDataset(Dataset):
    """Dataset for training the custom model."""

    def __init__(self, df: pd.DataFrame, tokenizer: CustomTokenizer, max_length: int = 160):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        encoded = self.tokenizer.encode(
            row["text"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        sentiment = SENTIMENT_LABELS.index(row["sentiment"])
        emotion = EMOTION_LABELS.index(row["emotion"])
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "emotion": torch.tensor(emotion, dtype=torch.long),
        }


def _load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataframe from CSV or Parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported dataset format for {path}")


def _prepare_dataloaders(
    config: TrainerConfig, tokenizer: CustomTokenizer
) -> Tuple[DataLoader, DataLoader]:
    df = _load_dataframe(config.train_path)
    df = df.dropna(subset=["text", "sentiment", "emotion"])

    if config.val_path:
        val_df = _load_dataframe(config.val_path).dropna(subset=["text", "sentiment", "emotion"])
    else:
        train_size = int(len(df) * config.train_split)
        val_size = len(df) - train_size
        train_df, val_df = random_split(
            df.to_dict("records"),
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_df = pd.DataFrame(train_df)
        val_df = pd.DataFrame(val_df)

    train_ds = SentimentDataset(train_df, tokenizer, max_length=config.max_length)
    val_ds = SentimentDataset(val_df, tokenizer, max_length=config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def _loader_from_path(
    path: str, tokenizer: CustomTokenizer, config: TrainerConfig, shuffle: bool = False
) -> DataLoader:
    """Build a dataloader from a prepared CSV/Parquet path."""
    df = _load_dataframe(path).dropna(subset=["text", "sentiment", "emotion"])
    dataset = SentimentDataset(df, tokenizer, max_length=config.max_length)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=2)


def _build_model(config: TrainerConfig, vocab_size: int) -> BiLSTMAttentionClassifier:
    return BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _evaluate(
    model: BiLSTMAttentionClassifier,
    data_loader: DataLoader,
    device: torch.device,
    sentiment_loss_fn: nn.Module,
    emotion_loss_fn: nn.Module,
    fp16: bool,
) -> Dict[str, float]:
    """Run evaluation loop for validation or test splits."""
    model.eval()
    total_loss = 0.0
    sentiment_correct = 0
    emotion_correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment"].to(device)
            emotion_labels = batch["emotion"].to(device)

            with autocast(enabled=fp16 and device.type == "cuda"):
                sentiment_logits, emotion_logits, _ = model(
                    input_ids, attention_mask=attention_mask
                )
                loss = sentiment_loss_fn(sentiment_logits, sentiment_labels) + 0.5 * emotion_loss_fn(
                    emotion_logits, emotion_labels
                )

            total_loss += loss.item()
            sentiment_correct += (sentiment_logits.argmax(dim=1) == sentiment_labels).sum().item()
            emotion_correct += (emotion_logits.argmax(dim=1) == emotion_labels).sum().item()
            total += sentiment_labels.size(0)

    num_batches = max(1, len(data_loader))
    return {
        "val_loss": total_loss / num_batches,
        "sentiment_acc": sentiment_correct / max(1, total),
        "emotion_acc": emotion_correct / max(1, total),
    }


def train_model(config: TrainerConfig) -> str:
    """Main training loop with optional FP16 and Postgres tracking."""
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(_load_dataframe(config.train_path)["text"].dropna().tolist())
    tokenizer.save(config.tokenizer_path)

    train_loader, val_loader = _prepare_dataloaders(config, tokenizer)

    device = torch.device(config.device)
    model = _build_model(config, tokenizer.vocab_size).to(device)

    scaler = GradScaler(enabled=config.fp16 and device.type == "cuda")

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    sentiment_loss_fn = nn.CrossEntropyLoss()
    emotion_loss_fn = nn.CrossEntropyLoss()

    tracker = TrainingRunTracker()
    run_id = tracker.start_run(config.run_name, vars(config))

    best = {"val_loss": float("inf"), "sentiment_acc": 0.0, "emotion_acc": 0.0}

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment"].to(device)
            emotion_labels = batch["emotion"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=config.fp16 and device.type == "cuda"):
                sentiment_logits, emotion_logits, _ = model(
                    input_ids, attention_mask=attention_mask
                )
                loss = sentiment_loss_fn(
                    sentiment_logits, sentiment_labels
                ) + 0.5 * emotion_loss_fn(emotion_logits, emotion_labels)

            scaler.scale(loss).backward()
            if config.gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if step and step % config.log_every == 0:
                logger.info("Epoch %s Step %s Loss %.4f", epoch + 1, step, loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            sentiment_loss_fn=sentiment_loss_fn,
            emotion_loss_fn=emotion_loss_fn,
            fp16=config.fp16,
        )
        avg_val_loss = val_metrics["val_loss"]
        sentiment_acc = val_metrics["sentiment_acc"]
        emotion_acc = val_metrics["emotion_acc"]

        tracker.log_epoch(
            run_id,
            epoch + 1,
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "sentiment_acc": sentiment_acc,
                "emotion_acc": emotion_acc,
            },
        )

        logger.info(
            "Epoch %s complete. train_loss=%.4f val_loss=%.4f sentiment_acc=%.4f emotion_acc=%.4f",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            sentiment_acc,
            emotion_acc,
        )

        if avg_val_loss < best["val_loss"]:
            best.update(
                {
                    "val_loss": avg_val_loss,
                    "sentiment_acc": sentiment_acc,
                    "emotion_acc": emotion_acc,
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_path": config.tokenizer_path,
                    "config": vars(config),
                },
                config.checkpoint_path(),
            )
            logger.info("Saved new best checkpoint to %s", config.checkpoint_path())

    test_metrics = None
    if config.test_path:
        test_loader = _loader_from_path(config.test_path, tokenizer, config, shuffle=False)
        test_metrics = _evaluate(
            model,
            test_loader,
            device,
            sentiment_loss_fn=sentiment_loss_fn,
            emotion_loss_fn=emotion_loss_fn,
            fp16=config.fp16,
        )
        tracker.log_epoch(
            run_id,
            config.num_epochs + 1,
            {
                "train_loss": None,
                "val_loss": test_metrics["val_loss"],
                "sentiment_acc": test_metrics["sentiment_acc"],
                "emotion_acc": test_metrics["emotion_acc"],
            },
            phase="test",
        )
        logger.info(
            "Test metrics: loss=%.4f sentiment_acc=%.4f emotion_acc=%.4f",
            test_metrics["val_loss"],
            test_metrics["sentiment_acc"],
            test_metrics["emotion_acc"],
        )

    tracker.finish_run(run_id, "completed", best, test_metrics=test_metrics)
    return config.checkpoint_path()
