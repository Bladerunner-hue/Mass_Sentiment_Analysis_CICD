"""Read-only access to training history stored in Postgres."""

import os
from typing import Dict, List

from flask import current_app
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.extensions import db


class TrainingHistoryService:
    """Helper to inspect training_runs and epoch_metrics without ORM models."""

    def __init__(self):
        self.db_url = os.environ.get("TRAINING_DB_URL")
        self.engine: Engine = create_engine(self.db_url) if self.db_url else db.engine
        self._columns_cache: Dict[str, List[str]] = {}

    def _columns(self, table: str) -> List[str]:
        if table in self._columns_cache:
            return self._columns_cache[table]
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = :table
                        """
                    ),
                    {"table": table},
                )
                cols = [r[0] for r in rows]
                self._columns_cache[table] = cols
                return cols
        except Exception as exc:  # pragma: no cover - diagnostics only
            current_app.logger.warning("Unable to inspect columns for %s: %s", table, exc)
            return []

    def _table_exists(self, table: str) -> bool:
        return bool(self._columns(table))

    def fetch_runs(self, limit: int = 25) -> List[Dict]:
        if not self._table_exists("training_runs"):
            return []

        desired = [
            "id",
            "run_name",
            "dataset_name",
            "status",
            "best_val_loss",
            "best_sentiment_acc",
            "best_emotion_acc",
            "test_sentiment_acc",
            "test_emotion_acc",
            "started_at",
            "finished_at",
            "notes",
        ]
        available = [c for c in desired if c in self._columns("training_runs")]
        if not available:
            return []

        order_col = "started_at" if "started_at" in available else "id"
        query = f"SELECT {', '.join(available)} FROM training_runs ORDER BY {order_col} DESC LIMIT :limit"
        with self.engine.connect() as conn:
            rows = conn.execute(text(query), {"limit": limit}).mappings().all()
            return [dict(r) for r in rows]

    def fetch_epoch_metrics(self, run_id: int) -> List[Dict]:
        if not self._table_exists("epoch_metrics"):
            return []

        desired = [
            "epoch",
            "phase",
            "train_loss",
            "val_loss",
            "sentiment_acc",
            "emotion_acc",
            "created_at",
        ]
        available = [c for c in desired if c in self._columns("epoch_metrics")]
        if not available:
            return []

        query = f"""
            SELECT {', '.join(available)}
            FROM epoch_metrics
            WHERE training_run_id = :run_id
            ORDER BY epoch
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(query), {"run_id": run_id}).mappings().all()
            return [dict(r) for r in rows]
