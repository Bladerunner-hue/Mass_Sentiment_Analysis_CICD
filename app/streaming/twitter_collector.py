"""Lightweight Twitter Streaming collector writing Parquet micro-batches."""

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


class TwitterStreamCollector:
    """Collect tweets with Tweepy StreamingClient and flush to Parquet."""

    def __init__(
        self,
        bearer_token: str,
        keywords: List[str],
        output_dir: str = "data/raw/twitter_stream",
        flush_interval: int = 60,
        buffer_limit: int = 1000,
    ):
        self.bearer_token = bearer_token
        self.keywords = keywords
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self.buffer_limit = buffer_limit
        self.buffer = queue.Queue(maxsize=buffer_limit)
        self._stop_event = threading.Event()
        self._listener = None

    def _make_partition_path(self) -> Path:
        now = datetime.utcnow()
        return self.output_dir / f"date={now.date()}" / f"hour={now.hour:02d}"

    def _flush_buffer(self):
        items = []
        while not self.buffer.empty():
            items.append(self.buffer.get())
        if not items:
            return
        df = pd.DataFrame(items)
        part_dir = self._make_partition_path()
        part_dir.mkdir(parents=True, exist_ok=True)
        part_path = part_dir / f"part-{int(time.time())}.parquet"
        df.to_parquet(part_path, index=False)
        print(f"Flushed {len(df)} tweets to {part_path}")

    def _flush_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self._flush_buffer()

    def _build_listener(self):
        try:
            import tweepy
        except ImportError as exc:
            raise RuntimeError("tweepy is required for TwitterStreamCollector") from exc

        collector = self

        class Listener(tweepy.StreamingClient):
            def on_connect(self_inner):
                print("Connected to Twitter stream.")

            def on_tweet(self_inner, tweet):
                payload = {
                    "tweet_id": tweet.id,
                    "text": tweet.text,
                    "created_at": getattr(tweet, "created_at", None),
                    "author_id": getattr(tweet, "author_id", None),
                    "lang": getattr(tweet, "lang", None),
                    "collected_at": datetime.utcnow().isoformat(),
                }
                try:
                    collector.buffer.put_nowait(payload)
                except queue.Full:
                    collector._flush_buffer()
                    collector.buffer.put_nowait(payload)

            def on_errors(self_inner, errors):
                print(f"Streaming error: {errors}")

            def on_connection_error(self_inner):
                print("Connection error; attempting restart in 15s.")
                time.sleep(15)
                self_inner.disconnect()

        listener = Listener(self.bearer_token, wait_on_rate_limit=True)
        # Reset existing rules
        existing = listener.get_rules().data or []
        if existing:
            listener.delete_rules([r.id for r in existing])
        # Add keyword rules
        listener.add_rules([tweepy.StreamRule(k) for k in self.keywords])
        return listener

    def start(self):
        self._listener = self._build_listener()
        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()
        try:
            self._listener.filter(tweet_fields=["created_at", "lang", "author_id"], threaded=False)
        except KeyboardInterrupt:
            print("Stopping stream...")
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        if self._listener:
            try:
                self._listener.disconnect()
            except Exception:
                pass
        self._flush_buffer()
